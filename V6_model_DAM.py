import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# GATED RESIDUAL NETWORK
# =====================================================

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.gate_final = nn.Linear(hidden_size, output_size * 2)

        if input_size != output_size:
            self.res_proj = nn.Linear(input_size, output_size)
        else:
            self.res_proj = None

        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.lin1(x))
        h = self.lin2(h)
        gated = self.gate_final(h)
        gated = F.glu(gated)

        if self.res_proj is not None:
            x = self.res_proj(x)

        return self.norm(x + self.dropout(gated))


# =====================================================
# VARIABLE SELECTION NETWORK
# =====================================================

class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features, n_hidden, dropout=0.1):
        super().__init__()
        self.n_features = n_features

        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, n_hidden, n_hidden, dropout)
            for _ in range(n_features)
        ])

        self.v_grn = GatedResidualNetwork(n_features, n_hidden, n_features, dropout)

    def forward(self, x):
        batch, seq, n_f = x.shape

        flat_x = x.reshape(batch * seq, n_f)
        weights = F.softmax(self.v_grn(flat_x), dim=-1)
        weights = weights.view(batch, seq, n_f, 1)

        processed = []
        for i in range(self.n_features):
            feat_in = x[:, :, i].unsqueeze(-1)
            processed.append(self.feature_grns[i](feat_in))

        processed = torch.stack(processed, dim=2)
        output = torch.sum(weights * processed, dim=2)
        return output


# =====================================================
# NON CROSSING QUANTILES
# =====================================================

class NonCrossingQuantileHead(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.p50 = nn.Linear(n_in, 1)
        self.delta_low = nn.Sequential(
            nn.Linear(n_in, 1),
            nn.Softplus()
        )
        self.delta_high = nn.Sequential(
            nn.Linear(n_in, 1),
            nn.Softplus()
        )

    def forward(self, x):
        p50 = self.p50(x)
        d_low = self.delta_low(x)
        d_high = self.delta_high(x)

        p10 = p50 - d_low
        p90 = p50 + d_high
        return torch.cat([p10, p50, p90], dim=-1)


# =====================================================
# SERIES DECOMPOSITION
# =====================================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


# =====================================================
# DAM V3
# =====================================================

class DAM_V3(nn.Module):
    """
    DAM V3:
    - Return target model
    - GDAM + DAM features
    - Spike-aware momentum / regime features
    - Per-step cross-attention pooling (replaces mean pooling)
    - Non-crossing quantile output

    CHANGE FROM PREVIOUS VERSION
    -----------------------------
    Mean pooling (`attn_out.mean(dim=1)`) produced ONE static context
    vector broadcast to all 96 decoder steps. That has two problems
    at a 1344-step input window:

      1. Averaging 1344 timesteps dilutes signal (worse than at 240 steps).
      2. A single context vector can't tell decoder step 1 apart from
         decoder step 96 -- both get identical history information.

    Fix: replace the pooling step with cross-attention where each
    future timestep's embedding (from future_vsn) acts as the QUERY,
    and the encoder output (attn_out) acts as KEY/VALUE. Each of the
    96 future steps now pulls its OWN slice of relevant history,
    instead of all 96 steps sharing one averaged vector.
    """

    def __init__(
        self,
        n_past_features=32,
        n_future_features=13,
        n_hidden=256,
        decomp_kernel=97,
        n_cross_attn_heads=8
    ):
        super().__init__()

        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.n_hidden = n_hidden

        self.decomp = SeriesDecomp(decomp_kernel)

        # -------------------------------------------------
        # VSN
        # -------------------------------------------------
        self.past_vsn = VariableSelectionNetwork(
            n_past_features,
            n_hidden // 2
        )

        self.future_vsn = VariableSelectionNetwork(
            n_future_features,
            n_hidden // 2
        )

        # -------------------------------------------------
        # Residual Path
        # -------------------------------------------------
        self.res_cnn = nn.Conv1d(
            n_hidden // 2,
            n_hidden // 2,
            kernel_size=3,
            padding=1
        )

        self.trend_linear = nn.Linear(
            n_hidden // 2,
            n_hidden // 2
        )

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------
        self.encoder_lstm = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=n_hidden * 2,
            num_heads=8,
            batch_first=True
        )

        self.attn_dropout = nn.Dropout(0.2)

        # -------------------------------------------------
        # Cross-attention pooling (NEW)
        # -------------------------------------------------
        # future embeddings are (B, 96, H/2); encoder output is
        # (B, 1344, 2H). Project the future embedding up to 2H so it
        # can act as a query against the encoder's key/value space.
        encoder_dim = n_hidden * 2
        self.query_proj = nn.Linear(n_hidden // 2, encoder_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=n_cross_attn_heads,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(encoder_dim)
        self.cross_attn_dropout = nn.Dropout(0.2)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
        # context per step is now (B, 96, 2H) instead of a repeated
        # single vector, concatenated with the future embedding (H/2)
        context_dim = (n_hidden * 2) + (n_hidden // 2)

        self.decoder = nn.LSTM(
            input_size=context_dim,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )

        self.quantile_head = NonCrossingQuantileHead(n_hidden)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x_past, x_future, return_attn_weights=False):
        # =================================================
        # Feature selection
        # =================================================
        past = self.past_vsn(x_past)        # (B, 1344, H/2)
        future = self.future_vsn(x_future)  # (B, 96, H/2)

        # =================================================
        # Decomposition
        # =================================================
        residual, trend = self.decomp(past)

        residual_feat = F.relu(
            self.res_cnn(residual.permute(0, 2, 1))
        ).permute(0, 2, 1)

        trend_feat = self.trend_linear(trend)

        enc_input = torch.cat(
            [residual_feat, trend_feat],
            dim=-1
        )  # (B, 1344, H)

        # =================================================
        # Encoder
        # =================================================
        enc_out, _ = self.encoder_lstm(enc_input)  # (B, 1344, 2H)

        attn_out, _ = self.self_attention(
            enc_out,
            enc_out,
            enc_out
        )

        attn_out = self.attn_dropout(attn_out)  # (B, 1344, 2H)

        # =================================================
        # Cross-attention pooling (replaces mean pooling)
        # =================================================
        # query: one query vector PER future step, derived from
        # that step's own future-feature embedding.
        query = self.query_proj(future)  # (B, 96, 2H)

        context, cross_attn_weights = self.cross_attention(
            query=query,
            key=attn_out,
            value=attn_out,
            need_weights=return_attn_weights
        )  # context: (B, 96, 2H)

        context = self.cross_attn_norm(query + self.cross_attn_dropout(context))
        # ^ residual + norm around the cross-attention, so the model
        # can fall back toward "just use this step's own future
        # features" if attending over history isn't useful for a
        # given step, instead of being forced to use the attended
        # value outright.

        # =================================================
        # Decoder
        # =================================================
        decoder_input = torch.cat(
            [future, context],
            dim=-1
        )  # (B, 96, 2H + H/2) -- per-step context, not broadcast

        dec_out, _ = self.decoder(decoder_input)
        dec_out = self.dropout(dec_out)

        quantiles = self.quantile_head(dec_out)  # (B, 96, 3)

        if return_attn_weights:
            return quantiles, cross_attn_weights
        return quantiles


# =====================================================
# QUICK SANITY TEST
# =====================================================

if __name__ == "__main__":
    model = DAM_V3(
        n_past_features=32,
        n_future_features=13,
        n_hidden=256
    )

    xp = torch.randn(8, 1344, 32)
    xf = torch.randn(8, 96, 13)

    out = model(xp, xf)

    print("Output shape:", out.shape)
    print("P10 <= P50:", bool((out[:, :, 0] <= out[:, :, 1]).all()))
    print("P50 <= P90:", bool((out[:, :, 1] <= out[:, :, 2]).all()))

    # Confirm per-step context actually differs across future steps
    # (sanity check that we are NOT back to broadcasting one vector)
    out2, weights = model(xp, xf, return_attn_weights=True)
    print("Attn weights shape:", weights.shape)  # (B, 96, 1344)
    step0 = weights[0, 0]
    step50 = weights[0, 50]
    print("Context differs across steps:", not torch.allclose(step0, step50))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
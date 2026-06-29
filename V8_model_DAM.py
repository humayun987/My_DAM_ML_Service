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

        self.res_proj = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.lin1(x))
        h = self.lin2(h)
        gated = F.glu(self.gate_final(h))

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
        weights = F.softmax(self.v_grn(flat_x), dim=-1).view(batch, seq, n_f, 1)

        processed = torch.stack(
            [self.feature_grns[i](x[:, :, i].unsqueeze(-1)) for i in range(n_f)],
            dim=2,
        )

        return torch.sum(weights * processed, dim=2)


# =====================================================
# NON-CROSSING QUANTILE HEAD
# =====================================================

class NonCrossingQuantileHead(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.p50 = nn.Linear(n_in, 1)
        self.delta_low = nn.Sequential(nn.Linear(n_in, 1), nn.Softplus())
        self.delta_high = nn.Sequential(nn.Linear(n_in, 1), nn.Softplus())

    def forward(self, x):
        p50 = self.p50(x)
        p10 = p50 - self.delta_low(x)
        p90 = p50 + self.delta_high(x)
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
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


# =====================================================
# DAM V7 MODEL
# =====================================================

class DAM_V3(nn.Module):
    """
    DAM quantile forecasting model.

    Same architecture as before:
    - VSN
    - Decomposition
    - BiLSTM encoder
    - Self-attention
    - Cross-attention
    - Non-crossing quantile head

    The target is now log-price (scaled), not diff/return.
    """

    def __init__(
        self,
        n_past_features=48,
        n_future_features=32,
        n_hidden=256,
        decomp_kernel=97,
        n_cross_attn_heads=8,
    ):
        super().__init__()

        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.n_hidden = n_hidden

        self.decomp = SeriesDecomp(decomp_kernel)

        # Feature selection
        self.past_vsn = VariableSelectionNetwork(n_past_features, n_hidden // 2)
        self.future_vsn = VariableSelectionNetwork(n_future_features, n_hidden // 2)

        # Residual path
        self.res_cnn = nn.Conv1d(n_hidden // 2, n_hidden // 2, kernel_size=3, padding=1)
        self.trend_linear = nn.Linear(n_hidden // 2, n_hidden // 2)

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=n_hidden * 2,
            num_heads=8,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(0.2)

        # Cross-attention
        encoder_dim = n_hidden * 2
        self.query_proj = nn.Linear(n_hidden // 2, encoder_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=n_cross_attn_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(encoder_dim)
        self.cross_attn_dropout = nn.Dropout(0.2)

        # Decoder
        context_dim = encoder_dim + (n_hidden // 2)

        self.decoder = nn.LSTM(
            input_size=context_dim,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=2,
            dropout=0.2,
        )

        self.quantile_head = NonCrossingQuantileHead(n_hidden)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_past, x_future, return_attn_weights=False):
        past = self.past_vsn(x_past)
        future = self.future_vsn(x_future)

        residual, trend = self.decomp(past)

        residual_feat = F.relu(
            self.res_cnn(residual.permute(0, 2, 1))
        ).permute(0, 2, 1)

        trend_feat = self.trend_linear(trend)

        enc_input = torch.cat([residual_feat, trend_feat], dim=-1)

        enc_out, _ = self.encoder_lstm(enc_input)

        attn_out, _ = self.self_attention(enc_out, enc_out, enc_out)
        attn_out = self.attn_dropout(attn_out)

        query = self.query_proj(future)

        context, cross_attn_weights = self.cross_attention(
            query=query,
            key=attn_out,
            value=attn_out,
            need_weights=return_attn_weights,
        )

        context = self.cross_attn_norm(query + self.cross_attn_dropout(context))

        decoder_input = torch.cat([future, context], dim=-1)
        dec_out, _ = self.decoder(decoder_input)
        dec_out = self.dropout(dec_out)

        quantiles = self.quantile_head(dec_out)

        if return_attn_weights:
            return quantiles, cross_attn_weights
        return quantiles


if __name__ == "__main__":
    model = DAM_V3(n_past_features=48, n_future_features=32, n_hidden=256)

    xp = torch.randn(8, 1344, 48)
    xf = torch.randn(8, 96, 32)

    out = model(xp, xf)
    print("Output shape:", out.shape)
    print("P10 <= P50:", bool((out[:, :, 0] <= out[:, :, 1]).all()))
    print("P50 <= P90:", bool((out[:, :, 1] <= out[:, :, 2]).all()))

    out2, weights = model(xp, xf, return_attn_weights=True)
    print("Attn weights shape:", weights.shape)
    print("Context differs across steps:", not torch.allclose(weights[0, 0], weights[0, 50]))
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")   
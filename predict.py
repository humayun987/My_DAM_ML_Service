"""
predict.py

Inference for DAM V8 model.

Key differences from V6:
  - Model predicts log-price directly (no baseline addition)
  - Inverse transform: scaled log → expm1 → actual price
  - Uses V8 artifacts: best_model_dam_v7.pth, dam_scaler_v7.joblib, dam_log_price_scaler_v7.joblib
"""

import json
import joblib
import torch
import numpy as np
import pandas as pd

from V8_model_DAM import DAM_V3
from V8_data_loader_DAM import PAST_FEATURES, FUTURE_FEATURES, SCALE_FEATURES, BINARY_FEATURES
from V8_build_inputs_from_json_DAM import build_dam_inputs_from_json

# =====================================================
# PATHS  — update these to match your saved artifacts
# =====================================================

MODEL_PATH        = "best_model_dam_v7.pth"
SCALER_PATH       = "dam_scaler_v7.joblib"
TARGET_SCALER_PATH= "dam_log_price_scaler_v7.joblib"

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# =====================================================
# LOAD ARTIFACTS ONCE
# =====================================================

feature_scaler = joblib.load(SCALER_PATH)
target_scaler  = joblib.load(TARGET_SCALER_PATH)

target_mean  = float(target_scaler.mean_[0])
target_scale = float(target_scaler.scale_[0])

model = DAM_V3(
    n_past_features=len(PAST_FEATURES),
    n_future_features=len(FUTURE_FEATURES),
    n_hidden=256,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

print("Model loaded")

# Columns to scale (excludes binary flags)
PAST_SCALE_COLS   = [c for c in PAST_FEATURES   if c in SCALE_FEATURES]
FUTURE_SCALE_COLS = [c for c in FUTURE_FEATURES if c in SCALE_FEATURES]


def _scale_inputs(x_past_df: pd.DataFrame, x_future_df: pd.DataFrame):
    """
    Scale past and future features using the train-fitted feature_scaler.
    Binary features are passed through unchanged.
    """
    x_past_scaled = x_past_df.copy()
    x_past_scaled[PAST_SCALE_COLS] = feature_scaler.transform(
        x_past_df[PAST_SCALE_COLS]
    )

    # Future: build a dummy frame with all SCALE_FEATURES to use the scaler,
    # then pull only the future columns back out
    x_future_scaled = x_future_df.copy()
    dummy = pd.DataFrame(
        np.zeros((len(x_future_df), len(SCALE_FEATURES))),
        columns=SCALE_FEATURES,
    )
    dummy[FUTURE_SCALE_COLS] = x_future_df[FUTURE_SCALE_COLS].values
    dummy_scaled = pd.DataFrame(
        feature_scaler.transform(dummy),
        columns=SCALE_FEATURES,
    )
    x_future_scaled[FUTURE_SCALE_COLS] = dummy_scaled[FUTURE_SCALE_COLS].values

    return x_past_scaled, x_future_scaled


def predict_dam_from_payload(payload: dict) -> dict:
    """
    Main inference entry point.

    Args:
        payload: dict with keys:
            - prediction_date : str  e.g. "2026-06-20"
            - region          : str  e.g. "Telangana"
            - market_data     : list of market records (DAM + GDAM)
            - weather_data    : list of hourly weather records

    Returns:
        dict with keys:
            - region
            - prediction_date
            - forecast : list of 96 dicts with block, P10, P50, P90
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary")

    region          = payload.get("region", "Telangana")
    prediction_date = payload.get("prediction_date")
    if not prediction_date:
        raise ValueError("payload must contain 'prediction_date'")

    # Build raw feature DataFrames
    x_past_df, x_future_df = build_dam_inputs_from_json(
        payload=payload,
        prediction_date=prediction_date,
        region=region,
        save_csv=False,
    )

    # Validate shapes
    if x_past_df.shape != (1344, len(PAST_FEATURES)):
        raise ValueError(
            f"Expected x_past (1344, {len(PAST_FEATURES)}), got {x_past_df.shape}"
        )
    if x_future_df.shape != (96, len(FUTURE_FEATURES)):
        raise ValueError(
            f"Expected x_future (96, {len(FUTURE_FEATURES)}), got {x_future_df.shape}"
        )

    # Scale
    x_past_scaled, x_future_scaled = _scale_inputs(x_past_df, x_future_df)

    # Tensors
    x_past_tensor   = torch.FloatTensor(x_past_scaled.values).unsqueeze(0).to(device)
    x_future_tensor = torch.FloatTensor(x_future_scaled.values).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        pred_q = model(x_past_tensor, x_future_tensor)  # (1, 96, 3)

    pred_q = pred_q.cpu().numpy()[0]  # (96, 3) — scaled log-price

    # Inverse transform: scaled log → log → price
    def inv(arr):
        log_price = arr * target_scale + target_mean
        return np.clip(np.expm1(log_price), 0, 10_000)

    p10 = inv(pred_q[:, 0])
    p50 = inv(pred_q[:, 1])
    p90 = inv(pred_q[:, 2])

    forecast = [
        {
            "block": i + 1,
            "P10":   round(float(p10[i]), 2),
            "P50":   round(float(p50[i]), 2),
            "P90":   round(float(p90[i]), 2),
        }
        for i in range(96)
    ]

    return {
        "region":          region,
        "prediction_date": str(pd.Timestamp(prediction_date).date()),
        "forecast":        forecast,
    }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    with open("json_inputs/payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    output = predict_dam_from_payload(payload)

    with open("dam_forecast.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Forecast saved: dam_forecast.json")
    print(f"First 3 blocks: {output['forecast'][:3]}")
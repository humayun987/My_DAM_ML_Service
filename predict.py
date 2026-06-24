import json
import joblib
import torch
import numpy as np
import pandas as pd

from V6_model_DAM import DAM_V3
from V6_data_loader_DAM import PAST_FEATURES, FUTURE_FEATURES, SCALE_FEATURES
from V6_build_inputs_from_json_DAM import build_dam_inputs_from_json

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = "best_model_dam_v6.pth"
SCALER_PATH = "dam_scaler6.joblib"
TARGET_SCALER_PATH = "dam_return_scaler6.joblib"

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# =====================================================
# LOAD ARTIFACTS ONCE
# =====================================================

feature_scaler = joblib.load(SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

target_mean = float(target_scaler.mean_[0])
target_scale = float(target_scaler.scale_[0])

model = DAM_V3(
    n_past_features=len(PAST_FEATURES),
    n_future_features=len(FUTURE_FEATURES),
    n_hidden=256,
)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded")


def _scale_inputs(x_past_df: pd.DataFrame, x_future_df: pd.DataFrame):
    x_past_scaled = x_past_df.copy()
    past_scale_cols = [c for c in PAST_FEATURES if c in SCALE_FEATURES]
    if past_scale_cols:
        x_past_scaled[past_scale_cols] = feature_scaler.transform(x_past_scaled[past_scale_cols])

    x_future_scaled = x_future_df.copy()
    future_scale_cols = [c for c in FUTURE_FEATURES if c in SCALE_FEATURES]

    dummy = pd.DataFrame(
        np.zeros((len(x_future_df), len(SCALE_FEATURES))),
        columns=SCALE_FEATURES,
    )
    dummy[future_scale_cols] = x_future_df[future_scale_cols]

    dummy_scaled = feature_scaler.transform(dummy)
    dummy_scaled = pd.DataFrame(dummy_scaled, columns=SCALE_FEATURES)
    x_future_scaled[future_scale_cols] = dummy_scaled[future_scale_cols]

    return x_past_scaled, x_future_scaled


def predict_dam_from_payload(payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary")

    region = payload.get("region", "Telangana")
    prediction_date = payload.get("prediction_date")
    if not prediction_date:
        raise ValueError("payload must contain prediction_date")

    x_past_df, x_future_df, baseline_df = build_dam_inputs_from_json(
        payload=payload,
        prediction_date=prediction_date,
        region=region,
        save_csv=False,
    )

    baseline_price = baseline_df["baseline_price"].values
    if len(baseline_price) != 96:
        raise ValueError("baseline must contain exactly 96 rows")

    expected_past_features = len(PAST_FEATURES)
    expected_future_features = len(FUTURE_FEATURES)

    if x_past_df.shape != (1344, expected_past_features):
        raise ValueError(
            f"Expected x_past shape (1344, {expected_past_features}), got {x_past_df.shape}"
        )

    if x_future_df.shape != (96, expected_future_features):
        raise ValueError(
            f"Expected x_future shape (96, {expected_future_features}), got {x_future_df.shape}"
        )

    x_past_scaled, x_future_scaled = _scale_inputs(x_past_df, x_future_df)

    x_past_tensor = torch.FloatTensor(x_past_scaled.values).unsqueeze(0).to(device)
    x_future_tensor = torch.FloatTensor(x_future_scaled.values).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_q = model(x_past_tensor, x_future_tensor)

    pred_q = pred_q.cpu().numpy()[0]  # (96, 3)
    pred_diffs = pred_q * target_scale + target_mean

    p10_diff = pred_diffs[:, 0]
    p50_diff = pred_diffs[:, 1]
    p90_diff = pred_diffs[:, 2]

    p10_price = np.clip(baseline_price + p10_diff, 0, 10000)
    p50_price = np.clip(baseline_price + p50_diff, 0, 10000)
    p90_price = np.clip(baseline_price + p90_diff, 0, 10000)

    forecast_rows = []
    for i in range(96):
        forecast_rows.append({
            "block": i + 1,
            "P10": round(float(p10_price[i]), 2),
            "P50": round(float(p50_price[i]), 2),
            "P90": round(float(p90_price[i]), 2),
        })

    return {
        "region": region,
        "prediction_date": str(pd.Timestamp(prediction_date).date()),
        "forecast": forecast_rows,
    }


if __name__ == "__main__":
    with open("json_inputs/payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    output = predict_dam_from_payload(payload)

    with open("dam_forecast.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Forecast Saved: dam_forecast.json")
    print(output["forecast"][:3])
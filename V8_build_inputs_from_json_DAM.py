# """
# V8_build_inputs_from_json_DAM.py

# Builds x_past and x_future DataFrames from a JSON payload for the V8 model.

# Key differences from V6:
#   - No baseline price needed (model predicts log-price directly)
#   - Uses V8 feature lists and _engineer_features
#   - Future window includes daily regime context from history
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# import holidays as holidays_lib

# from V8_data_loader_DAM import (
#     _engineer_features,
#     _add_daily_context,
#     PAST_STEPS,
#     PAST_FEATURES,
#     FUTURE_FEATURES,
# )

# FUTURE_STEPS = 96


# def _parse_dt(s):
#     return pd.to_datetime(s, errors="coerce")


# def _load_json_payload(payload=None, payload_path=None):
#     if payload is not None:
#         if not isinstance(payload, dict):
#             raise ValueError("payload must be a dictionary")
#         if "market_data" not in payload or "weather_data" not in payload:
#             raise ValueError("payload must contain 'market_data' and 'weather_data'")
#         return payload["market_data"], payload["weather_data"]

#     if payload_path is None:
#         raise ValueError("Provide either payload or payload_path")

#     with open(payload_path, "r", encoding="utf-8") as f:
#         payload = json.load(f)

#     if "market_data" not in payload or "weather_data" not in payload:
#         raise ValueError("payload.json must contain 'market_data' and 'weather_data'")

#     return payload["market_data"], payload["weather_data"]


# def build_dam_inputs_from_json(
#     payload=None,
#     payload_path=None,
#     prediction_date=None,
#     region="Telangana",
#     output_dir="v8_inputs",
#     save_csv=False,
# ):
#     """
#     Returns:
#         x_past_df   : DataFrame (1344, n_past_features)
#         x_future_df : DataFrame (96,   n_future_features)
#     """
#     if payload is not None:
#         if prediction_date is None:
#             prediction_date = payload.get("prediction_date")
#         if region is None:
#             region = payload.get("region", "Telangana")

#     if prediction_date is None:
#         raise ValueError("prediction_date required")

#     prediction_date = pd.Timestamp(prediction_date).normalize()

#     if save_csv:
#         os.makedirs(output_dir, exist_ok=True)

#     market_data, weather_data = _load_json_payload(
#         payload=payload,
#         payload_path=payload_path,
#     )

#     # =====================================================
#     # MARKET DATA
#     # =====================================================
#     market = pd.DataFrame(market_data)
#     market["datetime_block"] = _parse_dt(market["datetime_block"])
#     market = market[market["region"] == region].copy()
#     market = market.dropna(subset=["datetime_block"])

#     for c in ["mcp_rs_mwh", "cleared_buy_mw", "cleared_sell_mw"]:
#         if c in market.columns:
#             market[c] = pd.to_numeric(market[c], errors="coerce")

#     # Split GDAM and DAM (ignore RTM and others)
#     gdam = market[market["market"] == "GDAM"].sort_values("datetime_block").copy()
#     dam  = market[market["market"] == "DAM"].sort_values("datetime_block").copy()

#     gdam = gdam.rename(columns={
#         "datetime_block": "datetime",
#         "mcp_rs_mwh":     "gdam_price",
#         "cleared_buy_mw": "buy_mw",
#         "cleared_sell_mw":"sell_mw",
#     })[["datetime", "gdam_price", "buy_mw", "sell_mw"]]

#     dam = dam.rename(columns={
#         "datetime_block": "datetime",
#         "mcp_rs_mwh":     "dam_price",
#         "cleared_buy_mw": "dam_buy_mw",
#         "cleared_sell_mw":"dam_sell_mw",
#     })[["datetime", "dam_price", "dam_buy_mw", "dam_sell_mw"]]

#     df = pd.merge(gdam, dam, on="datetime", how="inner").sort_values("datetime").copy()

#     if df.empty:
#         raise ValueError(
#             f"No overlapping DAM+GDAM rows found for region={region}. "
#             "Check that both markets exist in market_data."
#         )

#     # =====================================================
#     # WEATHER DATA
#     # =====================================================
#     weather = pd.DataFrame(weather_data)
#     weather["datetime_hour"] = _parse_dt(weather["datetime_hour"])
#     weather = weather[weather["region"] == region].copy()
#     weather = weather.dropna(subset=["datetime_hour"])

#     weather = weather.rename(columns={
#         "temperature":     "temp",
#         "cloud_cover":     "cloud",
#         "wind_speed":      "wind",
#         "solar_irradiance":"solar",
#     })

#     for c in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
#         if c in weather.columns:
#             weather[c] = pd.to_numeric(weather[c], errors="coerce")

#     weather["datetime_hour"] = weather["datetime_hour"].dt.floor("h")
#     df["hour_dt"] = df["datetime"].dt.floor("h")

#     df = df.merge(
#         weather[["datetime_hour", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
#         left_on="hour_dt", right_on="datetime_hour", how="left",
#     ).drop(columns=["datetime_hour"])

#     for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
#         if col in df.columns:
#             df[col] = df[col].ffill().bfill()

#     # =====================================================
#     # ENGINEER FEATURES  (same pipeline as training)
#     # =====================================================
#     engineered = _engineer_features(df)

#     # =====================================================
#     # PAST WINDOW  (1344 blocks before prediction_date)
#     # =====================================================
#     hist = engineered[engineered["datetime"] < prediction_date].tail(PAST_STEPS).copy()

#     if len(hist) != PAST_STEPS:
#         raise ValueError(
#             f"Need {PAST_STEPS} past rows before {prediction_date.date()}; "
#             f"got {len(hist)}. Provide at least 14 days of history."
#         )

#     x_past_df = hist[PAST_FEATURES].reset_index(drop=True)

#     # =====================================================
#     # FUTURE WINDOW  (96 blocks on prediction_date)
#     # =====================================================
#     future_dt = pd.date_range(
#         start=prediction_date,
#         periods=FUTURE_STEPS,
#         freq="15min",
#     )

#     future_df = pd.DataFrame({"datetime": future_dt})
#     future_df["hour_dt"] = future_df["datetime"].dt.floor("h")

#     future_df = future_df.merge(
#         weather[["datetime_hour", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
#         left_on="hour_dt", right_on="datetime_hour", how="left",
#     ).drop(columns=["datetime_hour"])

#     for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
#         if col in future_df.columns:
#             future_df[col] = future_df[col].ffill().bfill()

#     # Calendar features
#     future_df["hour"]        = future_df["datetime"].dt.hour
#     future_df["hour_sin"]    = np.sin(2 * np.pi * future_df["hour"] / 24)
#     future_df["hour_cos"]    = np.cos(2 * np.pi * future_df["hour"] / 24)
#     future_df["day_of_week"] = future_df["datetime"].dt.dayofweek
#     future_df["dow_sin"]     = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
#     future_df["dow_cos"]     = np.cos(2 * np.pi * future_df["day_of_week"] / 7)

#     india_holidays = holidays_lib.India()
#     future_df["is_holiday"] = future_df["datetime"].dt.date.apply(
#         lambda d: int(d in india_holidays)
#     )
#     future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)

#     future_df["solar_hour_interaction"] = future_df["solar"] * future_df["hour_sin"]

#     # Daily regime context — carry forward last known values from history
#     # These are yesterday's stats relative to prediction_date, already in hist
#     regime_cols = [
#         "dam_yesterday_mean", "dam_yesterday_std",
#         "dam_yesterday_min",  "dam_yesterday_max",
#         "dam_roll_mean_7d",   "dam_roll_std_7d",
#         "dam_roll_mean_14d",  "dam_roll_std_14d",
#         "gdam_yesterday_mean","gdam_yesterday_std",
#         "gdam_roll_mean_7d",  "gdam_roll_std_7d",
#         "gdam_roll_mean_14d", "gdam_roll_std_14d",
#         "spread_yesterday_mean","spread_yesterday_std",
#         "spread_roll_mean_7d", "spread_roll_std_7d",
#         "spread_roll_mean_14d","spread_roll_std_14d",
#         "low_price_regime",   "high_price_regime",
#     ]

#     # Take the last row of hist for regime values (most recent past day)
#     last_hist = hist.iloc[-1]
#     for col in regime_cols:
#         if col in last_hist.index:
#             future_df[col] = last_hist[col]
#         else:
#             future_df[col] = 0

#     x_future_df = future_df[FUTURE_FEATURES].reset_index(drop=True)

#     # =====================================================
#     # VALIDATE
#     # =====================================================
#     if x_past_df.isnull().any().any():
#         null_cols = x_past_df.columns[x_past_df.isnull().any()].tolist()
#         raise ValueError(f"x_past contains NaNs in columns: {null_cols}")

#     if x_future_df.isnull().any().any():
#         null_cols = x_future_df.columns[x_future_df.isnull().any()].tolist()
#         raise ValueError(f"x_future contains NaNs in columns: {null_cols}")

#     # =====================================================
#     # OPTIONAL SAVE
#     # =====================================================
#     if save_csv:
#         x_past_df.to_csv(os.path.join(output_dir, "x_past.csv"),   index=False)
#         x_future_df.to_csv(os.path.join(output_dir, "x_future.csv"), index=False)
#         print(f"Saved inputs to {output_dir}/")

#     print(f"x_past  : {x_past_df.shape}")
#     print(f"x_future: {x_future_df.shape}")

#     return x_past_df, x_future_df


"""
V8_build_inputs_from_json_DAM_fixed.py

Build x_past and x_future for DAM V8 inference from a JSON payload.

Design goals:
1) Mirror the training-time feature engineering as closely as possible.
2) Use only information that is available at forecast time.
3) Be explicit about every step so debugging is easy.
4) Fail loudly with helpful messages when the payload is incomplete.

Expected payload structure:
{
  "prediction_date": "2026-06-20",
  "region": "Telangana",
  "market_data": [...],
  "weather_data": [...]
}

Market rows need at least:
  region, market, datetime_block, mcp_rs_mwh, cleared_buy_mw, cleared_sell_mw

Weather rows need at least:
  region, datetime_hour, temperature, humidity, cloud_cover, wind_speed,
  solar_irradiance, rain
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import holidays as holidays_lib
import numpy as np
import pandas as pd

from V8_data_loader_DAM import (
    _add_daily_context,
    _engineer_features,
    PAST_FEATURES,
    FUTURE_FEATURES,
    PAST_STEPS,
)

FUTURE_STEPS = 96


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _parse_dt(series_or_value):
    return pd.to_datetime(series_or_value, errors="coerce")


def _require_keys(payload: Dict, keys: List[str]) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"Payload missing required keys: {missing}")


def _load_payload(payload: Optional[Dict] = None, payload_path: Optional[str] = None):
    if payload is None and payload_path is None:
        raise ValueError("Provide either payload or payload_path")

    if payload is None:
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    _require_keys(payload, ["market_data", "weather_data"])
    return payload


def _clean_market_df(payload: Dict, region: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return separate cleaned DAM and GDAM frames with a common datetime column.
    """
    market = pd.DataFrame(payload["market_data"]).copy()
    if market.empty:
        raise ValueError("market_data is empty")

    market["datetime_block"] = _parse_dt(market.get("datetime_block"))
    market = market[market["region"] == region].copy()
    market = market.dropna(subset=["datetime_block"])

    needed_cols = ["market", "mcp_rs_mwh", "cleared_buy_mw", "cleared_sell_mw"]
    missing = [c for c in needed_cols if c not in market.columns]
    if missing:
        raise ValueError(f"market_data is missing required columns: {missing}")

    for c in ["mcp_rs_mwh", "cleared_buy_mw", "cleared_sell_mw"]:
        market[c] = pd.to_numeric(market[c], errors="coerce")

    # Keep only markets used by the model.
    market = market[market["market"].isin(["DAM", "GDAM"])].copy()

    gdam = (
        market[market["market"] == "GDAM"]
        .sort_values("datetime_block")
        .rename(
            columns={
                "datetime_block": "datetime",
                "mcp_rs_mwh": "gdam_price",
                "cleared_buy_mw": "buy_mw",
                "cleared_sell_mw": "sell_mw",
            }
        )[["datetime", "gdam_price", "buy_mw", "sell_mw"]]
    )

    dam = (
        market[market["market"] == "DAM"]
        .sort_values("datetime_block")
        .rename(
            columns={
                "datetime_block": "datetime",
                "mcp_rs_mwh": "dam_price",
                "cleared_buy_mw": "dam_buy_mw",
                "cleared_sell_mw": "dam_sell_mw",
            }
        )[["datetime", "dam_price", "dam_buy_mw", "dam_sell_mw"]]
    )

    if gdam.empty:
        raise ValueError(f"No GDAM rows found for region={region}")
    if dam.empty:
        raise ValueError(f"No DAM rows found for region={region}")

    return dam, gdam


def _clean_weather_df(payload: Dict, region: str) -> pd.DataFrame:
    weather = pd.DataFrame(payload["weather_data"]).copy()
    if weather.empty:
        raise ValueError("weather_data is empty")

    weather["datetime_hour"] = _parse_dt(weather.get("datetime_hour"))
    weather = weather[weather["region"] == region].copy()
    weather = weather.dropna(subset=["datetime_hour"])

    rename_map = {
        "temperature": "temp",
        "cloud_cover": "cloud",
        "wind_speed": "wind",
        "solar_irradiance": "solar",
    }
    weather = weather.rename(columns=rename_map)

    needed_cols = ["temp", "humidity", "cloud", "wind", "solar", "rain"]
    missing = [c for c in needed_cols if c not in weather.columns]
    if missing:
        raise ValueError(f"weather_data is missing required columns: {missing}")

    for c in needed_cols:
        weather[c] = pd.to_numeric(weather[c], errors="coerce")

    # one row per hour
    weather["datetime_hour"] = weather["datetime_hour"].dt.floor("h")
    weather = (
        weather.sort_values("datetime_hour")
        .drop_duplicates(subset=["datetime_hour"], keep="last")
        .reset_index(drop=True)
    )

    return weather


def _merge_market_history(dam: pd.DataFrame, gdam: pd.DataFrame) -> pd.DataFrame:
    """
    Inner merge ensures we only keep blocks where both DAM and GDAM exist.
    """
    df = pd.merge(
        gdam,
        dam,
        on="datetime",
        how="inner",
        validate="one_to_one",
    ).sort_values("datetime").reset_index(drop=True)

    if df.empty:
        raise ValueError("No overlapping DAM/GDAM timestamps after merge")

    return df


def _attach_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weather = weather.copy()

    df["hour_dt"] = df["datetime"].dt.floor("h")

    df = df.merge(
        weather[["datetime_hour", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
        left_on="hour_dt",
        right_on="datetime_hour",
        how="left",
    ).drop(columns=["datetime_hour"])

    # We allow hourly weather to be broadcast to the 15-min blocks.
    # If forecast weather is sparse, forward/back-fill within the local frame.
    for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
        df[col] = df[col].ffill().bfill()

    return df


def _build_history_features(raw_history: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild features for the historical section.

    This mirrors _engineer_features() but keeps the code path explicit for inference.
    We do not create target labels here; we only need the input features.
    """
    df = raw_history.copy()

    # Base feature engineering (same logic as training loader)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Standardize GDAM buy/sell names to match loader expectations.
    if "buy_mw" in df.columns:
        df = df.rename(columns={"buy_mw": "gdam_buy_mw", "sell_mw": "gdam_sell_mw"})

    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    try:
        india_holidays = holidays_lib.India()
        df["is_holiday"] = df["datetime"].dt.date.apply(lambda d: int(d in india_holidays))
    except Exception:
        df["is_holiday"] = 0

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["solar_hour_interaction"] = df["solar"] * df["hour_sin"]
    df["price_spread"] = df["dam_price"] - df["gdam_price"]
    df["gdam_dam_ratio"] = df["gdam_price"] / df["dam_price"].replace(0, np.nan)

    df["dam_demand_supply_ratio"] = df["dam_buy_mw"] / (df["dam_sell_mw"] + 1.0)
    df["gdam_demand_supply_ratio"] = df["gdam_buy_mw"] / (df["gdam_sell_mw"] + 1.0)

    df["date"] = df["datetime"].dt.date

    daily_ctx = _add_daily_context(df)
    df = df.merge(daily_ctx, on="date", how="left")

    # Base target-related columns exist in the training loader; we keep them for consistency.
    df["dam_price_raw"] = df["dam_price"].astype(float)
    df["dam_log_price_raw"] = np.log1p(df["dam_price_raw"])

    # Clean numeric issues.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def _build_future_features(
    history_engineered: pd.DataFrame,
    weather: pd.DataFrame,
    prediction_date: pd.Timestamp,
    region: str,
) -> pd.DataFrame:
    """
    Build the 96 future rows for the prediction day.

    Important:
    - The daily context columns are forecast-time-known as-of yesterday features.
    - So we copy them from the last available historical row.
    - Weather is broadcast from hourly data to 15-minute blocks.
    """
    future_dt = pd.date_range(start=prediction_date, periods=FUTURE_STEPS, freq="15min")

    future = pd.DataFrame({"datetime": future_dt})
    future["hour_dt"] = future["datetime"].dt.floor("h")

    future = future.merge(
        weather[["datetime_hour", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
        left_on="hour_dt",
        right_on="datetime_hour",
        how="left",
    ).drop(columns=["datetime_hour"])

    for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
        future[col] = future[col].ffill().bfill()

    future["hour"] = future["datetime"].dt.hour
    future["hour_sin"] = np.sin(2 * np.pi * future["hour"] / 24)
    future["hour_cos"] = np.cos(2 * np.pi * future["hour"] / 24)

    future["day_of_week"] = future["datetime"].dt.dayofweek
    future["dow_sin"] = np.sin(2 * np.pi * future["day_of_week"] / 7)
    future["dow_cos"] = np.cos(2 * np.pi * future["day_of_week"] / 7)

    try:
        india_holidays = holidays_lib.India()
        future["is_holiday"] = future["datetime"].dt.date.apply(lambda d: int(d in india_holidays))
    except Exception:
        future["is_holiday"] = 0

    future["is_weekend"] = (future["day_of_week"] >= 5).astype(int)
    future["solar_hour_interaction"] = future["solar"] * future["hour_sin"]

    # As-of-yesterday regime context:
    last_hist = history_engineered.iloc[-1]

    daily_context_cols = [
        "dam_yesterday_mean", "dam_yesterday_std", "dam_yesterday_min", "dam_yesterday_max",
        "dam_roll_mean_7d", "dam_roll_std_7d", "dam_roll_mean_14d", "dam_roll_std_14d",
        "gdam_yesterday_mean", "gdam_yesterday_std", "gdam_roll_mean_7d", "gdam_roll_std_7d",
        "gdam_roll_mean_14d", "gdam_roll_std_14d",
        "spread_yesterday_mean", "spread_yesterday_std",
        "spread_roll_mean_7d", "spread_roll_std_7d",
        "spread_roll_mean_14d", "spread_roll_std_14d",
        "low_price_regime", "high_price_regime",
    ]

    for col in daily_context_cols:
        future[col] = last_hist[col] if col in history_engineered.columns else 0

    future["date"] = future["datetime"].dt.date

    return future


def build_dam_inputs_from_json(
    payload: Optional[Dict] = None,
    payload_path: Optional[str] = None,
    prediction_date: Optional[str] = None,
    region: str = "Telangana",
    output_dir: str = "v8_inputs",
    save_csv: bool = False,
    verbose: bool = True,
):
    """
    Returns:
        x_past_df   : DataFrame (1344, n_past_features)
        x_future_df : DataFrame (96,   n_future_features)

    This function is intentionally strict:
    - It checks raw payload structure.
    - It mirrors the training feature engineering.
    - It raises a helpful error if there is not enough history.
    """
    payload = _load_payload(payload, payload_path)

    if prediction_date is None:
        prediction_date = payload.get("prediction_date")
    if prediction_date is None:
        raise ValueError("prediction_date is required")

    if "region" in payload and payload["region"]:
        region = payload["region"]

    prediction_date = pd.Timestamp(prediction_date).normalize()

    dam, gdam = _clean_market_df(payload, region)
    weather = _clean_weather_df(payload, region)
    history_raw = _merge_market_history(dam, gdam)
    history_raw = _attach_weather(history_raw, weather)

    # Build history features using the same logical feature pipeline as training.
    history_engineered = _build_history_features(history_raw)

    if verbose:
        print("=" * 72)
        print("RAW / MERGE CHECKS")
        print("=" * 72)
        print(f"DAM rows          : {len(dam)}")
        print(f"GDAM rows         : {len(gdam)}")
        print(f"After DAM/GDAM merge: {len(history_raw)}")
        print(f"After feature engineering: {len(history_engineered)}")
        print(f"Prediction date   : {prediction_date.date()}")
        print()

    # Historical blocks available before the forecast day
    hist_before_pred = history_engineered[history_engineered["datetime"] < prediction_date].copy()
    hist_before_pred = hist_before_pred.sort_values("datetime").reset_index(drop=True)

    if verbose:
        print(f"Rows before prediction date: {len(hist_before_pred)}")
        if len(hist_before_pred) > 0:
            print(f"History start: {hist_before_pred['datetime'].min()}")
            print(f"History end  : {hist_before_pred['datetime'].max()}")
        print()

    if len(hist_before_pred) < PAST_STEPS:
        # Helpful breakdown for debugging
        missing = PAST_STEPS - len(hist_before_pred)
        raise ValueError(
            f"Need {PAST_STEPS} past rows before {prediction_date.date()}, "
            f"but only {len(hist_before_pred)} are available after feature engineering. "
            f"Missing {missing} rows. "
            f"Check whether the payload contains 14 full days of DAM+GDAM history, "
            f"and whether weather rows are missing for some hours."
        )

    # Past window is the last 1344 rows strictly before prediction_date
    x_past_df = hist_before_pred.tail(PAST_STEPS)[PAST_FEATURES].reset_index(drop=True)

    # Future window is the prediction day itself
    future_df = _build_future_features(
        history_engineered=hist_before_pred,
        weather=weather,
        prediction_date=prediction_date,
        region=region,
    )

    x_future_df = future_df[FUTURE_FEATURES].reset_index(drop=True)

    # Final validation
    if x_past_df.shape != (PAST_STEPS, len(PAST_FEATURES)):
        raise ValueError(f"x_past has wrong shape: {x_past_df.shape}")
    if x_future_df.shape != (FUTURE_STEPS, len(FUTURE_FEATURES)):
        raise ValueError(f"x_future has wrong shape: {x_future_df.shape}")

    if x_past_df.isna().any().any():
        bad = x_past_df.columns[x_past_df.isna().any()].tolist()
        raise ValueError(f"x_past contains NaNs in columns: {bad}")

    if x_future_df.isna().any().any():
        bad = x_future_df.columns[x_future_df.isna().any()].tolist()
        raise ValueError(f"x_future contains NaNs in columns: {bad}")

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        x_past_df.to_csv(os.path.join(output_dir, "x_past.csv"), index=False)
        x_future_df.to_csv(os.path.join(output_dir, "x_future.csv"), index=False)
        if verbose:
            print(f"Saved inputs to {output_dir}/")

    if verbose:
        print(f"x_past  : {x_past_df.shape}")
        print(f"x_future: {x_future_df.shape}")

    return x_past_df, x_future_df


if __name__ == "__main__":
    with open("json_inputs/payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    x_past_df, x_future_df = build_dam_inputs_from_json(
        payload=payload,
        prediction_date=payload.get("prediction_date"),
        region=payload.get("region", "Telangana"),
        save_csv=True,
        output_dir="v8_inputs",
        verbose=True,
    )

    print()
    print("First 3 past rows:")
    print(x_past_df.head(3))
    print()
    print("First 3 future rows:")
    print(x_future_df.head(3))

import os
import json
import numpy as np
import pandas as pd
import holidays

from V6_data_loader_DAM import (
    _engineer_features,
    PAST_STEPS,
    PAST_FEATURES,
    FUTURE_FEATURES,
)

FUTURE_STEPS = 96


def _parse_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _load_json_payload(payload=None, payload_path=None):
    if payload is not None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")
        if "market_data" not in payload or "weather_data" not in payload:
            raise ValueError("payload must contain 'market_data' and 'weather_data'")
        return payload["market_data"], payload["weather_data"]

    if payload_path is None:
        raise ValueError("Provide either payload or payload_path")

    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "market_data" not in payload or "weather_data" not in payload:
        raise ValueError("payload.json must contain 'market_data' and 'weather_data'")

    return payload["market_data"], payload["weather_data"]


def _rename_market_subset(subset: pd.DataFrame, market_name: str) -> pd.DataFrame:
    subset = subset.copy()

    required = {"datetime_block", "mcp_rs_mwh"}
    missing = required - set(subset.columns)
    if missing:
        raise ValueError(f"{market_name} missing required columns: {sorted(missing)}")

    rename_map = {
        "datetime_block": "datetime",
        "mcp_rs_mwh": f"{market_name.lower()}_price",
    }

    if market_name == "GDAM":
        buy_src = "cleared_buy_mw" if "cleared_buy_mw" in subset.columns else "buy_mw"
        sell_src = "cleared_sell_mw" if "cleared_sell_mw" in subset.columns else "sell_mw"
        if buy_src not in subset.columns or sell_src not in subset.columns:
            raise ValueError("GDAM market slice must contain buy/sell MW columns.")
        rename_map[buy_src] = "buy_mw"
        rename_map[sell_src] = "sell_mw"
    else:
        buy_src = "cleared_buy_mw" if "cleared_buy_mw" in subset.columns else "buy_mw"
        sell_src = "cleared_sell_mw" if "cleared_sell_mw" in subset.columns else "sell_mw"
        if buy_src not in subset.columns or sell_src not in subset.columns:
            raise ValueError("DAM market slice must contain buy/sell MW columns.")
        rename_map[buy_src] = "dam_buy_mw"
        rename_map[sell_src] = "dam_sell_mw"

    return subset.rename(columns=rename_map)


def build_dam_inputs_from_json(
    payload=None,
    payload_path=None,
    prediction_date=None,
    region="Telangana",
    output_dir="v6_inputs",
    save_csv=False,
):
    if payload is not None:
        if prediction_date is None:
            prediction_date = payload.get("prediction_date")
        if region is None:
            region = payload.get("region", "Telangana")

    if prediction_date is None:
        raise ValueError("prediction_date required")

    prediction_date = pd.Timestamp(prediction_date).normalize()

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)

    market_data, weather_data = _load_json_payload(
        payload=payload,
        payload_path=payload_path,
    )

    market = pd.DataFrame(market_data)
    weather = pd.DataFrame(weather_data)

    # =====================================================
    # LOAD MARKET
    # =====================================================
    market["datetime_block"] = _parse_dt(market["datetime_block"])
    market = market[market["region"] == region].copy()
    market = market.dropna(subset=["datetime_block"])

    for c in ["mcp_rs_mwh", "buy_mw", "sell_mw", "cleared_buy_mw", "cleared_sell_mw"]:
        if c in market.columns:
            market[c] = pd.to_numeric(market[c], errors="coerce")

    # =====================================================
    # SPLIT GDAM / DAM
    # =====================================================
    gdam = market[market["market"] == "GDAM"].sort_values("datetime_block").copy()
    dam = market[market["market"] == "DAM"].sort_values("datetime_block").copy()

    gdam = _rename_market_subset(gdam, "GDAM")
    dam = _rename_market_subset(dam, "DAM")

    df = pd.merge(
        gdam[["datetime", "gdam_price", "buy_mw", "sell_mw"]],
        dam[["datetime", "dam_price", "dam_buy_mw", "dam_sell_mw"]],
        on="datetime",
        how="inner",
    ).sort_values("datetime").copy()

    # =====================================================
    # WEATHER
    # =====================================================
    weather["datetime_hour"] = _parse_dt(weather["datetime_hour"])
    weather = weather[weather["region"] == region].copy()
    weather = weather.dropna(subset=["datetime_hour"])

    weather = weather.rename(columns={
        "datetime_hour": "hour_dt",
        "temperature": "temp",
        "cloud_cover": "cloud",
        "wind_speed": "wind",
        "solar_irradiance": "solar",
    })

    for c in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
        if c in weather.columns:
            weather[c] = pd.to_numeric(weather[c], errors="coerce")

    weather["hour_dt"] = weather["hour_dt"].dt.floor("h")
    df["hour_dt"] = df["datetime"].dt.floor("h")

    df = df.merge(
        weather[["hour_dt", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
        on="hour_dt",
        how="left",
    )

    for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # =====================================================
    # ENGINEER FEATURES
    # =====================================================
    engineered = _engineer_features(df)

    # =====================================================
    # PAST WINDOW
    # =====================================================
    hist = engineered[engineered["datetime"] < prediction_date].tail(PAST_STEPS).copy()

    if len(hist) != PAST_STEPS:
        raise ValueError(f"Need {PAST_STEPS} rows before cutoff; got {len(hist)}")

    x_past_df = hist[PAST_FEATURES].copy()

    # =====================================================
    # FUTURE WINDOW
    # =====================================================
    future_dt = pd.date_range(
        start=prediction_date,
        periods=FUTURE_STEPS,
        freq="15min",
    )

    future_df = pd.DataFrame({"datetime": future_dt})
    future_df["hour_dt"] = future_df["datetime"].dt.floor("h")

    future_df = future_df.merge(
        weather[["hour_dt", "temp", "humidity", "cloud", "wind", "solar", "rain"]],
        on="hour_dt",
        how="left",
    )

    for col in ["temp", "humidity", "cloud", "wind", "solar", "rain"]:
        if col in future_df.columns:
            future_df[col] = future_df[col].ffill().bfill()

    future_df["hour"] = future_df["datetime"].dt.hour
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)

    future_df["day_of_week"] = future_df["datetime"].dt.dayofweek
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)

    india_holidays = holidays.India()
    future_df["is_holiday"] = future_df["datetime"].dt.date.apply(
        lambda d: int(d in india_holidays)
    )
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)

    future_df["solar_hour_interaction"] = future_df["solar"] * future_df["hour_sin"]

    x_future_df = future_df[FUTURE_FEATURES].copy()

    # =====================================================
    # BASELINE
    # =====================================================
    dam_price_map = (
        engineered[["datetime", "dam_price"]]
        .drop_duplicates("datetime")
        .set_index("datetime")["dam_price"]
    )

    prev_dt = future_dt - pd.Timedelta(days=1)
    baseline = dam_price_map.reindex(prev_dt).to_numpy(dtype=np.float32)

    if np.isnan(baseline).any():
        bad = np.where(np.isnan(baseline))[0][:10]
        raise ValueError(
            f"Baseline contains NaNs. Missing previous-day DAM data for blocks: {bad.tolist()}"
        )

    baseline_df = pd.DataFrame(
        {
            "datetime": future_dt,
            "baseline_price": baseline,
        }
    )

    # =====================================================
    # OPTIONAL SAVE
    # =====================================================
    if save_csv:
        x_past_path = os.path.join(output_dir, "x_past.csv")
        x_future_path = os.path.join(output_dir, "x_future.csv")
        baseline_path = os.path.join(output_dir, "baseline.csv")

        x_past_df.to_csv(x_past_path, index=False)
        x_future_df.to_csv(x_future_path, index=False)
        baseline_df.to_csv(baseline_path, index=False)

        print("Saved:")
        print(f"  {x_past_path}")
        print(f"  {x_future_path}")
        print(f"  {baseline_path}")

    print("Shapes:")
    print(f"  x_past: {x_past_df.shape}")
    print(f"  x_future: {x_future_df.shape}")
    print(f"  baseline: {baseline_df.shape}")

    return x_past_df, x_future_df, baseline_df
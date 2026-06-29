import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG
# =====================================================

PAST_STEPS = 1344          # 14 days * 96 blocks
TARGET_STEPS = 96          # 1 day ahead
FLOOR = 100.0              # for safe ratio computations

# =====================================================
# FEATURE LISTS
# =====================================================

PAST_FEATURES = [
    # Current / historical market state
    "dam_price",
    "gdam_price",
    "dam_buy_mw",
    "dam_sell_mw",
    "gdam_buy_mw",
    "gdam_sell_mw",
    "dam_demand_supply_ratio",
    "gdam_demand_supply_ratio",
    "price_spread",
    "gdam_dam_ratio",

    # Weather
    "temp",
    "humidity",
    "rain",
    "cloud",
    "wind",
    "solar",
    "solar_hour_interaction",

    # Calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",

    # DAM daily regime context (yesterday + rolling)
    "dam_yesterday_mean",
    "dam_yesterday_std",
    "dam_yesterday_min",
    "dam_yesterday_max",
    "dam_roll_mean_7d",
    "dam_roll_std_7d",
    "dam_roll_mean_14d",
    "dam_roll_std_14d",

    # GDAM daily regime context
    "gdam_yesterday_mean",
    "gdam_yesterday_std",
    "gdam_roll_mean_7d",
    "gdam_roll_std_7d",
    "gdam_roll_mean_14d",
    "gdam_roll_std_14d",

    # Spread daily regime context
    "spread_yesterday_mean",
    "spread_yesterday_std",
    "spread_roll_mean_7d",
    "spread_roll_std_7d",
    "spread_roll_mean_14d",
    "spread_roll_std_14d",

    # Regime flags
    "low_price_regime",
    "high_price_regime",
]

FUTURE_FEATURES = [
    # Weather
    "temp",
    "humidity",
    "rain",
    "cloud",
    "wind",
    "solar",
    "solar_hour_interaction",

    # Calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",

    # DAM daily regime context (known from history at forecast time)
    "dam_yesterday_mean",
    "dam_yesterday_std",
    "dam_yesterday_min",
    "dam_yesterday_max",
    "dam_roll_mean_7d",
    "dam_roll_std_7d",
    "dam_roll_mean_14d",
    "dam_roll_std_14d",

    # GDAM daily regime context
    "gdam_yesterday_mean",
    "gdam_yesterday_std",
    "gdam_roll_mean_7d",
    "gdam_roll_std_7d",
    "gdam_roll_mean_14d",
    "gdam_roll_std_14d",

    # Spread daily regime context
    "spread_yesterday_mean",
    "spread_yesterday_std",
    "spread_roll_mean_7d",
    "spread_roll_std_7d",
    "spread_roll_mean_14d",
    "spread_roll_std_14d",

    # Regime flags
    "low_price_regime",
    "high_price_regime",
]

BINARY_FEATURES = {
    "is_holiday",
    "is_weekend",
    "low_price_regime",
    "high_price_regime",
}

SCALE_FEATURES = [
    c for c in dict.fromkeys(PAST_FEATURES + FUTURE_FEATURES)
    if c not in BINARY_FEATURES
]

# =====================================================
# DATASET
# =====================================================

class DAMDataset(Dataset):
    def __init__(self, x_past, x_future, y_target):
        self.x_past = torch.FloatTensor(x_past)
        self.x_future = torch.FloatTensor(x_future)
        self.y_target = torch.FloatTensor(y_target)

    def __len__(self):
        return len(self.y_target)

    def __getitem__(self, idx):
        return (
            self.x_past[idx],
            self.x_future[idx],
            self.y_target[idx],
        )

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def _add_daily_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build day-level regime features using only past information.
    All rolling features are shifted by 1 day to avoid leakage.
    """
    daily = (
        df.groupby("date", as_index=True)
        .agg(
            dam_day_mean=("dam_price", "mean"),
            dam_day_std=("dam_price", "std"),
            dam_day_min=("dam_price", "min"),
            dam_day_max=("dam_price", "max"),
            gdam_day_mean=("gdam_price", "mean"),
            gdam_day_std=("gdam_price", "std"),
            spread_day_mean=("price_spread", "mean"),
            spread_day_std=("price_spread", "std"),
        )
        .sort_index()
    )

    # Yesterday stats
    daily["dam_yesterday_mean"] = daily["dam_day_mean"].shift(1)
    daily["dam_yesterday_std"] = daily["dam_day_std"].shift(1)
    daily["dam_yesterday_min"] = daily["dam_day_min"].shift(1)
    daily["dam_yesterday_max"] = daily["dam_day_max"].shift(1)

    daily["gdam_yesterday_mean"] = daily["gdam_day_mean"].shift(1)
    daily["gdam_yesterday_std"] = daily["gdam_day_std"].shift(1)

    daily["spread_yesterday_mean"] = daily["spread_day_mean"].shift(1)
    daily["spread_yesterday_std"] = daily["spread_day_std"].shift(1)

    # Rolling features over prior days only
    def roll_mean_std(series: pd.Series, window: int, prefix: str):
        shifted = series.shift(1)
        min_periods = max(3, window // 2)
        daily[f"{prefix}_roll_mean_{window}d"] = shifted.rolling(
            window=window, min_periods=min_periods
        ).mean()
        daily[f"{prefix}_roll_std_{window}d"] = shifted.rolling(
            window=window, min_periods=min_periods
        ).std()

    for w in (7, 14):
        roll_mean_std(daily["dam_day_mean"], w, "dam")
        roll_mean_std(daily["gdam_day_mean"], w, "gdam")
        roll_mean_std(daily["spread_day_mean"], w, "spread")

    # Regime flags
    daily["low_price_regime"] = (daily["dam_roll_mean_7d"] < 3000).astype(int)
    daily["high_price_regime"] = (daily["dam_roll_mean_7d"] > 7000).astype(int)

    daily = daily.reset_index()
    return daily


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from the already-merged historical dataset.

    Required raw columns:
      datetime, gdam_price, buy_mw, sell_mw, dam_price,
      dam_buy_mw, dam_sell_mw, temp, humidity, rain, cloud, wind, solar
    """
    df = df.copy()

    if "datetime" not in df.columns:
        raise ValueError("Missing required column: datetime")

    required_cols = [
        "gdam_price",
        "buy_mw",
        "sell_mw",
        "dam_price",
        "dam_buy_mw",
        "dam_sell_mw",
        "temp",
        "humidity",
        "rain",
        "cloud",
        "wind",
        "solar",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Standardize GDAM buy/sell names
    df = df.rename(columns={"buy_mw": "gdam_buy_mw", "sell_mw": "gdam_sell_mw"})

    # Calendar features
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    try:
        import holidays
        india_holidays = holidays.India()
        df["is_holiday"] = df["datetime"].dt.date.apply(lambda d: int(d in india_holidays))
    except Exception:
        df["is_holiday"] = 0

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Core market interactions
    df["solar_hour_interaction"] = df["solar"] * df["hour_sin"]
    df["price_spread"] = df["dam_price"] - df["gdam_price"]
    df["gdam_dam_ratio"] = df["gdam_price"] / df["dam_price"].replace(0, np.nan)

    df["dam_demand_supply_ratio"] = df["dam_buy_mw"] / (df["dam_sell_mw"] + 1.0)
    df["gdam_demand_supply_ratio"] = df["gdam_buy_mw"] / (df["gdam_sell_mw"] + 1.0)

    # Daily context features
    df["date"] = df["datetime"].dt.date
    daily_ctx = _add_daily_context(df)
    df = df.merge(daily_ctx, on="date", how="left")

    # Log-price target base column
    df["dam_price_raw"] = df["dam_price"].astype(float)
    df["dam_log_price_raw"] = np.log1p(df["dam_price_raw"])

    # Clean up infinities / NaNs created by ratios and rolling stats
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


# =====================================================
# WINDOW CREATION
# =====================================================

def create_windowed_data(
    df,
    past_steps=PAST_STEPS,
    target_steps=TARGET_STEPS,
    target_dates=None,
):
    """
    Create windowed samples from an engineered dataframe.

    Target is dam_log_price_target_scaled (log-price target),
    not a baseline difference.
    """
    if target_dates is None:
        target_dates = sorted(df["date"].unique())
    else:
        target_dates = list(target_dates)

    X_past, X_future, Y_target = [], [], []

    for target_date in target_dates:
        day_d_data = df[df["date"] == target_date].copy()
        if len(day_d_data) != target_steps:
            continue

        cutoff_timestamp = pd.Timestamp(target_date)
        cutoff_rows = df[df["datetime"] < cutoff_timestamp]

        if len(cutoff_rows) == 0:
            continue

        cutoff_idx = cutoff_rows.index[-1]
        start_idx = cutoff_idx - past_steps + 1
        if start_idx < 0:
            continue

        x_past_window = df.iloc[start_idx:cutoff_idx + 1][PAST_FEATURES].values
        x_future_window = day_d_data[FUTURE_FEATURES].values
        y_target_window = day_d_data["dam_log_price_target_scaled"].values

        X_past.append(x_past_window)
        X_future.append(x_future_window)
        Y_target.append(y_target_window)

    return (
        np.array(X_past),
        np.array(X_future),
        np.array(Y_target),
    )


# =====================================================
# LOADERS
# =====================================================

def get_loaders(
    csv_path,
    batch_size=8,
    validation_days=60,
    train_end_date="2026-05-10",
):
    df_raw = pd.read_csv(csv_path)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

    # Engineer features on the full series first.
    # Rolling features use only past information because of the shift(1).
    df = _engineer_features(df_raw)

    cutoff_date = pd.to_datetime(train_end_date).date()
    df = df[df["datetime"].dt.date <= cutoff_date].copy().reset_index(drop=True)

    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) <= validation_days:
        raise ValueError("Not enough unique dates for train/validation split.")

    train_dates = unique_dates[:-validation_days]
    val_dates = unique_dates[-validation_days:]

    train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)

    # Validation keeps a bit of history for the past window
    val_start = pd.Timestamp(val_dates[0])
    history_rows = PAST_STEPS + (TARGET_STEPS * 2)
    history_df = (
        df[df["datetime"] < val_start]
        .tail(history_rows)
        .copy()
        .reset_index(drop=True)
    )

    val_df = pd.concat(
        [history_df, df[df["date"].isin(val_dates)].copy()],
        ignore_index=True
    ).reset_index(drop=True)

    # Fit scalers ONLY on train data
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[SCALE_FEATURES])

    target_scaler = StandardScaler()
    target_scaler.fit(train_df[["dam_log_price_raw"]])

    # Scale feature columns
    train_df.loc[:, SCALE_FEATURES] = feature_scaler.transform(train_df[SCALE_FEATURES])
    val_df.loc[:, SCALE_FEATURES] = feature_scaler.transform(val_df[SCALE_FEATURES])

    # Direct-price target (scaled log raw DAM price)
    train_df["dam_log_price_target_scaled"] = target_scaler.transform(
        train_df[["dam_log_price_raw"]]
    ).astype(np.float32).ravel()

    val_df["dam_log_price_target_scaled"] = target_scaler.transform(
        val_df[["dam_log_price_raw"]]
    ).astype(np.float32).ravel()

    # Create windows
    X_past_train, X_future_train, Y_train = create_windowed_data(
        train_df,
        past_steps=PAST_STEPS,
        target_steps=TARGET_STEPS,
        target_dates=train_dates,
    )
    X_past_val, X_future_val, Y_val = create_windowed_data(
        val_df,
        past_steps=PAST_STEPS,
        target_steps=TARGET_STEPS,
        target_dates=val_dates,
    )

    # Safety masks
    train_mask = (
        ~np.isnan(X_past_train).any(axis=(1, 2)) &
        ~np.isnan(X_future_train).any(axis=(1, 2)) &
        ~np.isnan(Y_train).any(axis=1)
    )
    val_mask = (
        ~np.isnan(X_past_val).any(axis=(1, 2)) &
        ~np.isnan(X_future_val).any(axis=(1, 2)) &
        ~np.isnan(Y_val).any(axis=1)
    )

    X_past_train = X_past_train[train_mask]
    X_future_train = X_future_train[train_mask]
    Y_train = Y_train[train_mask]

    X_past_val = X_past_val[val_mask]
    X_future_val = X_future_val[val_mask]
    Y_val = Y_val[val_mask]

    if len(Y_train) == 0:
        raise ValueError("No training samples created after windowing.")
    if len(Y_val) == 0:
        raise ValueError("No validation samples created after windowing.")

    train_loader = DataLoader(
        DAMDataset(X_past_train, X_future_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        DAMDataset(X_past_val, X_future_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, feature_scaler, target_scaler


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    df_path = "historical_dataset_extended.csv"

    if os.path.exists(df_path):
        train_loader, val_loader, feature_scaler, target_scaler = get_loaders(df_path)

        joblib.dump(feature_scaler, "dam_scaler_v7.joblib")
        joblib.dump(target_scaler, "dam_log_price_scaler_v7.joblib")

        print("Scalers saved:")
        print(" - dam_scaler_v7.joblib")
        print(" - dam_log_price_scaler_v7.joblib")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches  : {len(val_loader)}")

        sample_p, sample_f, sample_y = next(iter(train_loader))
        print(f"x_past shape   : {sample_p.shape}")
        print(f"x_future shape : {sample_f.shape}")
        print(f"y_target shape  : {sample_y.shape}")
    else:
        print(f"File not found: {df_path}")
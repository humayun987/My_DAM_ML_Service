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

PAST_STEPS = 1344
TARGET_STEPS = 96
FLOOR = 100.0


# =====================================================
# FEATURE LISTS
# =====================================================

PAST_FEATURES = [
    "gdam_return",
    "dam_return",
    "gdam_price",
    "dam_price",
    "gdam_buy_mw",
    "gdam_sell_mw",
    "dam_buy_mw",
    "dam_sell_mw",
    "gdam_demand_supply_ratio",
    "dam_demand_supply_ratio",
    "price_spread",
    "solar_hour_interaction",
    "gdam_volatility",
    "dam_volatility",
    "gdam_return_4h",
    "gdam_return_12h",
    "dam_return_4h",
    "dam_return_12h",
    "spread_change_4h",
    "spread_change_12h",
    "temp",
    "humidity",
    "rain",
    "cloud",
    "wind",
    "solar",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
]

FUTURE_FEATURES = [
    "temp",
    "humidity",
    "rain",
    "cloud",
    "wind",
    "solar",
    "solar_hour_interaction",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
]

SCALE_FEATURES = [
    "gdam_price",
    "dam_price",
    "gdam_buy_mw",
    "gdam_sell_mw",
    "dam_buy_mw",
    "dam_sell_mw",
    "gdam_demand_supply_ratio",
    "dam_demand_supply_ratio",
    "price_spread",
    "solar_hour_interaction",
    "gdam_volatility",
    "dam_volatility",
    "temp",
    "humidity",
    "rain",
    "cloud",
    "wind",
    "solar",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
]


# =====================================================
# DATASET
# =====================================================

class DAMDataset(Dataset):
    def __init__(self, x_past, x_future, y_target, y_baseline):
        self.x_past = torch.FloatTensor(x_past)
        self.x_future = torch.FloatTensor(x_future)
        self.y_target = torch.FloatTensor(y_target)
        self.y_baseline = torch.FloatTensor(y_baseline)

    def __len__(self):
        return len(self.y_target)

    def __getitem__(self, idx):
        return (
            self.x_past[idx],
            self.x_future[idx],
            self.y_target[idx],
            self.y_baseline[idx],
        )


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all V3 features from the raw master CSV.
    The raw CSV is kept unchanged.
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

    # Ensure datetime order
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Rename GDAM volumes for clarity
    df = df.rename(columns={
        "buy_mw": "gdam_buy_mw",
        "sell_mw": "gdam_sell_mw",
    })

    # Calendar / time features
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    try:
        import holidays
        india_holidays = holidays.India()
        df["is_holiday"] = df["datetime"].dt.date.apply(
            lambda d: int(d in india_holidays)
        )
    except Exception:
        df["is_holiday"] = 0

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Interactions / spreads
    df["solar_hour_interaction"] = df["solar"] * df["hour_sin"]
    df["price_spread"] = df["dam_price"] - df["gdam_price"]

    # Volatility
    df["gdam_volatility"] = df["gdam_price"].rolling(window=96).std()
    df["dam_volatility"] = df["dam_price"].rolling(window=96).std()

    # Demand-supply ratios
    df["gdam_demand_supply_ratio"] = df["gdam_buy_mw"] / (df["gdam_sell_mw"] + 1.0)
    df["dam_demand_supply_ratio"] = df["dam_buy_mw"] / (df["dam_sell_mw"] + 1.0)

    # Baselines
    df["gdam_baseline"] = df["gdam_price"].shift(96)
    df["dam_baseline"] = df["dam_price"].shift(96)

    # Returns (keep as input features)
    df["gdam_return"] = (
        (df["gdam_price"] - df["gdam_baseline"])
        / df["gdam_baseline"].clip(lower=FLOOR)
    )

    df["dam_return"] = (
        (df["dam_price"] - df["dam_baseline"])
        / df["dam_baseline"].clip(lower=FLOOR)
    )

    # NEW TARGET: difference
    df["dam_diff"] = df["dam_price"] - df["dam_baseline"]

    # Momentum features
    df["gdam_return_4h"] = (
        (df["gdam_price"] - df["gdam_price"].shift(16))
        / df["gdam_price"].shift(16).clip(lower=FLOOR)
    )

    df["gdam_return_12h"] = (
        (df["gdam_price"] - df["gdam_price"].shift(48))
        / df["gdam_price"].shift(48).clip(lower=FLOOR)
    )

    df["dam_return_4h"] = (
        (df["dam_price"] - df["dam_price"].shift(16))
        / df["dam_price"].shift(16).clip(lower=FLOOR)
    )

    df["dam_return_12h"] = (
        (df["dam_price"] - df["dam_price"].shift(48))
        / df["dam_price"].shift(48).clip(lower=FLOOR)
    )

    df["spread_change_4h"] = df["price_spread"] - df["price_spread"].shift(16)
    df["spread_change_12h"] = df["price_spread"] - df["price_spread"].shift(48)

    # Drop rows that cannot be used because of shifts / rolling windows
    df = df.dropna().reset_index(drop=True)

    # Add date column for date-based split
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

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
    Create windowed samples from an already-engineered and already-scaled dataframe.

    Target is dam_diff_scaled (scaled difference), not return.
    """
    if target_dates is None:
        target_dates = sorted(df["date"].unique())
    else:
        target_dates = list(target_dates)

    X_past, X_future, Y_target, Y_baseline = [], [], [], []

    for target_date in target_dates:
        day_d_data = df[df["date"] == target_date].copy()

        if len(day_d_data) != target_steps:
            continue

        # Cutoff is start of the target day.
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

        y_target_window = day_d_data["dam_diff_scaled"].values
        y_baseline_window = day_d_data["dam_baseline"].values

        X_past.append(x_past_window)
        X_future.append(x_future_window)
        Y_target.append(y_target_window)
        Y_baseline.append(y_baseline_window)

    return (
        np.array(X_past),
        np.array(X_future),
        np.array(Y_target),
        np.array(Y_baseline),
    )


# =====================================================
# LOADERS
# =====================================================

def get_loaders(
    csv_path,
    batch_size=16,
    validation_days=30,
    train_end_date="2026-05-10",
):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Engineer features on full data first (no scaling yet)
    df = _engineer_features(df)
    cutoff_date = pd.to_datetime(train_end_date).date()
    df = df[df["date"] <= cutoff_date].copy().reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())

    if len(unique_dates) <= validation_days:
        raise ValueError("Not enough unique dates for train/validation split.")

    train_dates = unique_dates[:-validation_days]
    val_dates = unique_dates[-validation_days:]

    train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)

    # Validation needs some historical context for past windows
    val_start = pd.Timestamp(val_dates[0])
    history_rows = PAST_STEPS + (TARGET_STEPS * 2)
    history_df = (
        df[df["datetime"] < val_start]
        .tail(history_rows)
        .copy()
        .reset_index(drop=True)
    )

    val_df = pd.concat(
        [
            history_df,
            df[df["date"].isin(val_dates)].copy(),
        ],
        ignore_index=True
    ).reset_index(drop=True)

    # Fit scalers ONLY on train data
    scaler = StandardScaler()
    scaler.fit(train_df[SCALE_FEATURES])

    return_scaler = StandardScaler()
    return_scaler.fit(train_df[["dam_diff"]])

    # Transform train
    train_df[SCALE_FEATURES] = scaler.transform(train_df[SCALE_FEATURES])
    train_df["dam_diff_scaled"] = return_scaler.transform(train_df[["dam_diff"]])

    # Transform validation
    val_df[SCALE_FEATURES] = scaler.transform(val_df[SCALE_FEATURES])
    val_df["dam_diff_scaled"] = return_scaler.transform(val_df[["dam_diff"]])

    # Create windows
    X_past_train, X_future_train, Y_train, B_train = create_windowed_data(
        train_df,
        past_steps=PAST_STEPS,
        target_steps=TARGET_STEPS,
        target_dates=train_dates,
    )

    X_past_val, X_future_val, Y_val, B_val = create_windowed_data(
        val_df,
        past_steps=PAST_STEPS,
        target_steps=TARGET_STEPS,
        target_dates=val_dates,
    )

    # Safety mask
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
    B_train = B_train[train_mask]

    X_past_val = X_past_val[val_mask]
    X_future_val = X_future_val[val_mask]
    Y_val = Y_val[val_mask]
    B_val = B_val[val_mask]

    if len(Y_train) == 0:
        raise ValueError("No training samples created after windowing.")
    if len(Y_val) == 0:
        raise ValueError("No validation samples created after windowing.")

    train_dataset = DAMDataset(
        X_past_train,
        X_future_train,
        Y_train,
        B_train,
    )

    val_dataset = DAMDataset(
        X_past_val,
        X_future_val,
        Y_val,
        B_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, scaler, return_scaler


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    df_path = "historical_dataset_extended.csv"

    if os.path.exists(df_path):
        train_loader, val_loader, scaler, return_scaler = get_loaders(df_path)

        joblib.dump(scaler, "dam_scaler6.joblib")
        joblib.dump(return_scaler, "dam_return_scaler6.joblib")

        print("Scalers saved:")
        print(" - dam_scaler6.joblib")
        print(" - dam_return_scaler6.joblib")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        sample_p, sample_f, sample_y, sample_b = next(iter(train_loader))
        print(f"x_past shape: {sample_p.shape}")
        print(f"x_future shape: {sample_f.shape}")
        print(f"y_target shape: {sample_y.shape}")
        print(f"baseline shape: {sample_b.shape}")
    else:
        print(f"File not found: {df_path}")
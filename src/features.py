import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "lag1", "lag2", "lag3", "lag24", "lag48", "lag168",
    "roll_mean_3", "roll_mean_24", "roll_mean_168",
    "roll_std_3", "roll_std_24",
    "hour_sin", "hour_cos", "dayofweek", "is_weekend", "temperature",
]

HISTORY_FEATURE_COLUMNS = FEATURE_COLUMNS[:11]


def create_features(df):

    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    if df["datetime"].isna().any():
        raise ValueError("datetime contains invalid or missing values")

    if df["datetime"].duplicated().any():
        duplicates = df.loc[df["datetime"].duplicated(False), "datetime"]
        raise ValueError(
            "Duplicate timestamps are not allowed: "
            + ", ".join(duplicates.astype(str).head(5))
        )

    # SORT
    df = df.sort_values(
        "datetime"
    )

    df = df.set_index(
        "datetime"
    )

    # TARGET
    target = "imbalance"

    # LAG FEATURES
    # 1 orë më parë
    df["lag1"] = (
        df[target]
        .shift(freq="1h")
    )

    # 2 orë më parë
    df["lag2"] = (
        df[target]
        .shift(freq="2h")
    )

    # 3 orë më parë
    df["lag3"] = (
        df[target]
        .shift(freq="3h")
    )

    # E njëjta orë një ditë më parë
    df["lag24"] = (
        df[target]
        .shift(freq="24h")
    )

    # Dy ditë më parë
    df["lag48"] = (
        df[target]
        .shift(freq="48h")
    )

    # E njëjta orë një javë më parë
    df["lag168"] = (
        df[target]
        .shift(freq="168h")
    )

    # ROLLING FEATURES
    # Mesatarja e 3 orëve të kaluara
    df["roll_mean_3"] = (
        df[target]
        .shift(freq="1h")
        .rolling("3h")
        .mean()
    )

    # Mesatarja e 24 orëve të kaluara
    df["roll_mean_24"] = (
        df[target]
        .shift(freq="1h")
        .rolling("24h")
        .mean()
    )

    # Mesatarja e 168 orëve të kaluara
    df["roll_mean_168"] = (
        df[target]
        .shift(freq="1h")
        .rolling("168h")
        .mean()
    )

    # Devijimi standard 3 orët e kaluara
    df["roll_std_3"] = (
        df[target]
        .shift(freq="1h")
        .rolling("3h")
        .std()
    )

    # Devijimi standard 24 orët e kaluara
    df["roll_std_24"] = (
        df[target]
        .shift(freq="1h")
        .rolling("24h")
        .std()
    )

    # TIME FEATURES
    df["hour_sin"] = (
        np.sin(
            2 * np.pi * df.index.hour / 24
        )
    )

    df["hour_cos"] = (
        np.cos(
            2 * np.pi * df.index.hour / 24
        )
    )

    # 0 = Monday ... 6 = Sunday
    df["dayofweek"] = (
        df.index.dayofweek
    )

    # Weekend
    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    # RESET INDEX
    df = df.reset_index()

    return df


def validate_feature_frame(frame):
    """Return model features in the canonical order after strict validation."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    result = frame.loc[:, FEATURE_COLUMNS]
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError("Model features contain missing values: " + ", ".join(bad))
    return result

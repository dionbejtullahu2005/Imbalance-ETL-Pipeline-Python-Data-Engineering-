import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # Sigurohu që është renditur sipas kohës
    df = df.sort_values(
        "datetime"
    )

    df = df.set_index(
        "datetime"
    )

    # LAG FEATURES
    df["lag1"] = (
        df["new_actual"]
        .shift(
            freq="1h"
        )
    )

    df["lag2"] = (
        df["new_actual"]
        .shift(
            freq="2h"
        )
    )

    df["lag3"] = (
        df["new_actual"]
        .shift(
            freq="3h"
        )
    )

    # E njejta ore dje
    df["lag24"] = (
        df["new_actual"]
        .shift(
            freq="24h"
        )
    )

    # Dy dite me pare
    df["lag48"] = (
        df["new_actual"]
        .shift(
            freq="48h"
        )
    )

    # Java e kaluar
    df["lag168"] = (
        df["new_actual"]
        .shift(
            freq="168h"
        )
    )

    # ROLLING FEATURES
    df["roll_mean_3"] = (
        df["new_actual"]
        .shift(freq="1h")
        .rolling("3h")
        .mean()
    )

    df["roll_mean_24"] = (
        df["new_actual"]
        .shift(freq="1h")
        .rolling("24h")
        .mean()
    )

    df["roll_mean_168"] = (
        df["new_actual"]
        .shift(freq="1h")
        .rolling("168h")
        .mean()
    )

    df["roll_std_3"] = (
        df["new_actual"]
        .shift(freq="1h")
        .rolling("3h")
        .std()
    )

    df["roll_std_24"] = (
        df["new_actual"]
        .shift(freq="1h")
        .rolling("24h")
        .std()
    )

    # TIME FEATURES
    df["hour_sin"] = (
        np.sin(
            2*np.pi*df.index.hour/24
        )
    )

    df["hour_cos"] = (
        np.cos(
            2*np.pi*df.index.hour/24
        )
    )

    df["dayofweek"] = (
        df.index.dayofweek
    )

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    # Kthe datetime si kolonë
    df = df.reset_index()

    return df
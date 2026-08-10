import pandas as pd
import numpy as np

def detect_anomalies(
        df,
        threshold=3
):

    result = df.copy()

    #Z score
    mean_delta = (
        result["percent_delta"]
        .mean()
    )

    std_delta = (
        result["percent_delta"]
        .std()
    )

    result["z_score"] = (
        (
        result["percent_delta"]
        -
        mean_delta
        )
        /
        std_delta
    )

    #identifikimi
    anomalies = result[
        result["z_score"].abs()
        >
        threshold
    ]

    anomalies = anomalies.sort_values(
        "z_score",
        ascending=False
    )

    return anomalies
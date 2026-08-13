import pandas as pd
import numpy as np

def detect_anomalies(
        df,
        threshold=3,
        method="mad"
):

    result = df.copy()

    values = pd.to_numeric(result["percent_delta"], errors="coerce")
    if method == "mad":
        center = values.median()
        mad = (values - center).abs().median()
        scale = 1.4826 * mad
        result["z_score"] = 0.0 if not np.isfinite(scale) or scale == 0 else (values - center) / scale
        reason = "robust_mad"
    elif method == "zscore":
        center = values.mean()
        scale = values.std()
        result["z_score"] = 0.0 if not np.isfinite(scale) or scale == 0 else (values - center) / scale
        reason = "zscore"
    else:
        raise ValueError("method must be 'mad' or 'zscore'")

    result["anomaly_severity"] = result["z_score"].abs()
    result["anomaly_reason"] = reason

    #identifikimi
    anomalies = result[
        result["anomaly_severity"]
        >
        threshold
    ]

    anomalies = anomalies.sort_values(
        "anomaly_severity",
        ascending=False
    )

    return anomalies

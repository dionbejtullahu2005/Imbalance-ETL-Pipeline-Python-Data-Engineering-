import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation import recursive_predict
from src.features import FEATURE_COLUMNS, create_features


def test_recursive_predictions_ignore_horizon_actuals():
    times = pd.date_range("2026-01-01", periods=260, freq="h")
    raw = pd.DataFrame({
        "datetime": times,
        "imbalance": np.sin(np.arange(260) / 12),
        "temperature": 15 + np.cos(np.arange(260) / 24),
    })
    featured = create_features(raw).dropna(subset=FEATURE_COLUMNS)
    model = LinearRegression().fit(featured[FEATURE_COLUMNS], featured["imbalance"])
    history = raw.iloc[:240]
    horizon_a = raw.iloc[240:].copy()
    horizon_b = horizon_a.copy()
    horizon_b["imbalance"] = 1_000_000.0
    first = recursive_predict(model, history, horizon_a)
    second = recursive_predict(model, history, horizon_b)
    np.testing.assert_allclose(first, second)

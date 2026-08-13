import numpy as np
import pandas as pd
import pytest

from src.features import create_features


def test_current_target_never_changes_current_features():
    times = pd.date_range("2026-01-01", periods=200, freq="h")
    base = pd.DataFrame({
        "datetime": times,
        "imbalance": np.arange(200, dtype=float),
        "temperature": 10.0,
    })
    changed = base.copy()
    changed.loc[199, "imbalance"] = 999999.0
    left = create_features(base).iloc[-1]
    right = create_features(changed).iloc[-1]
    feature_names = [column for column in left.index if column.startswith(("lag", "roll_"))]
    assert left[feature_names].tolist() == pytest.approx(right[feature_names].tolist())


def test_duplicate_timestamps_are_rejected():
    frame = pd.DataFrame({
        "datetime": ["2026-01-01 00:00", "2026-01-01 00:00"],
        "imbalance": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        create_features(frame)

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import FEATURE_COLUMNS
from src.models import train_final_model


def test_final_model_is_explicitly_selected(tmp_path, monkeypatch):
    import src.models as models_module

    monkeypatch.setattr(models_module, "MODEL_DIR", tmp_path)
    rows = 20
    frame = pd.DataFrame({column: np.arange(rows, dtype=float) for column in FEATURE_COLUMNS})
    frame["imbalance"] = np.arange(rows, dtype=float)
    frame["datetime"] = pd.date_range("2026-01-01", periods=rows, freq="h")
    fitted = GradientBoostingRegressor(random_state=42).fit(
        frame[FEATURE_COLUMNS], frame["imbalance"]
    )
    model, path = train_final_model(
        frame, model_name="gradient_boosting", trained_model=fitted
    )
    metadata = json.loads((tmp_path / "final_model_metadata.json").read_text())
    assert isinstance(model, GradientBoostingRegressor)
    assert path.name == "final_model.pkl"
    assert metadata["model_name"] == "gradient_boosting"

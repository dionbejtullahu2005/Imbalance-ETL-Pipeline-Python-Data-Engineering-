import pytest

from src.extract import extract_excel
from src.transform import transform


def test_transform_metrics():
    hourly, _, _ = extract_excel()
    result = transform(hourly.head(2))
    assert result.loc[0, "imbalance"] == pytest.approx(
        result.loc[0, "imbalance_calculated"]
    )
    assert result.loc[0, "total_euro"] == pytest.approx(
        result.loc[0, "total_euro_calculated"]
    )
    assert result.loc[0, "plan_dev"] == pytest.approx(
        result.loc[0, "plan_dev_calculated"]
    )

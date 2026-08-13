import pandas as pd
import requests

import src.weather as weather_module


def test_future_period_survives_archive_unavailability(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.HTTPError("archive unavailable")

    live = pd.DataFrame({
        "datetime": pd.date_range(
            "2026-08-12", periods=24, freq="h", tz=weather_module.TIMEZONE
        ),
        "temperature": 25.0,
        "weather_source": "open_meteo",
        "weather_type": "forecast",
    })
    monkeypatch.setattr(weather_module, "get_historical_temperature", unavailable)
    monkeypatch.setattr(weather_module, "get_forecast_temperature", lambda: live)
    monkeypatch.setattr(
        weather_module.pd.Timestamp,
        "now",
        classmethod(lambda cls, tz=None: pd.Timestamp("2026-08-12", tz=tz)),
    )

    result = weather_module.get_temperature_for_period("2026-08-01", "2026-08-31")
    assert len(result) == 744
    assert result["temperature"].notna().sum() == 24
    assert result["temperature"].isna().sum() == 720
    assert result.attrs["fallback_hours"] == 720

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LATITUDE = 42.6629
LONGITUDE = 21.1655

TIMEZONE = "Europe/Tirane"


def _session():
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _weather_frame(data, weather_type):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    if len(times) != len(temperatures) or not times:
        raise ValueError("Weather response has invalid hourly arrays")
    weather = pd.DataFrame({
        "datetime": pd.to_datetime(times),
        "temperature": temperatures,
    })
    weather["datetime"] = weather["datetime"].dt.tz_localize(TIMEZONE)
    weather["weather_source"] = "open_meteo"
    weather["weather_type"] = weather_type
    return weather


# HISTORICAL WEATHER
def get_historical_temperature(
    start_date,
    end_date
):

    url = (
        "https://archive-api.open-meteo.com"
        "/v1/archive"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date": start_date,
        "end_date": end_date,

        "hourly": "temperature_2m",

        "timezone": TIMEZONE
    }

    response = _session().get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()


    return _weather_frame(data, "observed")


# FUTURE WEATHER FORECAST
def get_forecast_temperature():

    url = (
        "https://api.open-meteo.com"
        "/v1/forecast"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": "temperature_2m",

        "timezone": TIMEZONE,

        "forecast_days": 16
    }

    response = _session().get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()


    weather = _weather_frame(data, "forecast")
    weather["forecast_issue_time"] = pd.Timestamp.now(tz=TIMEZONE)
    return weather


def get_temperature_for_period(start_date, end_date):
    """Return available observed/live weather for any requested period.

    Hours outside API availability are returned with missing temperature and
    ``weather_type='climatology_fallback_required'``. The forecasting layer
    fills those hours from training-only hourly climatology, instead of
    failing or presenting fabricated values as a live weather forecast.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("Weather end_date must not precede start_date")

    requested = pd.DataFrame({
        "datetime": pd.date_range(
            start=start.tz_localize(TIMEZONE),
            end=(end + pd.Timedelta(days=1)).tz_localize(TIMEZONE),
            freq="h",
            inclusive="left",
        )
    })
    pieces = []
    today = pd.Timestamp.now(tz=TIMEZONE).normalize().tz_localize(None)

    historical_end = min(end, today - pd.Timedelta(days=1))
    if start <= historical_end:
        try:
            pieces.append(get_historical_temperature(
                start.strftime("%Y-%m-%d"),
                historical_end.strftime("%Y-%m-%d"),
            ))
        except requests.RequestException:
            # Very recent dates may not yet exist in the archive API.
            pass

    if end >= today:
        try:
            pieces.append(get_forecast_temperature())
        except requests.RequestException:
            pass

    if pieces:
        available = pd.concat(pieces, ignore_index=True)
        available = available.sort_values("datetime").drop_duplicates(
            "datetime", keep="last"
        )
        result = requested.merge(available, on="datetime", how="left")
    else:
        result = requested.copy()
        result["temperature"] = pd.NA
        result["weather_source"] = pd.NA
        result["weather_type"] = pd.NA

    missing = result["temperature"].isna()
    result.loc[missing, "weather_source"] = "training_climatology"
    result.loc[missing, "weather_type"] = "climatology_fallback_required"
    result.attrs["available_hours"] = int((~missing).sum())
    result.attrs["fallback_hours"] = int(missing.sum())
    return result

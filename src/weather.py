import pandas as pd
import requests


LATITUDE = 42.6629
LONGITUDE = 21.1655

TIMEZONE = "Europe/Tirane"


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

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    data = response.json()

    weather = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    data["hourly"]["time"]
                ),

            "temperature":
                data["hourly"]["temperature_2m"]
        }
    )

    # timezone-aware
    weather["datetime"] = (
        weather["datetime"]
        .dt.tz_localize(
            TIMEZONE
        )
    )


    return weather


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

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    data = response.json()


    weather = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    data["hourly"]["time"]
                ),

            "temperature":
                data["hourly"]["temperature_2m"]
        }
    )

    weather["datetime"] = (
        weather["datetime"]
        .dt.tz_localize(
            TIMEZONE
        )
    )

    return weather

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from zoneinfo import ZoneInfo
from src.features import create_features
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    /
    "output"
    /
    "models"
    /
    "linear_regression_final.pkl"
)


OUTPUT_DIR = (
    BASE_DIR
    /
    "output"
    /
    "forecast"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


try:
    TIMEZONE = ZoneInfo("Europe/Pristina")

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("Europe/Tirane")



FEATURE_COLUMNS = [

    "lag1",
    "lag2",
    "lag3",

    "lag24",
    "lag48",
    "lag168",

    "roll_mean_3",
    "roll_mean_24",
    "roll_mean_168",

    "roll_std_3",
    "roll_std_24",

    "hour_sin",
    "hour_cos",

    "dayofweek",
    "is_weekend"
]



# CREATE FUTURE DATEs
def create_future_dates(
        year,
        month
):

    start = pd.Timestamp(
        year=year,
        month=month,
        day=1,
        tz=TIMEZONE
    )


    end = (
        start
        +
        pd.offsets.MonthEnd(1)
    )


    dates = pd.date_range(
        start=start,
        end=end,
        freq="h",
        inclusive="left"
    )


    future = pd.DataFrame(
        {
            "datetime": dates
        }
    )


    future["date"] = (
        future["datetime"]
        .dt.date
    )


    future["hour"] = (
        future["datetime"]
        .dt.hour
        +
        1
    )


    return future



# FORECAST
def forecast_month(
        history_df,
        year,
        month
):


    print(
        f"FORECAST {year}-{month:02d}"
    )


    model = joblib.load(
        MODEL_PATH
    )



    future = create_future_dates(
        year,
        month
    )


    predictions = []



    history = history_df[
        [
            "datetime",
            "new_actual"
        ]
    ].copy()



    history = history.sort_values(
        "datetime"
    )



    for _, row in future.iterrows():


        current_time = row["datetime"]



        temp = pd.concat(
            [

                history,

                pd.DataFrame(
                    {
                        "datetime":
                        [
                            current_time
                        ],

                        "new_actual":
                        [
                            np.nan
                        ]
                    }
                )

            ],

            ignore_index=True
        )



        # krijo features
        features = create_features(
            temp
        )



        current = (
            features
            .iloc[-1]
        )



        X = pd.DataFrame(
            [
                current[FEATURE_COLUMNS]
            ]
        )



        # mbush mungesat
        X = X.fillna(
            history_df["new_actual"]
            .median()
        )



        # ML prediction

        ml_prediction = (
            model
            .predict(X)[0]
        )



        # Weekly seasonal pattern

        seasonal_prediction = (
            current["lag168"]
        )



        if pd.isna(
            seasonal_prediction
        ):

            seasonal_prediction = (
                current["roll_mean_24"]
            )



        if pd.isna(
            seasonal_prediction
        ):

            seasonal_prediction = (
                ml_prediction
            )



        # Hybrid model

        final_prediction = (

            0.7 *
            seasonal_prediction

            +

            0.3 *
            ml_prediction

        )



        predictions.append(
            final_prediction
        )



        # shto forecast ne histori
        history = pd.concat(
            [

                history,

                pd.DataFrame(
                    {
                        "datetime":
                        [
                            current_time
                        ],

                        "new_actual":
                        [
                            final_prediction
                        ]
                    }
                )

            ],

            ignore_index=True
        )



    future["predicted_MWh"] = predictions



    output_file = (
        OUTPUT_DIR
        /
        f"{year}_{month:02d}_hybrid_forecast.csv"
    )



    future.to_csv(
        output_file,
        index=False
    )



    print(
        "Forecast saved:",
        output_file
    )



    return (
        future,
        output_file
    )

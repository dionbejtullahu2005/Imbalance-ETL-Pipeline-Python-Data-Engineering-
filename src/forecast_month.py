import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.features import create_features


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

# TIMEZONE
try:
    TIMEZONE = ZoneInfo("Europe/Pristina")

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("Europe/Tirane")


# MODEL FEATURES
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
    "is_weekend",

    "temperature"

]


# CREATE FUTURE DATES
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

    now = pd.Timestamp.now(
        tz=TIMEZONE
    ).floor("h")

    # Nëse po parashikojmë muajin aktual,
    # fillojmë nga ora aktuale.
    if (
        year == now.year
        and month == now.month
    ):

        start = max(
            start,
            now
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


# FORECAST MONTH

def forecast_month(
    history_df,
    year,
    month,
    error_distribution,
    weather_df
):

    print(
        f"FORECAST {year}-{month:02d}"
    )

    # EXPECTED HISTORICAL FORECAST ERROR

    expected_error_mwh = np.mean(
        np.abs(
            error_distribution
        )
    )

    # DYNAMIC HISTORICAL IMBALANCE PRICE
    hourly_price = (
        history_df
        .groupby("hour")["price"]
        .mean()
    )

    # Fallback nëse ndonjë orë nuk ka të dhëna
    average_imbalance_price = (
        history_df["price"]
        .mean()
    )

    if pd.isna(
        average_imbalance_price
    ):

        average_imbalance_price = 0.0

    # ERROR DISTRIBUTIOn
    lower_error = np.percentile(
        error_distribution,
        5
    )

    upper_error = np.percentile(
        error_distribution,
        95
    )

    # LOAD MODEL
    model = joblib.load(
        MODEL_PATH
    )

    # CREATE FUTURE DATES
    future = create_future_dates(
        year,
        month
    )

    # MERGE WEATHER
    future = future.merge(
        weather_df,
        on="datetime",
        how="left"
    )

    missing_temperature = (
        future["temperature"]
        .isna()
        .sum()
    )

    if missing_temperature > 0:

        print(
            f"WARNING: {missing_temperature} forecast hours "
            "do not have weather forecast data."
        )

    # PREDICTION LISTS
    predictions = []

    lower_predictions = []

    upper_predictions = []

    estimated_prices = []

    # HISTORY FOR RECURSIVE FORECAST
    history = history_df[
        [
            "datetime",
            "new_actual",
            "temperature"
        ]
    ].copy()

    history = history.sort_values(
        "datetime"
    )

    # HOURLY FORECAST
    for _, row in future.iterrows():

        current_time = (
            row["datetime"]
        )

        # Add current forecast timestamp
        temp = pd.concat(
            [
                history,

                pd.DataFrame(
                    {
                        "datetime": [
                            current_time
                        ],

                        "new_actual": [
                            np.nan
                        ],

                        "temperature": [
                            row["temperature"]
                        ]
                    }
                )
            ],
            ignore_index=True
        )

        # Create features
        features = create_features(
            temp
        )

        current = (
            features
            .iloc[-1]
        )

        # Prepare ML input
        X = pd.DataFrame(
            [
                current[
                    FEATURE_COLUMNS
                ]
            ]
        )

        # Fill missing values
        X = X.fillna(
            history_df[
                "new_actual"
            ].median()
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

        # Hybrid prediction
        final_prediction = (
            0.7
            *
            seasonal_prediction
            +
            0.3
            *
            ml_prediction
        )

        # Prediction interval
        lower_prediction = (
            final_prediction
            +
            lower_error
        )

        upper_prediction = (
            final_prediction
            +
            upper_error
        )

        # Save predictions
        predictions.append(
            final_prediction
        )

        lower_predictions.append(
            lower_prediction
        )

        upper_predictions.append(
            upper_prediction
        )

        # Dynamic imbalance price
        estimated_price = (
            hourly_price.get(
                row["hour"],
                average_imbalance_price
            )
        )

        if pd.isna(
            estimated_price
        ):

            estimated_price = (
                average_imbalance_price
            )

        estimated_prices.append(
            estimated_price
        )

        # Add forecast to history
        history = pd.concat(
            [
                history,

                pd.DataFrame(
                    {
                        "datetime": [
                            current_time
                        ],

                        "new_actual": [
                            final_prediction
                        ],

                        "temperature": [
                            row["temperature"]
                        ]
                    }
                )
            ],
            ignore_index=True
        )

    # SAVE FORECAST RESULTS
    future["predicted_MWh"] = (
        predictions
    )

    future["lower_bound_MWh"] = (
        lower_predictions
    )

    future["upper_bound_MWh"] = (
        upper_predictions
    )

    future["confidence_interval"] = (
        "90%"
    )

    # EXPECTED ERROR
    future["expected_error_MWh"] = (
        expected_error_mwh
    )

    # DYNAMIC IMBALANCE PRICE
    future[
        "estimated_imbalance_price_EUR_MWh"
    ] = estimated_prices

    # HOURLY ESTIMATED ERROR COST
    future[
        "estimated_error_cost_EUR"
    ] = (
        future[
            "expected_error_MWh"
        ]
        *
        future[
            "estimated_imbalance_price_EUR_MWh"
        ]
    )

    # TOTAL EXPECTED ERROR COST
    total_expected_error_cost = (
        future[
            "estimated_error_cost_EUR"
        ].sum()
    )

    # SAVE CSV
    output_file = (
        OUTPUT_DIR
        /
        f"{year}_{month:02d}_hybrid_forecast.csv"
    )

    future.to_csv(
        output_file,
        index=False
    )

    # FINAL OUTPUT
    print(
        f"Estimated Total Forecast Error Cost: "
        f"{total_expected_error_cost:.2f} EUR"
    )

    print(
        "Forecast saved:",
        output_file
    )

    return future, output_file

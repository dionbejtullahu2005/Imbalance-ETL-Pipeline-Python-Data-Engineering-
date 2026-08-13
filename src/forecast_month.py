import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.features import create_features, FEATURE_COLUMNS, HISTORY_FEATURE_COLUMNS, validate_feature_frame
from src.evaluation import recursive_predict

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "forecast"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LINEAR_MODEL_PATH = (
    BASE_DIR
    / "output"
    / "models"
    / "linear_regression.pkl"
)

RF_MODEL_PATH = (
    BASE_DIR
    / "output"
    / "models"
    / "random_forest.pkl"
)

GB_MODEL_PATH = (
    BASE_DIR
    / "output"
    / "models"
    / "gradient_boosting.pkl"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

def load_forecast_models():
    """Load artifacts explicitly; importing this module has no model I/O."""
    return {
        "linear_regression": joblib.load(LINEAR_MODEL_PATH),
        "random_forest": joblib.load(RF_MODEL_PATH),
        "gradient_boosting": joblib.load(GB_MODEL_PATH),
    }


# ==========================================================
# TIMEZONE
# ==========================================================

try:
    TIMEZONE = ZoneInfo(
        "Europe/Pristina"
    )

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo(
        "Europe/Tirane"
    )


# ==========================================================
# MODEL FEATURES
# ==========================================================

# ==========================================================
# CREATE FUTURE DATES
# ==========================================================

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

    # Fillimi i muajit pasues
    end = (
        start
        + pd.offsets.MonthBegin(1)
    )

    dates = pd.date_range(
        start=start,
        end=end,
        freq="h",
        inclusive="left"
    )

    # Pipeline-i përdor datetime pa timezone
    dates = dates.tz_localize(
        None
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

    # Excel përdor orët 1-24
    future["hour"] = (
        future["datetime"]
        .dt.hour
        + 1
    )

    return future


# ==========================================================
# FORECAST MONTH
# ==========================================================

def forecast_month(
    history_df,
    year,
    month,
    weather_df,
    global_conformal_radius,
    hourly_conformal_radii,
    expected_error_mwh,
    selected_model=None,
    selected_model_name="gradient_boosting",
    models=None,
    weather_mode="unspecified"
):

    print(
        f"FORECAST {year}-{month:02d}"
    )

    models = load_forecast_models() if models is None else dict(models)
    if selected_model is None:
        if selected_model_name not in models:
            raise ValueError(f"Unknown selected model: {selected_model_name}")
        selected_model = models[selected_model_name]
    linear_model = models.get("linear_regression")
    random_forest_model = models.get("random_forest")
    gradient_boosting_model = models.get("gradient_boosting")


    # ======================================================
    # VALIDATION
    # ======================================================

    required_history_columns = [
        "datetime",
        "imbalance",
        "price",
        "temperature",
        "hour"
    ]

    missing_columns = [
        column
        for column in required_history_columns
        if column not in history_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing forecast history columns: "
            + ", ".join(
                missing_columns
            )
        )


    # ======================================================
    # DYNAMIC HISTORICAL PRICE
    # ======================================================

    hourly_price = (
        history_df
        .groupby(
            "hour"
        )[
            "price"
        ]
        .mean()
    )

    average_imbalance_price = (
        history_df[
            "price"
        ]
        .mean()
    )

    if pd.isna(
        average_imbalance_price
    ):

        average_imbalance_price = 0.0


    # ======================================================
    # CREATE FORECAST PERIOD
    # ======================================================

    future = create_future_dates(
        year,
        month
    )


    # ======================================================
    # PREPARE WEATHER
    # ======================================================

    weather_df = (
        weather_df
        .copy()
    )

    weather_df["datetime"] = (
        pd.to_datetime(
            weather_df[
                "datetime"
            ],
            errors="coerce"
        )
    )

    if (
        weather_df[
            "datetime"
        ]
        .dt.tz
        is not None
    ):

        weather_df[
            "datetime"
        ] = (
            weather_df[
                "datetime"
            ]
            .dt.tz_localize(
                None
            )
        )

    weather_df = (
        weather_df
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last"
        )
    )


    # ======================================================
    # MERGE WEATHER
    # ======================================================

    future = future.merge(
        weather_df[
            [
                "datetime",
                "temperature"
            ]
        ],
        on="datetime",
        how="left"
    )


    # ======================================================
    # TEMPERATURE FALLBACK
    # ======================================================

    historical_temperature_by_hour = (
        history_df
        .groupby(
            "hour"
        )[
            "temperature"
        ]
        .median()
    )

    overall_temperature_median = (
        history_df[
            "temperature"
        ]
        .median()
    )

    if pd.isna(
        overall_temperature_median
    ):

        overall_temperature_median = 20.0


    missing_temperature = (
        future[
            "temperature"
        ]
        .isna()
    )

    if missing_temperature.any():

        missing_count = int(
            missing_temperature.sum()
        )

        print(
            f"WARNING: {missing_count} forecast hours "
            "do not have temperature data."
        )

        for index in future[
            missing_temperature
        ].index:

            hour = (
                future.at[
                    index,
                    "hour"
                ]
            )

            fallback_temperature = (
                historical_temperature_by_hour
                .get(
                    hour,
                    overall_temperature_median
                )
            )

            if pd.isna(
                fallback_temperature
            ):

                fallback_temperature = (
                    overall_temperature_median
                )

            future.at[
                index,
                "temperature"
            ] = fallback_temperature


    # ======================================================
    # HISTORY
    # ======================================================

    history = history_df[
        [
            "datetime",
            "imbalance",
            "temperature"
        ]
    ].copy()

    history["datetime"] = (
        pd.to_datetime(
            history[
                "datetime"
            ],
            errors="coerce"
        )
    )

    if (
        history[
            "datetime"
        ]
        .dt.tz
        is not None
    ):

        history[
            "datetime"
        ] = (
            history[
                "datetime"
            ]
            .dt.tz_localize(
                None
            )
        )

    history = (
        history
        .dropna(
            subset=[
                "datetime",
                "imbalance"
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if history.empty:
        raise ValueError("Forecast history is empty")
    if history["datetime"].duplicated().any():
        raise ValueError("Forecast history contains duplicate timestamps")
    if future["datetime"].min() <= history["datetime"].max():
        raise ValueError("Forecast period overlaps historical observations")
    if len(history) < 168:
        raise ValueError("At least 168 hourly historical observations are required")


    # ======================================================
    # FALLBACK VALUES
    # ======================================================

    historical_imbalance_median = (
        history_df[
            "imbalance"
        ]
        .median()
    )

    if pd.isna(
        historical_imbalance_median
    ):

        historical_imbalance_median = 0.0


    # ======================================================
    # RESULT LISTS
    # ======================================================

    linear_predictions = []

    random_forest_predictions = []

    gradient_boosting_predictions = []

    seasonal_predictions = []

    final_predictions = []

    lower_predictions = []

    upper_predictions = []

    estimated_prices = []

    interval_radii = []


    # ======================================================
    # RECURSIVE HOURLY FORECAST
    # ======================================================

    for _, row in future.iterrows():

        current_time = (
            row[
                "datetime"
            ]
        )

        current_temperature = (
            row[
                "temperature"
            ]
        )


        # --------------------------------------------------
        # Current row without known imbalance
        # --------------------------------------------------

        current_row = pd.DataFrame(
            {
                "datetime": [
                    current_time
                ],

                "imbalance": [
                    np.nan
                ],

                "temperature": [
                    current_temperature
                ]
            }
        )

        feature_history = (
            history
            .tail(168)
            .copy()
        )

        temp = pd.concat(
            [
                history,
                current_row
            ],
            ignore_index=True
        )


        # --------------------------------------------------
        # Create features
        # --------------------------------------------------

        features = create_features(
            temp
        )

        current = (
            features
            .iloc[-1]
        )


        # --------------------------------------------------
        # Prepare model input
        # --------------------------------------------------

        X = pd.DataFrame(
            [
                {
                    column:
                    current.get(
                        column,
                        np.nan
                    )
                    for column
                    in FEATURE_COLUMNS
                }
            ]
        )


        # --------------------------------------------------
        # Fill lag/rolling missing values
        # --------------------------------------------------

        missing_history_features = [
            column for column in HISTORY_FEATURE_COLUMNS
            if pd.isna(X.at[0, column])
        ]
        if missing_history_features:
            raise ValueError(
                "Insufficient contiguous history for features: "
                + ", ".join(missing_history_features)
            )


        # --------------------------------------------------
        # Temperature fallback
        # --------------------------------------------------

        if pd.isna(
            X.at[
                0,
                "temperature"
            ]
        ):

            X.at[
                0,
                "temperature"
            ] = (
                overall_temperature_median
            )


        # --------------------------------------------------
        # Time features fallback
        # --------------------------------------------------

        time_columns = [
            "hour_sin",
            "hour_cos",
            "dayofweek",
            "is_weekend"
        ]

        X[
            time_columns
        ] = (
            X[
                time_columns
            ]
            .fillna(
                0
            )
        )


        # --------------------------------------------------
        # Correct feature order
        # --------------------------------------------------

        X = validate_feature_frame(X)


        # ==================================================
        # MODEL PREDICTIONS
        # ==================================================

        linear_prediction = (
            float(linear_model.predict(X)[0]) if linear_model is not None else np.nan
        )

        random_forest_prediction = (
            float(random_forest_model.predict(X)[0])
            if random_forest_model is not None else np.nan
        )

        gradient_boosting_prediction = (
            float(gradient_boosting_model.predict(X)[0])
            if gradient_boosting_model is not None else np.nan
        )


        # ==================================================
        # SEASONAL PREDICTION
        # ==================================================

        seasonal_prediction = (
            current.get(
                "lag168",
                np.nan
            )
        )

        if pd.isna(
            seasonal_prediction
        ):

            seasonal_prediction = (
                current.get(
                    "lag24",
                    np.nan
                )
            )

        if pd.isna(
            seasonal_prediction
        ):

            seasonal_prediction = (
                current.get(
                    "roll_mean_24",
                    np.nan
                )
            )

        if pd.isna(
            seasonal_prediction
        ):

            seasonal_prediction = (
                gradient_boosting_prediction
            )

        seasonal_prediction = float(
            seasonal_prediction
        )


        # ==================================================
        # FINAL MODEL
        #
        # July backtest winner:
        # Gradient Boosting
        # ==================================================

        final_prediction = float(selected_model.predict(X)[0])

        hour_radius = (
            hourly_conformal_radii
            .get(
                int(
                    row["hour"]
                ),
                global_conformal_radius
            )
        )

        # ==================================================
        # 90% CONFORMAL PREDICTION INTERVAL
        # ==================================================

        lower_prediction = (
            final_prediction
            -
            hour_radius
        )

        upper_prediction = (
            final_prediction
            +
            hour_radius
        )

        # ==================================================
        # SAVE MODEL PREDICTIONS
        # ==================================================

        linear_predictions.append(
            linear_prediction
        )

        random_forest_predictions.append(
            random_forest_prediction
        )

        gradient_boosting_predictions.append(
            gradient_boosting_prediction
        )

        seasonal_predictions.append(
            seasonal_prediction
        )

        final_predictions.append(
            final_prediction
        )

        lower_predictions.append(
            lower_prediction
        )

        upper_predictions.append(
            upper_prediction
        )

        interval_radii.append(
            hour_radius
        )

        # ==================================================
        # DYNAMIC PRICE
        # ==================================================

        estimated_price = (
            hourly_price.get(
                row[
                    "hour"
                ],
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
            float(
                estimated_price
            )
        )


        # ==================================================
        # RECURSIVE HISTORY UPDATE
        # ==================================================

        history = pd.concat(
            [
                history,

                pd.DataFrame(
                    {
                        "datetime": [
                            current_time
                        ],

                        "imbalance": [
                            final_prediction
                        ],

                        "temperature": [
                            current_temperature
                        ]
                    }
                )
            ],
            ignore_index=True
        )


    # ======================================================
    # OUTPUT COLUMNS
    # ======================================================

    future[
        "linear_prediction_MWh"
    ] = (
        linear_predictions
    )

    future[
        "random_forest_prediction_MWh"
    ] = (
        random_forest_predictions
    )

    future[
        "gradient_boosting_prediction_MWh"
    ] = (
        gradient_boosting_predictions
    )

    future[
        "seasonal_prediction_MWh"
    ] = (
        seasonal_predictions
    )


    # ======================================================
    # FINAL FORECAST
    # ======================================================

    future[
        "predicted_imbalance_MWh"
    ] = (
        final_predictions
    )


    # ======================================================
    # INTERVAL
    # ======================================================

    future[
        "lower_bound_MWh"
    ] = (
        lower_predictions
    )

    future[
        "upper_bound_MWh"
    ] = (
        upper_predictions
    )

    future[
        "confidence_interval"
    ] = "nominal_90"

    future["selected_model"] = selected_model_name
    future["weather_mode"] = weather_mode


    # ======================================================
    # EXPECTED ERROR
    # ======================================================

    future[
        "expected_recursive_MAE_MWh"
    ] = (
        expected_error_mwh
    )


    # ======================================================
    # ESTIMATED DYNAMIC PRICE
    # ======================================================

    future[
        "estimated_imbalance_price_EUR_MWh"
    ] = (
        estimated_prices
    )


    # ======================================================
    # EXPECTED ERROR COST
    # ======================================================

    future[
        "estimated_error_cost_EUR"
    ] = (

        future[
            "expected_recursive_MAE_MWh"
        ]

        *

        future[
            "estimated_imbalance_price_EUR_MWh"
        ]
        .abs()
    )


    # ======================================================
    # TOTAL EXPECTED COST
    # ======================================================

    total_expected_error_cost = (
        future[
            "estimated_error_cost_EUR"
        ]
        .sum()
    )


    # ======================================================
    # SAVE CSV
    # ======================================================

    output_file = (
        OUTPUT_DIR
        /
        f"{year}_{month:02d}_forecast.csv"
    )

    future.to_csv(
        output_file,
        index=False
    )


    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    print(
        "Forecast hours:",
        len(
            future
        )
    )

    print(
        "Final model: "
        f"{selected_model_name}"
    )

    print(
        f"Expected historical MAE: "
        f"{expected_error_mwh:.6f} MWh"
    )

    print(
        f"Estimated Total Forecast Error Cost: "
        f"{total_expected_error_cost:.2f} EUR"
    )

    print(
        "Forecast saved:",
        output_file
    )


    return (
        future,
        output_file
    )


def forecast_future_months(
    history_df,
    target_year,
    target_month,
    weather_loader,
    global_conformal_radius,
    hourly_conformal_radii,
    expected_error_mwh,
    selected_model,
    selected_model_name,
    models,
):
    """
    Forecast sequential future months.

    If the final historical month is incomplete, the missing
    final hours are first generated as a recursive bridge
    forecast. This preserves lag and rolling-feature continuity.
    """

    # ======================================================
    # PREPARE HISTORY
    # ======================================================

    history = history_df.copy()

    required_history_columns = [
        "datetime",
        "imbalance",
        "temperature",
    ]

    missing_history_columns = [
        column
        for column in required_history_columns
        if column not in history.columns
    ]

    if missing_history_columns:
        raise ValueError(
            "Missing future-forecast history columns: "
            + ", ".join(missing_history_columns)
        )

    history["datetime"] = pd.to_datetime(
        history["datetime"],
        errors="coerce",
    )

    history["imbalance"] = pd.to_numeric(
        history["imbalance"],
        errors="coerce",
    )

    history["temperature"] = pd.to_numeric(
        history["temperature"],
        errors="coerce",
    )

    history = (
        history
        .dropna(
            subset=[
                "datetime",
                "imbalance",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if history.empty:
        raise ValueError(
            "History has no usable observations."
        )

    if history["datetime"].duplicated().any():
        duplicates = (
            history.loc[
                history["datetime"]
                .duplicated(keep=False),
                "datetime",
            ]
            .astype(str)
            .head(5)
            .tolist()
        )

        raise ValueError(
            "History contains duplicate timestamps: "
            + ", ".join(duplicates)
        )

    history["history_source"] = (
        history.get(
            "history_source",
            "actual",
        )
    )

    last_observation = (
        history["datetime"]
        .max()
    )

    # ======================================================
    # TARGET PERIOD
    # ======================================================

    target_period = pd.Period(
        year=target_year,
        month=target_month,
        freq="M",
    )

    historical_period = (
        last_observation.to_period("M")
    )

    if target_period <= historical_period:
        raise ValueError(
            "Target month must be after the final "
            "historical month."
        )

    # ======================================================
    # COMPLETE PARTIAL FINAL HISTORY MONTH
    # ======================================================

    history_month_end = (
        historical_period
        .end_time
        .floor("h")
    )

    bridge_history = pd.DataFrame()

    if last_observation < history_month_end:

        bridge_start = (
            last_observation
            + pd.Timedelta(hours=1)
        )

        bridge_end = history_month_end

        bridge_dates = pd.date_range(
            start=bridge_start,
            end=bridge_end,
            freq="h",
        )

        print(
            "\nIncomplete historical month detected."
        )

        print(
            "Generating bridge forecast:",
            bridge_start,
            "->",
            bridge_end,
        )

        # --------------------------------------------------
        # BRIDGE WEATHER
        # --------------------------------------------------

        bridge_weather = weather_loader(
            bridge_start.strftime(
                "%Y-%m-%d"
            ),
            bridge_end.strftime(
                "%Y-%m-%d"
            ),
        ).copy()

        required_weather_columns = [
            "datetime",
            "temperature",
        ]

        missing_weather_columns = [
            column
            for column in required_weather_columns
            if column not in bridge_weather.columns
        ]

        if missing_weather_columns:
            raise ValueError(
                "Bridge weather is missing columns: "
                + ", ".join(
                    missing_weather_columns
                )
            )

        bridge_weather["datetime"] = pd.to_datetime(
            bridge_weather["datetime"],
            errors="coerce",
        )

        if (
            bridge_weather["datetime"].dt.tz
            is not None
        ):
            bridge_weather["datetime"] = (
                bridge_weather["datetime"]
                .dt.tz_localize(None)
            )

        bridge_weather["temperature"] = (
            pd.to_numeric(
                bridge_weather["temperature"],
                errors="coerce",
            )
        )

        bridge_weather = (
            bridge_weather[
                [
                    "datetime",
                    "temperature",
                ]
            ]
            .dropna(
                subset=["datetime"]
            )
            .drop_duplicates(
                subset=["datetime"],
                keep="last",
            )
        )

        bridge = pd.DataFrame(
            {
                "datetime": bridge_dates,
            }
        )

        bridge = bridge.merge(
            bridge_weather,
            on="datetime",
            how="left",
            validate="one_to_one",
        )

        # --------------------------------------------------
        # TRAINING-ONLY TEMPERATURE CLIMATOLOGY
        # --------------------------------------------------

        temperature_history = (
            history[
                [
                    "datetime",
                    "temperature",
                ]
            ]
            .dropna(
                subset=["temperature"]
            )
            .copy()
        )

        temperature_history[
            "hour_of_day"
        ] = (
            temperature_history[
                "datetime"
            ]
            .dt.hour
        )

        historical_temperature_by_hour = (
            temperature_history
            .groupby(
                "hour_of_day"
            )["temperature"]
            .median()
        )

        overall_temperature_median = (
            temperature_history[
                "temperature"
            ]
            .median()
        )

        if pd.isna(
            overall_temperature_median
        ):
            overall_temperature_median = 20.0

        bridge["hour_of_day"] = (
            bridge["datetime"]
            .dt.hour
        )

        bridge["temperature_source"] = (
            np.where(
                bridge["temperature"].notna(),
                "weather_api",
                "training_climatology",
            )
        )

        bridge["temperature"] = (
            bridge["temperature"]
            .fillna(
                bridge["hour_of_day"]
                .map(
                    historical_temperature_by_hour
                )
            )
            .fillna(
                overall_temperature_median
            )
        )

        bridge = bridge.drop(
            columns=["hour_of_day"]
        )

        if bridge["temperature"].isna().any():
            unresolved_count = int(
                bridge["temperature"]
                .isna()
                .sum()
            )

            raise ValueError(
                "Bridge forecast contains "
                f"{unresolved_count} unresolved "
                "temperature values."
            )

        # --------------------------------------------------
        # RECURSIVE BRIDGE PREDICTIONS
        # --------------------------------------------------

        bridge_predictions = recursive_predict(
            selected_model,
            history,
            bridge,
        )

        if (
            len(bridge_predictions)
            !=
            len(bridge)
        ):
            raise ValueError(
                "Bridge prediction length does not "
                "match bridge timestamps."
            )

        bridge_history = pd.DataFrame(
            {
                "datetime":
                    bridge["datetime"],

                "imbalance":
                    bridge_predictions,

                "temperature":
                    bridge["temperature"],

                "price":
                    np.nan,

                "hour":
                    (
                        bridge["datetime"]
                        .dt.hour
                        + 1
                    ),

                "history_source":
                    "bridge_forecast",

                "temperature_source":
                    bridge[
                        "temperature_source"
                    ],
            }
        )

        history = pd.concat(
            [
                history,
                bridge_history,
            ],
            ignore_index=True,
            sort=False,
        )

        history = (
            history
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        print(
            "Bridge forecast hours:",
            len(bridge_history),
        )

    # ======================================================
    # VALIDATE HISTORY CONTINUITY
    # ======================================================

    full_history_range = pd.date_range(
        start=history["datetime"].min(),
        end=history["datetime"].max(),
        freq="h",
    )

    missing_history_hours = (
        full_history_range
        .difference(
            pd.DatetimeIndex(
                history["datetime"]
            )
        )
    )

    if len(missing_history_hours):
        preview = (
            missing_history_hours
            .astype(str)
            .tolist()[:5]
        )

        raise ValueError(
            "History remains non-contiguous after "
            "bridge generation. Missing examples: "
            + ", ".join(preview)
        )

    # ======================================================
    # FIRST MONTH TO FORECAST
    # ======================================================

    last_history_timestamp = (
        history["datetime"]
        .max()
    )

    next_period = (
        last_history_timestamp
        .to_period("M")
        + 1
    )

    if target_period < next_period:
        raise ValueError(
            "Target month precedes the first "
            "forecastable future month."
        )

    # ======================================================
    # FORECAST MONTHS SEQUENTIALLY
    # ======================================================

    target_forecast = None
    output_files = []

    for period in pd.period_range(
        next_period,
        target_period,
        freq="M",
    ):

        start_date = (
            period.start_time
            .strftime("%Y-%m-%d")
        )

        end_date = (
            period.end_time
            .normalize()
            .strftime("%Y-%m-%d")
        )

        weather = weather_loader(
            start_date,
            end_date,
        )

        if "temperature" not in weather.columns:
            raise ValueError(
                "Weather loader did not return "
                "a temperature column."
            )

        available_weather_hours = int(
            weather.attrs.get(
                "available_hours",
                weather["temperature"]
                .notna()
                .sum(),
            )
        )

        fallback_weather_hours = int(
            weather.attrs.get(
                "fallback_hours",
                weather["temperature"]
                .isna()
                .sum(),
            )
        )

        weather_mode = (
            f"available_weather="
            f"{available_weather_hours};"
            f"climatology_fallback="
            f"{fallback_weather_hours}"
        )

        print(
            "\nForecasting month:",
            str(period),
        )

        print(
            "Weather availability:",
            weather_mode,
        )

        (
            target_forecast,
            output_file,
        ) = forecast_month(
            history,
            period.year,
            period.month,
            weather_df=weather,
            global_conformal_radius=(
                global_conformal_radius
            ),
            hourly_conformal_radii=(
                hourly_conformal_radii
            ),
            expected_error_mwh=(
                expected_error_mwh
            ),
            selected_model=selected_model,
            selected_model_name=(
                selected_model_name
            ),
            models=models,
            weather_mode=weather_mode,
        )

        output_files.append(
            output_file
        )

        # --------------------------------------------------
        # EXTEND RECURSIVE HISTORY
        # --------------------------------------------------

        extension = pd.DataFrame(
            {
                "datetime":
                    target_forecast[
                        "datetime"
                    ],

                "imbalance":
                    target_forecast[
                        "predicted_imbalance_MWh"
                    ],

                "temperature":
                    target_forecast[
                        "temperature"
                    ],

                "price":
                    np.nan,

                "hour":
                    target_forecast[
                        "hour"
                    ],

                "history_source":
                    "monthly_forecast",
            }
        )

        history = pd.concat(
            [
                history,
                extension,
            ],
            ignore_index=True,
            sort=False,
        )

        history = (
            history
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    if target_forecast is None:
        raise ValueError(
            "No target forecast was generated."
        )

    if not output_files:
        raise ValueError(
            "No forecast files were generated."
        )

    return (
        target_forecast,
        output_files,
    )
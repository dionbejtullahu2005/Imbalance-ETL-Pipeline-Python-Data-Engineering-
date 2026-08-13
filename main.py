import pandas as pd

from pathlib import Path

from src.extract import extract_excel, extract_hourly_excel
from src.transform import transform
from src.validate import (
    validate_metrics,
    validate_time_series,
)
from src.load import (
    save_parquet,
    save_sqlite,
)
from src.features import (
    create_features,
    FEATURE_COLUMNS,
)
from src.models import (
    compare_models,
    train_final_model,
    train_explainable_models,
)
from src.report import (
    plot_forecast,
    plot_nomination_strategy_cost,
    generate_reports,
    plot_forecast,
    plot_nomination_strategy_cost
)
from src.forecast_month import forecast_future_months
from src.strategy_simulation import (
    simulate_nomination_strategies,
)
from src.feature_importance import (
    extract_feature_importance,
    plot_feature_importance,
)
from src.anomaly_detection import detect_anomalies
from src.weather import (
    get_historical_temperature,
    get_temperature_for_period,
)
from src.calibrate_interval import (
    calibrate_conformal_interval,
)


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent

JUNE_FILE = (
    BASE_DIR
    / "data"
    / "Imbalanc June 2026 (1).xlsx"
)

JULY_FILE = (
    BASE_DIR
    / "data"
    / "Imbalanc July 2026 (1).xlsx"
)

TRAINING_START = "2026-06-01"
TRAINING_END = "2026-07-30"

# Ndrysho vetëm këto dy vlera për muajin e ardhshëm.
FORECAST_YEAR = 2026
FORECAST_MONTH = 8

CONFORMAL_CONFIDENCE = 0.90
CONFORMAL_SPLITS = 5
MIN_HOURLY_SAMPLES = 30


# DATETIME NORMALIZATION
def normalize_datetime_column(df):
    """
    Normalize a dataframe's datetime column.

    The current Excel pipeline uses timezone-naive local market
    timestamps. Weather timestamps are converted to the same
    representation before merging.
    """

    result = df.copy()

    if "datetime" not in result.columns:
        raise ValueError(
            "Missing required datetime column."
        )

    result["datetime"] = pd.to_datetime(
        result["datetime"],
        errors="coerce",
    )

    invalid_count = int(
        result["datetime"].isna().sum()
    )

    if invalid_count:
        raise ValueError(
            f"Found {invalid_count} invalid datetime values."
        )

    if result["datetime"].dt.tz is not None:
        result["datetime"] = (
            result["datetime"]
            .dt.tz_localize(None)
        )

    return result


# DATA VALIDATION
def validate_input_data(df):
    """
    Validate transformed hourly input data.

    Raises ValueError when a critical validation fails.
    """

    time_result = validate_time_series(df)
    metric_result = validate_metrics(df)

    metrics_pass = (
        metric_result["status"]
        .eq("PASS")
        .all()
    )

    time_pass = (
        time_result["missing_hours_pass"]
        and
        time_result["duplicates_pass"]
        and
        time_result["chronological_order"]
    )

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        time_result["total_rows"],
    )

    print(
        "Days:",
        time_result["days"],
    )

    print(
        "Missing hours:",
        len(time_result["missing_hours"]),
    )

    print(
        "Duplicates:",
        time_result["duplicates"],
    )

    print(
        "Time validation:",
        "PASS" if time_pass else "FAIL",
    )

    print(
        "Metric validation:",
        "PASS" if metrics_pass else "FAIL",
    )

    if not time_pass:
        raise ValueError(
            "Critical time-series validation failed."
        )

    if not metrics_pass:
        failed_metrics = (
            metric_result.loc[
                metric_result["status"] != "PASS",
                "column",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Metric validation failed for: "
            + ", ".join(failed_metrics)
        )

    # HISTORICAL REPORTS

    hourly_report, peak_report, deviation_distribution = (
        generate_reports(df)
    )

    return time_result, metric_result

# MODEL SELECTION
def select_final_model(
    model_result,
    trained_models,
):
    """
    Select the best recursively evaluated model.

    compare_models() already sorts its result by the configured
    error-price proxy, so the first row is selected.
    """

    if model_result.empty:
        raise ValueError(
            "Model comparison returned no results."
        )

    model_key_by_label = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
    }

    selected_model_label = str(
        model_result.iloc[0]["Model"]
    )

    if selected_model_label not in model_key_by_label:
        raise ValueError(
            "Unsupported selected model label: "
            f"{selected_model_label}"
        )

    selected_model_name = (
        model_key_by_label[
            selected_model_label
        ]
    )

    if selected_model_name not in trained_models:
        raise ValueError(
            "Selected model was not trained: "
            f"{selected_model_name}"
        )

    selected_model = (
        trained_models[
            selected_model_name
        ]
    )

    return (
        selected_model_label,
        selected_model_name,
        selected_model,
    )


# TARGET MONTH HELPERS
def target_month_dates(
    year,
    month,
):
    """
    Return the first and final calendar dates of a target month.
    """

    start = pd.Timestamp(
        year=year,
        month=month,
        day=1,
    )

    end = (
        start
        + pd.offsets.MonthEnd(0)
    )

    return (
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


# MAIN PIPELINE
def main():

    forecast_start, forecast_end = (
        target_month_dates(
            FORECAST_YEAR,
            FORECAST_MONTH,
        )
    )

    print("\n" + "=" * 70)
    print("IMBALANCE FORECASTING PIPELINE")
    print("=" * 70)

    print(
        "Target month:",
        f"{FORECAST_YEAR}-{FORECAST_MONTH:02d}",
    )

    # 1. EXTRACT AND TRANSFORM
    june_raw = extract_hourly_excel(
        JUNE_FILE
    )

    july_raw = extract_hourly_excel(
        JULY_FILE
    )

    # Qershori përfshin transaksionet Uncover
    # në llogaritjen e imbalance-it.
    june_raw["imbalance_formula"] = (
        "with_uncover"
    )

    # Korriku nuk i përfshin kolonat Uncover
    # në target-in Imbalanc.
    july_raw["imbalance_formula"] = (
        "base"
    )

    df_raw = pd.concat(
        [
            june_raw,
            july_raw,
        ],
        ignore_index=True,
    )

    df_raw = (
        df_raw
        .drop_duplicates(
            subset=[
                "Date",
                "Hour",
                "Supplier",
            ],
            keep="last",
        )
        .sort_values(
            [
                "Date",
                "Hour",
            ]
        )
        .reset_index(drop=True)
    )

    df = transform(df_raw)
    df_features = create_features(df)

    df = normalize_datetime_column(df)
    df_features = normalize_datetime_column(
        df_features
    )

    # 2. TRAINING WEATHER
    print("\nLOADING TRAINING WEATHER DATA")

    training_weather = (
        get_historical_temperature(
            TRAINING_START,
            TRAINING_END,
        )
    )

    training_weather = (
        normalize_datetime_column(
            training_weather
        )
    )

    training_weather = (
        training_weather[
            [
                "datetime",
                "temperature",
            ]
        ]
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
    )

    # 3. MERGE TRAINING WEATHER

    df_features = df_features.merge(
        training_weather,
        on="datetime",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        training_weather,
        on="datetime",
        how="left",
        validate="one_to_one",
    )

    missing_training_weather = int(
        df_features["temperature"]
        .isna()
        .sum()
    )

    if missing_training_weather:
        raise ValueError(
            "Training weather is missing for "
            f"{missing_training_weather} hours."
        )

    # 4. VALIDATION

    validate_input_data(df)

    # 5. SAVE PROCESSED DATA

    parquet_file = save_parquet(df)
    sqlite_file = save_sqlite(df)

    # 6. RECURSIVE MODEL COMPARISON
    print("\n" + "=" * 70)
    print("RECURSIVE MODEL SELECTION")
    print("=" * 70)

    model_result = compare_models(
        df_features
    )

    display_columns = [
        column
        for column in [
            "Model",
            "MAE",
            "MAPE (%)",
            "Avg Error Price Proxy (€)",
            "Total Error Price Proxy (€)",
        ]
        if column in model_result.columns
    ]

    print(
        model_result[
            display_columns
        ].to_string(
            index=False
        )
    )

    # 7. TRAIN CANDIDATE MODELS

    trained_models = train_explainable_models(
        df_features
    )

    (
        selected_model_label,
        selected_model_name,
        selected_model,
    ) = select_final_model(
        model_result,
        trained_models,
    )

    # 8. PERSIST FINAL MODEL
    final_model, final_model_path = (
        train_final_model(
            df_features,
            model_name=selected_model_name,
            trained_model=selected_model,
        )
    )

    print(
        "\nSelected final model:",
        selected_model_label,
    )

    print(
        "Final model artifact:",
        final_model_path,
    )

    # 9. FINAL-MODEL FEATURE IMPORTANCE
    importance_df = extract_feature_importance(
        df_features,
        final_model,
        selected_model_name,
    )

    importance_file = plot_feature_importance(
        importance_df,
        selected_model_name,
    )

    # 10. RECURSIVE CONFORMAL CALIBRATION

    calibration = calibrate_conformal_interval(
        final_model,
        df_features,
        FEATURE_COLUMNS,
        target_column="imbalance",
        confidence=CONFORMAL_CONFIDENCE,
        splits=CONFORMAL_SPLITS,
        min_hourly_samples=MIN_HOURLY_SAMPLES,
    )

    global_conformal_radius = float(
        calibration["global_radius"]
    )

    hourly_conformal_radii = (
        calibration["hourly_radii"]
    )

    forecast_expected_error = float(
        calibration["expected_error_mwh"]
    )

    calibration_coverage = float(
        calibration["coverage"]
    )

    # 11. FORECAST TARGET AND INTERMEDIATE MONTHS

    print("\n" + "=" * 70)
    print("FUTURE MONTH FORECAST")
    print("=" * 70)

    month_forecast, forecast_files = (
        forecast_future_months(
            history_df=df_features,
            target_year=FORECAST_YEAR,
            target_month=FORECAST_MONTH,
            weather_loader=(
                get_temperature_for_period
            ),
            global_conformal_radius=(
                global_conformal_radius
            ),
            hourly_conformal_radii=(
                hourly_conformal_radii
            ),
            expected_error_mwh=(
                forecast_expected_error
            ),
            selected_model=final_model,
            selected_model_name=(
                selected_model_name
            ),
            models=trained_models,
        )
    )

    if month_forecast is None:
        raise ValueError(
            "No monthly forecast was generated."
        )

    if not forecast_files:
        raise ValueError(
            "No forecast output file was generated."
        )

    forecast_file = forecast_files[-1]

    # 12. TARGET-MONTH FORECAST GRAPH

    forecast_graph = plot_forecast(
        df,
        month_forecast,
        year=FORECAST_YEAR,
        month=FORECAST_MONTH,
    )

    # 13. NOMINATION STRATEGY

    strategy_results = (
        simulate_nomination_strategies(df)
    )

    strategy_graph = (
        plot_nomination_strategy_cost(
            strategy_results
        )
    )

    # 14. ANOMALY DETECTION

    anomalies = detect_anomalies(
        df,
        threshold=3,
        method="mad",
    )

    # 15. WEATHER COVERAGE
    weather_mode_values = (
        month_forecast["weather_mode"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    weather_mode = (
        weather_mode_values[0]
        if weather_mode_values
        else "unknown"
    )

    fallback_hours = int(
        month_forecast["temperature"]
        .isna()
        .sum()
    )

    # Temperature fallback is applied inside forecast_month(),
    # so the final forecast should contain no missing temperature.
    if fallback_hours:
        raise ValueError(
            "Forecast contains unresolved missing "
            f"temperature for {fallback_hours} hours."
        )

    # 16. FINAL SUMMARY
    estimated_cost_proxy = float(
        month_forecast[
            "estimated_error_cost_EUR"
        ]
        .sum()
    )

    print("\n" + "=" * 70)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 70)

    print(
        "Training period:",
        f"{TRAINING_START} -> {TRAINING_END}",
    )

    print(
        "Training rows:",
        len(df),
    )

    print(
        "Selected model:",
        selected_model_label,
    )

    print(
        "Final model file:",
        final_model_path,
    )

    print(
        "Conformal method:",
        "Recursive global conformal",
    )

    print(
        "Nominal coverage:",
        f"{CONFORMAL_CONFIDENCE * 100:.0f}%",
    )

    print(
        "Calibration coverage:",
        f"{calibration_coverage:.2f}%",
    )

    print(
        "Global conformal radius:",
        f"{global_conformal_radius:.6f} MWh",
    )

    print(
        "Expected recursive error:",
        f"{forecast_expected_error:.6f} MWh",
    )

    print(
        "Target forecast period:",
        f"{forecast_start} -> {forecast_end}",
    )

    print(
        "Target forecast rows:",
        len(month_forecast),
    )

    print(
        "Weather coverage:",
        weather_mode,
    )

    print(
        "Estimated error-price proxy:",
        f"{estimated_cost_proxy:.2f} EUR",
    )

    print(
        "Intermediate and target forecast files:",
        len(forecast_files),
    )

    for generated_file in forecast_files:
        print(
            "  -",
            generated_file,
        )

    print(
        "Anomalies detected:",
        len(anomalies),
    )

    print(
        "Processed Parquet:",
        parquet_file,
    )

    print(
        "Processed SQLite:",
        sqlite_file,
    )

    print(
        "Target forecast file:",
        forecast_file,
    )

    print(
        "Forecast graph:",
        forecast_graph,
    )

    print(
        "Feature importance graph:",
        importance_file,
    )

    print(
        "Strategy graph:",
        strategy_graph,
    )

    print("\nPipeline status: PASS")

    print(
        "Forecast status: GENERATED"
    )

    print(
        "Validation note: Future-month forecasts "
        "must be evaluated later when actual data "
        "becomes available."
    )


if __name__ == "__main__":
    main()

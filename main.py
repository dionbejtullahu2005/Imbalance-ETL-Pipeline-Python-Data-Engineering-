from src.extract import extract_excel
from src.transform import transform
from src.validate import (
validate_metrics,
validate_time_series
)
from src.gj_calculation import (
calculate_gj_summary
)
from src.validate_gj import (
compare_gj
)
from src.load import (
save_parquet,
save_sqlite
)
from src.features import create_features
from src.models import (
compare_models,
train_linear_prediction,
train_final_model,
train_explainable_models
)
from src.forecasting import (
rolling_average_prediction,
calculate_rolling_metrics
)
from src.report import (
generate_reports,
plot_forecasting_comparison,
plot_forecast,
plot_nomination_strategy_cost
)
from src.evaluation import (
evaluate_timeseries_model
)
from src.forecast_month import forecast_month
from src.strategy_simulation import simulate_nomination_strategies
from src.feature_importance import (
extract_feature_importance,
plot_feature_importance
)
from src.anomaly_detection import detect_anomalies
from src.weather import (
get_historical_temperature,
get_forecast_temperature
)

def main():

    # 1. EXTRACT
    df_j, df_gj = extract_excel()

    weather = get_historical_temperature(
        "2026-06-01",
        "2026-06-30"
    )

    print("\nGJ COLUMNS")
    print(df_gj.columns.tolist())

    # 2. TRANSFORM
    df = transform(df_j)

    # krijimi i lag features
    df_features = create_features(df)

    df_features = df_features.merge(
        weather,
        on="datetime",
        how="left"
    )

    df = df.merge(
        weather,
        on="datetime",
        how="left"
    )

    # 3. TIME VALIDATION
    print("\nTIME VALIDATION")

    time_result = validate_time_series(df)

    print(
        "Rows:",
        time_result["total_rows"]
    )

    print(
        "Days:",
        time_result["days"]
    )

    print(
        "Missing hours:",
        len(time_result["missing_hours"])
    )

    print(
        "Duplicates:",
        time_result["duplicates"]
    )

    # 4. METRIC VALIDATION
    print("\nHOURLY METRICS VALIDATION")

    metric_result = validate_metrics(df)

    print(metric_result)

    # 5. GJ CALCULATION
    print("\nGJ SUMMARY")

    gj_python = calculate_gj_summary(
        df,
        df_gj
    )

    print(
        "GJ items calculated:",
        len(gj_python)
    )

    print("\nGJ VALIDATION")

    gj_result = compare_gj(
        gj_python,
        df_gj
    )

    print(
        "GJ PASS:",
        (gj_result["Status"] == "PASS").sum(),
        "/",
        len(gj_result)
    )

    # 6. SAVE DATA
    print("\nLOAD DATA")

    parquet_file = save_parquet(df)

    sqlite_file = save_sqlite(df)

    print(
        "Parquet saved:",
        parquet_file
    )

    print(
        "SQLite saved:",
        sqlite_file
    )

    # 7. REPORTS
    print("\nREPORTING")

    hourly_report, peak_report, deviation_distribution = generate_reports(df)

    print("\n24H DEVIATION")
    print(hourly_report)

    print("\nPEAK HOURS 08-17")
    print(peak_report)

    print("\nDEVIATION DISTRIBUTION")
    print(deviation_distribution)

    # 8. ROLLING AVERAGE MODEL
    print("\nROLLING AVERAGE")

    df_rolling = rolling_average_prediction(
        df_features,
        window=3
    )

    rolling_result = calculate_rolling_metrics(
        df_rolling
    )

    print(rolling_result)

    # 9. TIME SERIES CROSS VALIDATION
    print("\nTIME SERIES CROSS VALIDATION")

    cv_result, cv_summary, error_distribution = evaluate_timeseries_model(
        df_features,
        splits=5
    )

    print("\nCV SUMMARY")
    print(cv_summary)

    # 10. MODEL COMPARISON
    print("\nMODEL COMPARISON")

    model_result = compare_models(
        df_features
    )

    print(model_result)

    # 11. TRAIN FINAL MODEL
    print("\nTRAIN FINAL MODEL")

    final_model, model_path = train_final_model(
        df_features
    )

    trained_models = train_explainable_models(
        df_features
    )

    trained_models["linear_regression"] = final_model

    print(
        "Model saved:",
        model_path
    )

    # 12. FEATURE IMPORTANCE
    print("\nFEATURE IMPORTANCE")

    for name, model in trained_models.items():

        importance_df = extract_feature_importance(
            df_features,
            model,
            name
        )

        print(
            "\n",
            name.upper()
        )

        print(
            importance_df
        )

        plot_feature_importance(
            importance_df,
            name
        )

    # 13. LINEAR REGRESSION PREDICTION
    print("\nLINEAR REGRESSION PREDICTION")

    linear_result, df_linear, model = train_linear_prediction(
        df_features
    )

    print(linear_result)

    # 14. FORECAST GRAPH
    print("\nGENERATING FORECAST GRAPH")

    plot_forecasting_comparison(
        df_rolling,
        df_linear
    )

    print(
        "\nGraph saved:"
        " output/reports/forecasting_comparison.png"
    )

    # 15. FORECAST WEATHER
    forecast_weather = get_forecast_temperature()

    print("\nFORECAST MONTH:")

    # 16. FORECAST MONTH
    month_forecast, file = forecast_month(
        df_features,
        2026,
        9,
        error_distribution=error_distribution,
        weather_df=forecast_weather
    )

    # 17. FORECAST REPORT
    print("\nFORECAST REPORT")

    output_file = plot_forecast(
        df,
        month_forecast,
        year=2026,
        month=9
    )

    print(
        "Forecast graph saved:",
        output_file
    )

    # 18. NOMINATION STRATEGY SIMULATION
    print(
        "\nNOMINATION STRATEGY SIMULATION"
    )

    strategy_results = simulate_nomination_strategies(
        df
    )

    plot_nomination_strategy_cost(
        strategy_results
    )

    print(
        strategy_results
    )

    # 19. ANOMALY DETECTION
    print("\nANOMALY DETECTION")

    anomalies = detect_anomalies(
        df,
        threshold=3
    )

    print(
        anomalies[
            [
                "datetime",
                "new_actual",
                "old_predicted",
                "percent_delta",
                "z_score"
            ]
        ]
    )

if __name__ == "__main__":
    main()

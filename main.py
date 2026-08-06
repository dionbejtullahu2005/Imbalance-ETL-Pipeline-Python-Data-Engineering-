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
from src.models import compare_models, train_linear_prediction, train_final_model
from src.forecasting import (
    rolling_average_prediction,
    calculate_rolling_metrics
)
from src.report import (
    generate_reports,
    plot_forecasting_comparison,
    plot_forecast
)
from src.evaluation import (
    evaluate_timeseries_model
)
from src.forecast_july import forecast_month


def main():

    # 1. EXTRACT
    df_j, df_gj = extract_excel()


    print("\nGJ COLUMNS")
    print(df_gj.columns.tolist())

    # 2. TRANSFORM
    df = transform(df_j)


    # krijimi i lag features
    df_features = create_features(df)

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
        (gj_result["Status"]=="PASS").sum(),
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

    # 10. TIME SERIES CROSS VALIDATION
    print("\nTIME SERIES CROSS VALIDATION")

    cv_result, cv_summary = evaluate_timeseries_model(
        df_features,
        splits=5
    )

    print("\nCV SUMMARY")

    print(cv_summary)

    print("\nMODEL COMPARISON")

    model_result = compare_models(
    df_features
    )

    print(model_result)

    print("\nTRAIN FINAL MODEL")

    final_model, model_path = train_final_model(
        df_features
    )

    print(
        "Model saved:",
        model_path
    )

    print("\nLINEAR REGRESSION PREDICTION")

    linear_result, df_linear, model = train_linear_prediction(
        df_features
    )


    print(linear_result)

    # 9. FORECAST GRAPH
    print("\nGENERATING FORECAST GRAPH")


    plot_forecasting_comparison(
        df_rolling,
        df_linear
    )

    print(
        "\nGraph saved:"
        " output/reports/forecasting_comparison.png"
    )

    print("\nFORECAST MONTH:")

    july_forecast, file = forecast_month(
        df,
        2026,
        7
    )    
    
    print(
        "Forecast saved:",
        file
    )

    print("\nFORECAST REPORT")

    output_file = plot_forecast(
        july_forecast,
        year=2026,
        month=7
    )

    print(f"Forecast graph saved: {output_file}")

if __name__ == "__main__":

    main()
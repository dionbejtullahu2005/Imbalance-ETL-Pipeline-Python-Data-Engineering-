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
from src.report import generate_reports
from src.forecasting import run_forecasting

def main():

    # 1. EXTRACT
    df_j, df_gj = extract_excel()

    print("\nGJ COLUMNS")
    print(df_gj.columns.tolist())

    # 2. TRANSFORM SHEET J
    df = transform(df_j)

    # 3. VALIDATE HOURLY DATA
    print("\nTIME VALIDATION")

    time_result = validate_time_series(df)


    for key, value in time_result.items():
        print(
            key,
            ":",
            value
        )

    # 4. VALIDATE CALCULATED COLUMNS
    print("\nHOURLY METRICS VALIDATION")


    metric_result = validate_metrics(df)


    print(metric_result)

    # 5. CALCULATE GJ FROM SHEET J
    print("\nGJ SUMMARY")

    gj_python = calculate_gj_summary(
        df,
        df_gj
    )

    print(gj_python)

    # 6. COMPARE GJ PYTHON VS EXCEL
    print("\nGJ VALIDATION")

    gj_result = compare_gj(
        gj_python,
        df_gj
    )

    print(gj_result)

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

    print("\nREPORTING")


    hourly_report, peak_report, deviation_distribution = generate_reports(df)


    print("\n24H DEVIATION")
    print(hourly_report)


    print("\nPEAK HOURS 08-17")
    print(peak_report)

    print("\nDEVIATION DISTRIBUTION")
    print(deviation_distribution)

    print("\nFORECASTING")

    forecast_result, df_forecast = run_forecasting(df)


    print(
        forecast_result
    )

if __name__ == "__main__":
    main()
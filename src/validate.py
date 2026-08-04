import pandas as pd

def compare_columns(
        df, 
        python_col,
        excel_col,
        tolerance=0.001
):
    difference = (
        df[python_col]
        -
        df[excel_col]
    ).abs()

    max_differnece = difference.max()
    passed = max_differnece <= tolerance

    return {
        "column": python_col,
        "max_difference": max_differnece,
        "status": 'PASS' if passed else "FAIL"
    }

def validate_metrics(df):
    results = []
    results.append(
        compare_columns(
            df, "new_actual", "excel_new_actual"
        )
    )

    results.append(
        compare_columns(
            df, "old_predicted", 'excel_old_predicted'
        )
    )

    results.append(
        compare_columns(
            df, "delta", "excel_delta"
        )
    )

    return pd.DataFrame(results)

def validate_time_series(df):
    results = {}

    results["total_rows"] = len(df)
    results["expected_rows"] = 720
    results['row_count_pass'] = (
        len(df) == 720
    )

    days = df["date"].nunique()
    results["days"] = days

    hours_per_day = (
        df
        .groupby("date")["hour"]
        .count()
    )

    results["missing_hours"] = (
        hours_per_day[hours_per_day != 24]
        .to_dict()
    )

    results["hours_complete"] = (
        len(results["missing_hours"])==0
    )

    duplicates = (
        df
        .duplicated(
            subset=["date", "hour"]
        )
        .sum()
    )

    results["duplicates"] = duplicates

    results["duplicates_pass"] = (
        duplicates == 0
    )

    return results
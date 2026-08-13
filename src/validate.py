import pandas as pd

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


try:
    TIMEZONE = ZoneInfo("Europe/Pristina")

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("Europe/Tirane")


# COLUMN COMPARISON
def compare_columns(
    df,
    excel_col,
    calculated_col,
    tolerance=0.001
):

    difference = (
        df[excel_col]
        -
        df[calculated_col]
    ).abs()

    max_difference = (
        difference.max()
    )

    passed = (
        max_difference <= tolerance
    )

    return {
        "column": excel_col,
        "max_difference": max_difference,
        "status": "PASS" if passed else "FAIL"
    }


# METRIC VALIDATION
def validate_metrics(df):

    results = []

    # IMBALANCE
    # Excel: Imbalanc
    # Python: imbalance_calculated
    results.append(
        compare_columns(
            df,
            "imbalance",
            "imbalance_calculated"
        )
    )

    # TOTAL EURO
    # Excel: Total Euro
    # Python: total_euro_calculated
    results.append(
        compare_columns(
            df,
            "total_euro",
            "total_euro_calculated"
        )
    )

    # PLAN DEVIATION
    # Excel: Plan dev
    # Python: plan_dev_calculated

    results.append(
        compare_columns(
            df,
            "plan_dev",
            "plan_dev_calculated"
        )
    )

    # MAPE / ABSOLUTE PLAN ERROR
    # Excel: MAPE
    # Python: mape_calculated

    results.append(
        compare_columns(
            df,
            "mape",
            "mape_calculated"
        )
    )

    return pd.DataFrame(
        results
    )


# EXPECTED HOURS PER DATE
def expected_hours_for_date(
    date_value
):

    start = pd.Timestamp(
        date_value
    ).tz_localize(
        TIMEZONE
    )

    end = (
        start
        +
        pd.Timedelta(days=1)
    )

    return len(
        pd.date_range(
            start=start,
            end=end,
            freq="h",
            inclusive="left"
        )
    )

# TIME SERIES VALIDATION
def validate_time_series(df):

    results = {}

    df = df.copy()

    # Përdor datetime nga transform.py
    if "datetime" not in df.columns:

        raise ValueError(
            "datetime column missing. Run transform() first."
        )

    df = df.sort_values(
        "datetime"
    )

    # BASIC INFORMATION
    results["total_rows"] = len(df)
    results["row_count_pass"] = len(df) > 0

    results["start_date"] = str(
        df["date"].min()
    )

    results["end_date"] = str(
        df["date"].max()
    )

    results["days"] = (
        df["date"].nunique()
    )

    # HOURS PER DAY

    hours_per_day = (
        df
        .groupby("date")["hour"]
        .count()
    )

    results["hours_per_day"] = (
        hours_per_day.to_dict()
    )

    invalid_days = {}

    for day, count in hours_per_day.items():

        expected = (
            expected_hours_for_date(
                day
            )
        )

        if count != expected:

            invalid_days[
                str(day)
            ] = {
                "expected": expected,
                "found": int(count)
            }

    results["invalid_days"] = (
        invalid_days
    )

    results["hours_complete"] = (
        len(invalid_days) == 0
    )

    # CHRONOLOGICAL ORDER
    results["chronological_order"] = (
        df["datetime"]
        .is_monotonic_increasing
    )

    # #MISSING HOURS
    full_range = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="h"
    )

    missing_hours = (
        full_range
        .difference(
            df["datetime"]
        )
    )

    results["missing_hours"] = (
        missing_hours
        .strftime(
            "%Y-%m-%d %H:%M"
        )
        .tolist()
    )

    results["missing_hours_pass"] = (
        len(missing_hours) == 0
    )

    # DUPLICATES

    duplicates = (
        df
        .duplicated(
            subset=[
                "date",
                "hour"
            ]
        )
        .sum()
    )

    results["duplicates"] = int(
        duplicates
    )

    results["duplicates_pass"] = (
        duplicates == 0
    )

    return results

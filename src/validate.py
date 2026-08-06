import pandas as pd
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    TIMEZONE = ZoneInfo("Europe/Pristina")

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("Europe/Tirane")

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


    max_difference = (
        difference.max()
    )


    passed = (
        max_difference <= tolerance
    )


    return {
        "column": python_col,
        "max_difference": max_difference,
        "status": "PASS" if passed else "FAIL"
    }

def validate_metrics(df):

    results = []

    results.append(
        compare_columns(
            df,
            "new_actual",
            "excel_new_actual"
        )
    )

    results.append(
        compare_columns(
            df,
            "old_predicted",
            "excel_old_predicted"
        )
    )


    results.append(
        compare_columns(
            df,
            "delta",
            "excel_delta"
        )
    )


    return pd.DataFrame(results)





def expected_hours_for_date(date_value):

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





def validate_time_series(df):

    results = {}

    df = df.copy()



    # ==================================
    # Përdor datetime nga transform.py
    # ==================================

    if "datetime" not in df.columns:

        raise ValueError(
            "datetime column missing. Run transform() first."
        )



    df = df.sort_values(
        "datetime"
    )



    results["total_rows"] = len(df)



    results["start_date"] = str(
        df["date"].min()
    )


    results["end_date"] = str(
        df["date"].max()
    )


    results["days"] = (
        df["date"].nunique()
    )



    # ==================================
    # Kontrolli DST për çdo ditë
    # ==================================

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
            expected_hours_for_date(day)
        )


        if count != expected:

            invalid_days[str(day)] = {

                "expected": expected,

                "found": int(count)

            }



    results["invalid_days"] = (
        invalid_days
    )


    results["hours_complete"] = (
        len(invalid_days) == 0
    )



    # ==================================
    # Renditja kohore
    # ==================================

    results["chronological_order"] = (
        df["datetime"]
        .is_monotonic_increasing
    )



    # ==================================
    # Orët që mungojnë
    # ==================================

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



    # ==================================
    # Dublikate
    # ==================================

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

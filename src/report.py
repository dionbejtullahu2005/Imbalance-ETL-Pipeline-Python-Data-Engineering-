from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    BASE_DIR /
    "output" /
    "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# HOURLY DEVIATION
def create_hourly_pivot(df):

    pivot = (
        df
        .groupby("hour")["percent_delta"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    return pivot

def create_peak_pivot(df):

    peak = df[
        (df["hour"] >= 8)
        &
        (df["hour"] <= 17)
    ]

    pivot = (
        peak
        .groupby("hour")["percent_delta"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    return pivot

# PLOTS
def plot_hourly_deviation(pivot):

    plt.figure(
        figsize=(10,5)
    )

    plt.plot(
        pivot["hour"],
        pivot["percent_delta"],
        marker="o"
    )

    plt.title(
        "Average Deviation by Hour"
    )

    plt.xlabel(
        "Hour"
    )

    plt.ylabel(
        "Deviation (%)"
    )

    plt.grid()

    plt.savefig(
        REPORT_DIR /
        "hourly_deviation.png",
        bbox_inches="tight"
    )

    plt.close()

def plot_actual_vs_prediction(df):

    plt.figure(
        figsize=(14,5)
    )

    plt.plot(
        df["datetime"],
        df["new_actual"],
        label="NEW Actual"
    )

    plt.plot(
        df["datetime"],
        df["old_predicted"],
        label="OLD Predicted"
    )

    plt.title(
        "Actual vs Predicted"
    )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "MWh"
    )

    plt.legend()

    plt.grid()

    plt.xticks(
        rotation=45
    )

    plt.savefig(
        REPORT_DIR /
        "actual_vs_prediction.png",
        bbox_inches="tight"
    )

    plt.close()

def plot_daily_deviation(df):

    daily = (
        df
        .groupby("date")
        ["percent_delta"]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(14,5)
    )

    plt.plot(
        daily["date"],
        daily["percent_delta"],
        marker="o"
    )

    plt.title(
        "Daily Average Deviation"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Deviation (%)"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid()

    plt.savefig(
        REPORT_DIR /
        "daily_deviation.png",
        bbox_inches="tight"
    )

    plt.close()

# DISTRIBUTION
def create_deviation_distribution(df):

    bins = [
        0,
        5,
        10,
        15,
        20,
        100
    ]

    labels = [
        "0-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "+20%"
    ]

    temp = df.copy()

    temp["deviation_group"] = pd.cut(
        temp["percent_delta"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        temp
        .groupby(
            "deviation_group",
            observed=False
        )
        .size()
        .reset_index(
            name="hours"
        )
    )

    return result

# MAIN REPORT GENERATION
def generate_reports(df):


    print(
        "Saving reports to:",
        REPORT_DIR.resolve()
    )

    hourly = create_hourly_pivot(
        df
    )

    peak = create_peak_pivot(
        df
    )

    deviation_distribution = (
        create_deviation_distribution(df)
    )

    plot_hourly_deviation(
        hourly
    )

    plot_actual_vs_prediction(
        df
    )

    plot_daily_deviation(
        df
    )

    return (
        hourly,
        peak,
        deviation_distribution
    )

# FORECASTING COMPARISON
def plot_forecasting_comparison(
        rolling_df,
        linear_df
):

    plt.figure(
        figsize=(16,6)
    )

    plt.plot(
        rolling_df["datetime"],
        rolling_df["new_actual"],
        label="NEW Actual"
    )

    plt.plot(
        rolling_df["datetime"],
        rolling_df["old_predicted"],
        label="OLD Prediction"
    )

    plt.plot(
        rolling_df["datetime"],
        rolling_df["rolling_prediction"],
        label="Rolling Average"
    )

    plt.plot(
        linear_df["datetime"],
        linear_df["linear_prediction"],
        label="Linear Regression"
    )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "MWh"
    )

    plt.title(
        "Forecast Comparison"
    )

    plt.legend()
    plt.grid()

    plt.xticks(
        rotation=45
    )

    plt.savefig(
        REPORT_DIR /
        "forecasting_comparison.png",
        bbox_inches="tight"
    )

    plt.close()

# JULY FORECAST
 # FORECAST REPORT
def plot_forecast(
        forecast_df,
        year,
        month
):

    import calendar


    month_name = calendar.month_name[month].lower()


    filename = (
        f"{month_name}_{year}_forecast.png"
    )


    plt.figure(
        figsize=(16,6)
    )


    plt.plot(
        forecast_df["datetime"],
        forecast_df["predicted_MWh"],
        label=f"{month_name.capitalize()} {year} Forecast"
    )


    plt.title(
        f"Electricity Imbalance Forecast - {month_name.capitalize()} {year}"
    )


    plt.xlabel(
        "Datetime"
    )


    plt.ylabel(
        "MWh"
    )


    plt.legend()

    plt.grid()


    plt.xticks(
        rotation=45
    )


    output_file = (
        REPORT_DIR /
        filename
    )


    plt.savefig(
        output_file,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Forecast graph saved:",
        output_file
    )


    return output_file
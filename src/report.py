from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "output" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def create_hourly_pivot(df):
    """
    Devijimi mesatar sipas ores 1-24
    """

    pivot = (
        df
        .groupby("hour")
        ["percent_delta"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    return pivot

def create_peak_pivot(df):
    """
    Devijimi per oren peak 8 - 17
    """

    peak_df=df[
        (df["hour"] >= 8)
        &
        (df["hour"] <= 17)
    ]

    pivot = (
        peak_df
        .groupby("hour")
        ["percent_delta"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    return pivot

def plot_hourly_deviation(pivot):
    plt.figure(figsize=(10,5))

    plt.plot(
        pivot["hour"],
        pivot["percent_delta"],
        marker="o"
    )

    plt.xlabel(
        "Ora"
    )

    plt.ylabel(
        "Devijimi mesatar sipas ores"
    )

    plt.grid()

    plt.savefig(
        REPORT_DIR / "hourly_deviation.png",
        bbox_inches="tight"
    )

    plt.close()

def plot_actual_vs_prediction(df):
    plt.figure(
        figsize=(12,5)
    )

    plt.plot(
        df.index,
        df["new_actual"],
        label='Actual'
    )

    plt.plot(
        df.index,
        df["old_predicted"],
        label="Predicted"
    )

    plt.xlabel("Data + Ora per 30 dite")
    plt.ylabel("MWh")
    plt.title("Actual vs Predicted")

    plt.legend()
    plt.grid()

    plt.savefig(
        REPORT_DIR / "actual_vs_prediction.png",
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
        figsize=(12,5)
    )

    plt.plot(
        daily["date"],
        daily["percent_delta"],
        marker="o"
    )

    plt.xlabel("Data")
    plt.ylabel("Devijimi mesatar ne %")
    plt.title("Devijimi mesatar sipas dites")

    plt.xticks(rotation=45)
    plt.grid()

    plt.savefig(
        REPORT_DIR / "daily_deviation.png",
        bbox_inches="tight"
    )

    plt.close()

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

    df = df.copy()

    df["deviation_group"] = pd.cut(
        df["percent_delta"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    distribution = (
        df.groupby("deviation_group", observed=False)
        .size()
        .reset_index(name="hours")
    )

    return distribution

def generate_reports(df):
    print("Saving reports to:", REPORT_DIR.resolve())

    hourly = create_hourly_pivot(df)
    peak = create_peak_pivot(df)
    deviation_distribution = create_deviation_distribution(df)

    plot_hourly_deviation(hourly)
    plot_actual_vs_prediction(df)
    plot_daily_deviation(df)

    return hourly, peak, deviation_distribution
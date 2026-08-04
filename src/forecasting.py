import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "output" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def rolling_average_prediction(df, window=3):
    """
    Parashikim me rolling average
    bazuar në 3 orët paraprake
    """

    df = df.copy()

    df["rolling_prediction"] = (
        df["new_actual"]
        .rolling(window=window)
        .mean()
    )

    return df



def calculate_forecast_metrics(df):
    """
    Krahasimi i OLD Predicted
    me Rolling Average Prediction
    """

    df = df.dropna(
        subset=[
            "rolling_prediction"
        ]
    )


    old_mae = (
        (df["new_actual"] - df["old_predicted"])
        .abs()
        .mean()
    )


    rolling_mae = (
        (df["new_actual"] - df["rolling_prediction"])
        .abs()
        .mean()
    )


    old_mape = (
        (
            abs(
                (df["new_actual"] -
                 df["old_predicted"])
                /
                df["new_actual"]
            )
        )
        .mean()
        *
        100
    )


    rolling_mape = (
        (
            abs(
                (df["new_actual"] -
                 df["rolling_prediction"])
                /
                df["new_actual"]
            )
        )
        .mean()
        *
        100
    )


    return pd.DataFrame(
        {
            "Model": [
                "OLD Predicted",
                "Rolling Average"
            ],

            "MAE": [
                old_mae,
                rolling_mae
            ],

            "MAPE (%)": [
                old_mape,
                rolling_mape
            ]
        }
    )



def plot_forecasting_comparison(df):

    plt.figure(
        figsize=(14,6)
    )


    plt.plot(
        df.index,
        df["new_actual"],
        label="NEW Actual"
    )


    plt.plot(
        df.index,
        df["old_predicted"],
        label="OLD Predicted"
    )


    plt.plot(
        df.index,
        df["rolling_prediction"],
        label="Rolling Average Prediction"
    )


    plt.xlabel(
        "Ora"
    )

    plt.ylabel(
        "MWh"
    )

    plt.title(
        "Forecast Comparison"
    )


    plt.legend()

    plt.grid()


    plt.savefig(
        REPORT_DIR /
        "forecasting_comparison.png",
        bbox_inches="tight"
    )


    plt.close()



def run_forecasting(df):

    df_forecast = rolling_average_prediction(
        df
    )


    metrics = calculate_forecast_metrics(
        df_forecast
    )


    plot_forecasting_comparison(
        df_forecast
    )


    return metrics, df_forecast
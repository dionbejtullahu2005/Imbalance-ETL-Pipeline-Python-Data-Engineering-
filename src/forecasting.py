import pandas as pd

def rolling_average_prediction(df, window=3):

    df = df.copy()

    df["rolling_prediction"] = (
        df["imbalance"]
        .shift(1)
        .rolling(window)
        .mean()
    )

    return df


def calculate_rolling_metrics(df):

    df = df.dropna(
        subset=[
            "rolling_prediction"
        ]
    )

    mae = (
        abs(
            df["imbalance"]
            -
            df["rolling_prediction"]
        )
        .mean()
    )


    mape = (
        abs(
            (
                df["imbalance"]
                -
                df["rolling_prediction"]
            )
            /
            df["imbalance"]
        )
        .mean()
        *
        100
    )


    return pd.DataFrame(
        {
            "Model":
            [
                "Rolling Average"
            ],

            "MAE":
            [
                mae
            ],

            "MAPE (%)":
            [
                mape
            ]
        }
    )
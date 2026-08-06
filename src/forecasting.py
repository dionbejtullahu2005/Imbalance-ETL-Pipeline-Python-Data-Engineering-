import pandas as pd


def rolling_average_prediction(df, window=3):

    df = df.copy()

    df["rolling_prediction"] = (
        df["new_actual"]
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
            df["new_actual"]
            -
            df["rolling_prediction"]
        )
        .mean()
    )


    mape = (
        abs(
            (
                df["new_actual"]
                -
                df["rolling_prediction"]
            )
            /
            df["new_actual"]
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
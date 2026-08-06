import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error
)

FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag3",
    "lag24",
    "roll_mean_3",
    "roll_mean_24",
    "roll_std_3",
    "hour_sin",
    "hour_cos",
    "dayofweek"
]

def evaluate_timeseries_model(df, splits=5):

    df = df.copy()

    # largojmë rreshtat pa features
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["new_actual"]
    )

    X = df[FEATURE_COLUMNS]

    y = df["new_actual"]

    tscv = TimeSeriesSplit(
        n_splits=splits
    )

    results = []

    fold = 1

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        mape = (
            mean_absolute_percentage_error(
                y_test,
                prediction
            )
            *
            100
        )

        results.append(
            {
                "Fold": fold,
                "Train Size": len(train_index),
                "Test Size": len(test_index),
                "MAE": mae,
                "MAPE (%)": mape
            }
        )

        fold += 1

    result_df = pd.DataFrame(results)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Average MAE",
                "Average MAPE (%)"
            ],

            "Value": [
                result_df["MAE"].mean(),
                result_df["MAPE (%)"].mean()
            ]
        }
    )

    return result_df, summary
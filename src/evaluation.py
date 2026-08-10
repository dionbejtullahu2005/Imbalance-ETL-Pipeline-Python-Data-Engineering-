import pandas as pd
import numpy as np

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

def evaluate_timeseries_model(
        df,
        splits=5
):

    df = df.copy()


    df = df.dropna(
        subset=FEATURE_COLUMNS + ["new_actual"]
    )


    X = df[FEATURE_COLUMNS]

    y = df["new_actual"]


    tscv = TimeSeriesSplit(
        n_splits=splits
    )


    results = []

    errors = []


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


        # ==========================
        # ERROR DISTRIBUTION
        # ==========================

        fold_errors = (
            y_test.reset_index(drop=True)
            -
            pd.Series(prediction)
        )


        errors.extend(
            fold_errors.tolist()
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



    result_df = pd.DataFrame(
        results
    )


    error_array = np.array(
        errors
    )


    # ==========================
    # 90% CONFIDENCE INTERVAL
    # ==========================

    lower_error = np.percentile(
        error_array,
        5
    )


    upper_error = np.percentile(
        error_array,
        95
    )


    summary = pd.DataFrame(
        {
            "Metric":[

                "Average MAE",

                "Average MAPE (%)",

                "90% Error Lower Bound",

                "90% Error Upper Bound"

            ],


            "Value":[

                result_df["MAE"].mean(),

                result_df["MAPE (%)"].mean(),

                lower_error,

                upper_error

            ]
        }
    )


    return (
        result_df,
        summary,
        error_array
    )

def calculate_prediction_interval(
        model,
        X_test,
        y_test,
        confidence=0.90
):

    predictions = model.predict(
        X_test
    )


    errors = (
        y_test.values
        -
        predictions
    )


    lower_error = np.quantile(
        errors,
        (1-confidence)/2
    )


    upper_error = np.quantile(
        errors,
        1-(1-confidence)/2
    )


    return (
        lower_error,
        upper_error
    )
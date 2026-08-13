import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error
)

from src.features import FEATURE_COLUMNS, create_features, validate_feature_frame


def recursive_predict(model, history, horizon):
    """Forecast a complete horizon without revealing any horizon targets."""
    recursive_history = (
        history[["datetime", "imbalance", "temperature"]]
        .dropna(subset=["datetime", "imbalance"])
        .sort_values("datetime")
        .reset_index(drop=True)
        .copy()
    )
    if recursive_history["datetime"].duplicated().any():
        raise ValueError("Recursive history contains duplicate timestamps")

    predictions = []

    horizon = (
        horizon
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    last_recursive_time = (
        recursive_history["datetime"].max()
        if not recursive_history.empty
        else None
    )

    for row in horizon.itertuples(index=False):

        current_time = pd.Timestamp(
            row.datetime
        )

        if (
            last_recursive_time is not None
            and
            current_time
            <=
            last_recursive_time
        ):
            raise ValueError(
                "Forecast horizon overlaps "
                "recursive history"
            )

        current_temperature = getattr(
            row,
            "temperature",
            np.nan,
        )

        # Vetëm 168 orët e fundit nevojiten për:
        # lag168 dhe roll_mean_168.
        feature_history = (
            recursive_history[
                [
                    "datetime",
                    "imbalance",
                    "temperature",
                ]
            ]
            .tail(168)
            .copy()
        )

        current_row = pd.DataFrame(
            {
                "datetime": [
                    current_time
                ],
                "imbalance": [
                    np.nan
                ],
                "temperature": [
                    current_temperature
                ],
            }
        )

        candidate = pd.concat(
            [
                feature_history,
                current_row,
            ],
            ignore_index=True,
        )

        current_features = (
            create_features(candidate)
            .tail(1)
        )

        X = validate_feature_frame(
            current_features
        )

        prediction = float(
            model.predict(X)[0]
        )

        predictions.append(
            prediction
        )

        recursive_history.loc[
            len(recursive_history)
        ] = {
            "datetime":
                current_time,

            "imbalance":
                prediction,

            "temperature":
                current_temperature,
        }

        last_recursive_time = (
            current_time
        )

    return np.asarray(
        predictions,
        dtype=float,
    )

def recursive_timeseries_residuals(model, df, splits=5):
    """Generate leakage-safe expanding-window residuals."""
    data = df.sort_values("datetime").reset_index(drop=True).copy()
    required = FEATURE_COLUMNS + ["datetime", "imbalance", "temperature"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing recursive evaluation columns: " + ", ".join(missing))
    usable = data.dropna(subset=FEATURE_COLUMNS + ["imbalance"]).copy()
    splitter = TimeSeriesSplit(n_splits=splits)
    records = []
    for fold, (train_index, test_index) in enumerate(splitter.split(usable), start=1):
        train = usable.iloc[train_index].copy()
        test = usable.iloc[test_index].copy()
        fold_model = clone(model)
        fold_model.fit(train[FEATURE_COLUMNS], train["imbalance"])
        history = data[data["datetime"] < test["datetime"].min()]
        predictions = recursive_predict(fold_model, history, test)
        for position, (_, row) in enumerate(test.iterrows()):
            actual = float(row["imbalance"])
            prediction = float(predictions[position])
            records.append({
                "fold": fold,
                "datetime": row["datetime"],
                "hour": int(row["hour"]),
                "actual": actual,
                "prediction": prediction,
                "signed_residual": actual - prediction,
                "absolute_residual": abs(actual - prediction),
            })
    return pd.DataFrame(records)

def evaluate_timeseries_model(
        df,
        splits=5
):

    df = df.copy()


    df = df.dropna(
        subset=FEATURE_COLUMNS + ["imbalance"]
    )


    X = df[FEATURE_COLUMNS]

    y = df["imbalance"]


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


        # ERROR DISTRIBUTION
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


    # 90% CONFIDENCE INTERVAL
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

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error
)

import joblib

from pathlib import Path


# ==========================================================
# MODEL DIRECTORY
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    /
    "output"
    /
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# FEATURES
# ==========================================================

FEATURE_COLUMNS = [

    "lag1",
    "lag2",
    "lag3",

    "lag24",
    "lag48",
    "lag168",

    "roll_mean_3",
    "roll_mean_24",
    "roll_mean_168",

    "roll_std_3",
    "roll_std_24",

    "hour_sin",
    "hour_cos",

    "dayofweek",
    "is_weekend",

    "temperature"
]


# ==========================================================
# MODELS
# ==========================================================

MODELS = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )

}


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model(
        model,
        X,
        y,
        price,
        splits=5
):

    tscv = TimeSeriesSplit(
        n_splits=splits
    )

    mae_scores = []
    mape_scores = []

    cost_scores = []

    total_cost_scores = []


    for train_index, test_index in tscv.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        price_test = price.iloc[test_index]


        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            X_test
        )


        # --------------------------------------------------
        # MAE
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            prediction
        )


        # --------------------------------------------------
        # MAPE
        # --------------------------------------------------

        mape = (
            mean_absolute_percentage_error(
                y_test,
                prediction
            )
            *
            100
        )


        # --------------------------------------------------
        # ERROR IN MWh
        # --------------------------------------------------

        absolute_error = (
            y_test.reset_index(drop=True)
            -
            pd.Series(
                prediction
            )
        ).abs()


        # --------------------------------------------------
        # COST OF ERROR
        #
        # € = absolute error MWh
        #      × imbalance price €/MWh
        # --------------------------------------------------

        error_cost = (
            absolute_error
            *
            price_test.reset_index(drop=True).abs()
        )


        # --------------------------------------------------
        # AVERAGE COST PER HOUR
        # --------------------------------------------------

        average_error_cost = (
            error_cost.mean()
        )


        # --------------------------------------------------
        # TOTAL COST
        # --------------------------------------------------

        total_error_cost = (
            error_cost.sum()
        )


        mae_scores.append(
            mae
        )

        mape_scores.append(
            mape
        )

        cost_scores.append(
            average_error_cost
        )

        total_cost_scores.append(
            total_error_cost
        )


    return (
        sum(mae_scores) / len(mae_scores),
        sum(mape_scores) / len(mape_scores),
        sum(cost_scores) / len(cost_scores),
        sum(total_cost_scores)
    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def compare_models(
        df_features
):

    df = df_features.copy()


    # ------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        +
        [
            "new_actual",
            "price"
        ]
    )


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing columns for model comparison: "
            +
            ", ".join(missing_columns)
        )


    # ------------------------------------------------------
    # REMOVE INVALID ROWS
    # ------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )


    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "new_actual"
    ]

    price = df[
        "price"
    ]


    results = []


    # ------------------------------------------------------
    # EVALUATE MODELS
    # ------------------------------------------------------

    for name, model in MODELS.items():

        (
            mae,
            mape,
            average_cost,
            total_cost
        ) = evaluate_model(
            model,
            X,
            y,
            price
        )


        results.append(
            {
                "Model": name,

                "MAE": mae,

                "MAPE (%)": mape,

                "Avg Error Cost (€)": average_cost,

                "Total Error Cost (€)": total_cost
            }
        )


    result_df = pd.DataFrame(
        results
    )


    # ------------------------------------------------------
    # SORT BY BUSINESS COST
    # ------------------------------------------------------

    result_df = result_df.sort_values(
        "Total Error Cost (€)"
    ).reset_index(
        drop=True
    )


    return result_df


# ==========================================================
# TRAIN FINAL MODEL
# ==========================================================

def train_final_model(
        df_features
):

    df = df_features.copy()


    df = df.dropna(
        subset=
        FEATURE_COLUMNS
        +
        [
            "new_actual"
        ]
    )


    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "new_actual"
    ]


    model = LinearRegression()


    model.fit(
        X,
        y
    )


    model_path = (
        MODEL_DIR
        /
        "linear_regression_final.pkl"
    )


    joblib.dump(
        model,
        model_path
    )


    return (
        model,
        model_path
    )

def train_explainable_models(df_features):

    df = df_features.copy()


    df = df.dropna(
        subset=
        FEATURE_COLUMNS
        +
        [
            "new_actual"
        ]
    )


    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "new_actual"
    ]


    models = {

        "random_forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),


        "gradient_boosting":
            GradientBoostingRegressor(
                random_state=42
            )

    }


    trained_models = {}


    for name, model in models.items():


        model.fit(
            X,
            y
        )


        model_path = (
            MODEL_DIR
            /
            f"{name}.pkl"
        )


        joblib.dump(
            model,
            model_path
        )


        trained_models[name] = model


        print(
            f"{name} saved:",
            model_path
        )


    return trained_models


# ==========================================================
# LINEAR REGRESSION PREDICTION
# ==========================================================

def train_linear_prediction(
        df_features
):

    df = df_features.copy()


    model_df = df.dropna(
        subset=
        FEATURE_COLUMNS
        +
        [
            "new_actual"
        ]
    )


    X = model_df[
        FEATURE_COLUMNS
    ]

    y = model_df[
        "new_actual"
    ]


    # ------------------------------------------------------
    # 80% TRAIN / 20% TEST
    # ------------------------------------------------------

    split = int(
        len(model_df) * 0.8
    )


    X_train = X.iloc[:split]
    X_test = X.iloc[split:]


    y_train = y.iloc[:split]
    y_test = y.iloc[split:]


    model = LinearRegression()


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    df_test = model_df.iloc[
        split:
    ].copy()


    df_test[
        "linear_prediction"
    ] = prediction


    # ------------------------------------------------------
    # MAE
    # ------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        prediction
    )


    # ------------------------------------------------------
    # MAPE
    # ------------------------------------------------------

    mape = (
        mean_absolute_percentage_error(
            y_test,
            prediction
        )
        *
        100
    )


    # ------------------------------------------------------
    # ERROR COST
    # ------------------------------------------------------

    absolute_error = (
        y_test.reset_index(drop=True)
        -
        pd.Series(prediction)
    ).abs()


    price_test = (
        df_test["price"]
        .reset_index(drop=True)
    )


    error_cost = (
        absolute_error
        *
        price_test.abs()
    )


    average_error_cost = (
        error_cost.mean()
    )


    total_error_cost = (
        error_cost.sum()
    )


    result = pd.DataFrame(
        {
            "Model": [
                "Linear Regression"
            ],

            "MAE": [
                mae
            ],

            "MAPE (%)": [
                mape
            ],

            "Avg Error Cost (€)": [
                average_error_cost
            ],

            "Total Error Cost (€)": [
                total_error_cost
            ]
        }
    )


    joblib.dump(
        model,
        MODEL_DIR
        /
        "linear_regression.pkl"
    )


    return (
        result,
        df_test,
        model
    )


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_linear_model():

    model = joblib.load(
        MODEL_DIR
        /
        "linear_regression.pkl"
    )

    return model

import numpy as np


def calculate_prediction_interval(
        model,
        X,
        y,
        confidence=0.90
):

    prediction = model.predict(
        X
    )


    errors = (
        y.values
        -
        prediction
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
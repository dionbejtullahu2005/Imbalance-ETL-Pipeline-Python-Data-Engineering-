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

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR /
    "output" /
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# FEATURES
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
    "is_weekend"
]

# MODELS
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

# TIME SERIES VALIDATION
def evaluate_model(
        model,
        X,
        y,
        splits=5
):

    tscv = TimeSeriesSplit(
        n_splits=splits
    )

    mae_scores = []
    mape_scores = []

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

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


        mae_scores.append(
            mae
        )

        mape_scores.append(
            mape
        )

    return (
        sum(mae_scores) /
        len(mae_scores),

        sum(mape_scores) /
        len(mape_scores)
    )

# MODEL COMPARISON
def compare_models(df_features):

    df = (
        df_features
        .dropna(
            subset=
            FEATURE_COLUMNS + ["new_actual"]
        )
        .reset_index(drop=True)
    )

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "new_actual"
    ]

    results = []

    for name, model in MODELS.items():
        mae, mape = evaluate_model(
            model,
            X,
            y
        )

        results.append(
            {
                "Model":
                    name,

                "MAE":
                    mae,

                "MAPE (%)":
                    mape
            }

        )

    return pd.DataFrame(
        results
    )

# TRAIN FINAL MODEL
def train_final_model(df_features):

    df = (
        df_features
        .dropna(
            subset=
            FEATURE_COLUMNS + ["new_actual"]
        )
        .reset_index(drop=True)
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

        MODEL_DIR /
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

# LINEAR REGRESSION TEST PREDICTION
def train_linear_prediction(df_features):

    df = df_features.copy()

    # heqim vetëm rreshtat pa feature
    model_df = df.dropna(
        subset=FEATURE_COLUMNS + ["new_actual"]
    )

    X = model_df[FEATURE_COLUMNS]
    y = model_df["new_actual"]

    model = LinearRegression()

    # TRAIN ME GJITHË HISTORIKUN
    model.fit(
        X,
        y
    )

    # PREDIKTIM PËR GJITHË DATASET
    predictions = model.predict(
        X
    )

    df_linear = model_df.copy()

    df_linear["linear_prediction"] = predictions

    mae = mean_absolute_error(
        y,
        predictions
    )

    mape = (
        mean_absolute_percentage_error(
            y,
            predictions
        )
        * 100
    )

    result = pd.DataFrame(
        {
            "Model":
            [
                "Linear Regression"
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

    # ruaj modelin final
    joblib.dump(
        model,
        MODEL_DIR /
        "linear_regression.pkl"
    )

    return (
        result,
        df_linear,
        model
    )

# LOAD MODEL
def load_linear_model():
    model = joblib.load(

        MODEL_DIR /
        "linear_regression.pkl"

    )

    return model
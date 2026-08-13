import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# PATHS

BASE_DIR = Path(__file__).resolve().parent

FORECAST_PATH = (
    BASE_DIR
    / "output"
    / "forecast"
    / "2026_07_forecast.csv"
)

ACTUAL_PATH = (
    BASE_DIR
    / "data"
    / "Imbalanc July 2026 (1).xlsx"
)

OUTPUT_PATH = (
    BASE_DIR
    / "output"
    / "forecast"
    / "2026_07_model_backtest.csv"
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
    "is_weekend",

    "temperature"
]


# LOAD FORECAST
forecast = pd.read_csv(
    FORECAST_PATH
)

forecast["datetime"] = pd.to_datetime(
    forecast["datetime"],
    errors="coerce"
)

if forecast["datetime"].duplicated().any():
    raise ValueError("Forecast contains duplicate timestamps")


# LOAD REAL JULY DATA
actual = pd.read_excel(
    ACTUAL_PATH,
    sheet_name="imbalanc h"
)

actual.columns = (
    actual.columns
    .astype(str)
    .str.strip()
)

actual = actual[
    actual["Supplier"].notna()
    &
    actual["Date"].notna()
    &
    actual["Hour"].notna()
].copy()

actual = actual[
    actual["Supplier"]
    .astype(str)
    .str.strip()
    .eq("EnerCo")
].copy()


# DATETIME
actual["Date"] = pd.to_datetime(
    actual["Date"],
    dayfirst=True,
    errors="coerce"
)

actual["Hour"] = pd.to_numeric(
    actual["Hour"],
    errors="coerce"
)

actual["datetime"] = (
    actual["Date"]
    +
    pd.to_timedelta(
        actual["Hour"] - 1,
        unit="h"
    )
)


# REAL VALUES

actual["actual_imbalance_MWh"] = pd.to_numeric(
    actual["Imbalanc"],
    errors="coerce"
)

actual["actual_price_EUR_MWh"] = pd.to_numeric(
    actual["Price"],
    errors="coerce"
)

if actual["datetime"].duplicated().any():
    raise ValueError("Actual data contains duplicate timestamps")


# MERGE FORECAST + ACTUAL
backtest = forecast.merge(
    actual[
        [
            "datetime",
            "actual_imbalance_MWh",
            "actual_price_EUR_MWh"
        ]
    ],
    on="datetime",
    how="inner",
    validate="one_to_one"
)

expected = pd.date_range("2026-07-01 00:00:00", "2026-07-31 23:00:00", freq="h")
actual_timestamps = pd.DatetimeIndex(actual["datetime"].dropna().unique())
missing_actual = expected.difference(actual_timestamps)
month_status = "COMPLETE_MONTH" if len(missing_actual) == 0 else "PARTIAL_MONTH"

# CHECK MODEL PREDICTION COLUMNS
required_prediction_columns = [
    "linear_prediction_MWh",
    "random_forest_prediction_MWh",
    "gradient_boosting_prediction_MWh",
    "seasonal_prediction_MWh"
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in backtest.columns
]

if missing_prediction_columns:

    raise ValueError(
        "Forecast CSV is missing prediction columns: "
        + ", ".join(missing_prediction_columns)
    )


# HYBRID
if "predicted_imbalance_MWh" in backtest.columns:

    backtest[
        "hybrid_prediction_MWh"
    ] = backtest[
        "predicted_imbalance_MWh"
    ]

else:

    backtest[
        "hybrid_prediction_MWh"
    ] = backtest[
        "predicted_MWh"
    ]


# SEASONAL

if "seasonal_prediction_MWh" not in backtest.columns:

    backtest[
        "seasonal_prediction_MWh"
    ] = np.nan


# HELPER FUNCTION

def evaluate_prediction(
    df,
    prediction_column,
    model_name
):

    valid = df[
        [
            "actual_imbalance_MWh",
            "actual_price_EUR_MWh",
            prediction_column
        ]
    ].dropna()

    if len(valid) == 0:

        return None

    actual_values = (
        valid[
            "actual_imbalance_MWh"
        ]
    )

    predicted_values = (
        valid[
            prediction_column
        ]
    )

    # MAE
    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    # Bias
    bias = (
        predicted_values
        -
        actual_values
    ).mean()

    # Correlation
    correlation = (
        actual_values
        .corr(
            predicted_values
        )
    )

    # Sign accuracy
    sign_accuracy = (
        np.sign(
            actual_values
        )
        ==
        np.sign(
            predicted_values
        )
    ).mean() * 100

    # Absolute error
    absolute_error = (
        actual_values
        -
        predicted_values
    ).abs()

    # Real financial cost
    error_cost = (
        absolute_error
        *
        valid[
            "actual_price_EUR_MWh"
        ].abs()
    )

    total_cost = (
        error_cost.sum()
    )

    average_cost = (
        error_cost.mean()
    )

    return {
        "Model": model_name,

        "Rows": len(valid),

        "MAE (MWh)": mae,

        "RMSE (MWh)": rmse,

        "Bias (MWh)": bias,

        "Correlation": correlation,

        "Sign Accuracy (%)":
            sign_accuracy,

        "Total Error Cost (€)":
            total_cost,

        "Average Error Cost €/h":
            average_cost
    }


# EVALUATE ALL MODELS
results = []

model_columns = [

    (
        "linear_prediction_MWh",
        "Linear Regression"
    ),

    (
        "random_forest_prediction_MWh",
        "Random Forest"
    ),

    (
        "gradient_boosting_prediction_MWh",
        "Gradient Boosting"
    ),

    (
        "seasonal_prediction_MWh",
        "Seasonal"
    ),

    (
        "hybrid_prediction_MWh",
        "Final Model"
    )
]


for column, name in model_columns:

    result = evaluate_prediction(
        backtest,
        column,
        name
    )

    if result is not None:

        results.append(
            result
        )


result_df = pd.DataFrame(
    results
)


# SORT BY BUSINESS COST
if len(result_df) > 0:

    result_df = (
        result_df
        .sort_values(
            "Total Error Cost (€)"
        )
        .reset_index(
            drop=True
        )
    )


# HYBRID INTERVAL COVERAGE

if (
    "lower_bound_MWh"
    in backtest.columns
    and
    "upper_bound_MWh"
    in backtest.columns
):

    backtest[
        "within_90_interval"
    ] = (

        (
            backtest[
                "actual_imbalance_MWh"
            ]
            >=
            backtest[
                "lower_bound_MWh"
            ]
        )

        &

        (
            backtest[
                "actual_imbalance_MWh"
            ]
            <=
            backtest[
                "upper_bound_MWh"
            ]
        )
    )

    coverage = (
        backtest[
            "within_90_interval"
        ]
        .mean()
        *
        100
    )

else:

    coverage = np.nan

# PRINT RESULTS
print("\n")
print("=" * 90)
print("JULY 2026 MODEL BACKTEST")
print("=" * 90)
print(f"Status: {month_status}")
print(f"Forecast rows: {len(forecast)}")
print(f"Actual rows: {len(actual)}")
print(f"Matched rows: {len(backtest)}")
print(f"Missing actual rows: {len(missing_actual)}")
if len(missing_actual):
    print(f"Missing actual range: {missing_actual.min()} -> {missing_actual.max()}")

print(
    result_df.to_string(
        index=False
    )
)

print("\nFINAL MODEL INTERVAL")

print(
    f"90% Interval Coverage: "
    f"{coverage:.2f}%"
)


# BEST MODEL BY MAE
if len(result_df) > 0:

    best_mae = (
        result_df
        .sort_values(
            "MAE (MWh)"
        )
        .iloc[0]
    )

    best_cost = (
        result_df
        .sort_values(
            "Total Error Cost (€)"
        )
        .iloc[0]
    )

    print("\nBEST BY MAE")

    print(
        best_mae[
            "Model"
        ],
        "->",
        round(
            best_mae[
                "MAE (MWh)"
            ],
            6
        ),
        "MWh"
    )

    print("\nBEST BY COST")

    print(
        best_cost[
            "Model"
        ],
        "->",
        round(
            best_cost[
                "Total Error Cost (€)"
            ],
            2
        ),
        "EUR"
    )


# WORST HYBRID HOURS
backtest[
    "hybrid_absolute_error_MWh"
] = (

    backtest[
        "actual_imbalance_MWh"
    ]

    -

    backtest[
        "hybrid_prediction_MWh"
    ]

).abs()


print("\nTOP 10 WORST FINAL-MODEL HOURS")

print(

    backtest
    .sort_values(
        "hybrid_absolute_error_MWh",
        ascending=False
    )
    [
        [
            "datetime",

            "actual_imbalance_MWh",

            "hybrid_prediction_MWh",

            "hybrid_absolute_error_MWh",

            "actual_price_EUR_MWh"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )

)


# SAVE
backtest.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nBacktest saved:",
    OUTPUT_PATH
)

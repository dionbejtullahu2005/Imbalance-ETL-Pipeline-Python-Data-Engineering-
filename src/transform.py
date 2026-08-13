import pandas as pd
import numpy as np


COLUMN_MAPPING = {

    "Supplier":
        "supplier",

    "Date":
        "date",

    "Hour":
        "hour",

    "Consumption (kWh)":
        "consumption_kwh",

    "Production":
        "production_kwh",

    "Reactive (kVarh)":
        "reactive_kvarh",

    "Consumption (MWh)":
        "consumption_mwh",

    "Production2":
        "production_mwh",

    "Plan MWh":
        "plan_mwh",

    "Plan MWh3":
        "plan_mwh3",

    "Plan MWh4":
        "plan_mwh4",

    "Plan MWh5":
        "plan_mwh5",

    "Imbalanc":
        "imbalance",

    "Price":
        "price",

    "Total Euro":
        "total_euro",

    "Plan dev":
        "plan_dev",

    "MAPE":
        "mape",

    "MAPE percetange":
        "mape_percentage",

    "0.01%":
        "percent_001",

    "0.00%":
        "percent_000",

    "Positive":
        "positive",

    "Negative":
        "negative",

    "ALPEX":
        "alpex",

    "ALPEX / Imbalanc":
        "alpex_imbalance",

    "Alpex price imbalances":
        "alpex_imbalance_price",

    "Uncover sell":
        "uncover_sell",

    "Uncover Buy":
        "uncover_buy",
}

VALID_IMBALANCE_FORMULAS = {
    "base",
    "with_uncover",
}


def transform(df):

    df = df.copy()

    # ==========================================================
    # RENAME
    # ==========================================================

    df = df.rename(
        columns=COLUMN_MAPPING
    )

    # ==========================================================
    # DATETIME
    # ==========================================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="coerce"
    )

    # Excel përdor 1-24.
    # pandas datetime përdor 0-23.
    df["datetime"] = (
        df["date"]
        + pd.to_timedelta(
            df["hour"] - 1,
            unit="h"
        )
    )

    # ==========================================================
    # NUMERIC COLUMNS
    # ==========================================================

    numeric_columns = [

        "consumption_kwh",
        "production_kwh",
        "reactive_kvarh",

        "consumption_mwh",
        "production_mwh",

        "plan_mwh",
        "plan_mwh3",
        "plan_mwh4",
        "plan_mwh5",

        "imbalance",
        "price",
        "total_euro",

        "plan_dev",
        "mape",
        "mape_percentage",

        "percent_001",
        "percent_000",

        "positive",
        "negative",

        "alpex",
        "alpex_imbalance",
        "alpex_imbalance_price",

        "uncover_sell",
        "uncover_buy",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # ==========================================================
    # SIGUROHEMI VLERAT E MUNGËSUARA
    # ==========================================================

    df["uncover_sell"] = (
        df["uncover_sell"]
        .fillna(0)
    )

    df["uncover_buy"] = (
        df["uncover_buy"]
        .fillna(0)
    )

    if "imbalance_formula" not in df.columns:
        raise ValueError(
            "Missing imbalance_formula. "
            "Set each source month to either "
            "'base' or 'with_uncover' before transform()."
        )

    invalid_formulas = (
        set(
            df["imbalance_formula"]
            .dropna()
            .astype(str)
            .unique()
        )
        -
        VALID_IMBALANCE_FORMULAS
    )

    if invalid_formulas:
        raise ValueError(
            "Unsupported imbalance formulas: "
            + ", ".join(
                sorted(invalid_formulas)
            )
        )

    if df["imbalance_formula"].isna().any():
        raise ValueError(
            "imbalance_formula contains missing values."
        )

    # ==========================================================
    # RILLLOGARITJA E IMBALANC
    # ==========================================================

    base_imbalance = (
        df["plan_mwh"]
        - df["production_mwh"]
        - df["consumption_mwh"]
        + df["plan_mwh3"]
    )

    uncover_adjustment = (
        df["uncover_sell"]
        - df["uncover_buy"]
    )

    df["imbalance_calculated"] = np.where(
        df["imbalance_formula"].eq(
            "with_uncover"
        ),
        base_imbalance
        + uncover_adjustment,
        base_imbalance,
    )
    # ==========================================================
    # RILLLOGARITJA E TOTAL EURO
    # ==========================================================

    df["total_euro_calculated"] = (
        df["imbalance_calculated"]
        *
        df["price"]
    )

    # ==========================================================
    # PLAN DEVIATION
    # ==========================================================

    df["plan_dev_calculated"] = (
        df["plan_mwh"]
        - df["consumption_mwh"]
        + df["plan_mwh3"]
        - df["production_mwh"]
    )

    # ==========================================================
    # ABSOLUTE ERROR
    # ==========================================================

    df["mape_calculated"] = (
        df["plan_dev_calculated"]
        .abs()
    )

    # ==========================================================
    # MAPE %
    # ==========================================================

    df["mape_percentage_calculated"] = np.where(

        df["consumption_mwh"] != 0,

        (
            df["mape_calculated"]
            /
            df["consumption_mwh"]
        ),

        np.nan
    )

    # ==========================================================
    # ALPEX IMBALANCE
    # ==========================================================

    df["alpex_imbalance_calculated"] = (

        df["imbalance_calculated"]
        *
        (
            df["alpex"]
            -
            df["price"]
        )
    )

    # ==========================================================
    # ALPEX PRICE DIFFERENCE
    # ==========================================================

    df["alpex_imbalance_price_calculated"] = (
        df["alpex"]
        -
        df["price"]
    )

    # ==========================================================
    # PERCENT DEVIATION
    # Compatibility with reporting/anomaly modules
    # ==========================================================

    df["percent_delta"] = (
        df["mape_percentage"]
        * 100
    )

    # ==========================================================
    # SORT
    # ==========================================================

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df
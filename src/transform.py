import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


try:
    TIMEZONE = ZoneInfo("Europe/Pristina")

except ZoneInfoNotFoundError:
    TIMEZONE = ZoneInfo("Europe/Tirane")

def clean_columns(df):

    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    column_map = {

        "Data": "date",

        "Ora": "hour",

        "Furnizuesi X Furnizimi (EMIFurnizuesiX) [MWh]":
            "emi",

        "Konsumi i Nominuar i  Furnizuesi X (EKNKFurnizuesiX) [MWh]":
            "eknk",

        "Energjia e Jobalancit (EJBFFurnizuesiXj) [MWh]":
            "ejb",

        "NEW Actual (abs)":
            "excel_new_actual",

        "OLD Predicted (abs)":
            "excel_old_predicted",

        "DELTA (ABS)":
            "excel_delta",

        "\\=(new-old)/old":
            "excel_percent_delta",

        "Çmimi i jobalancit [€/MWh]":
            "price",

        "Pagesa e Energjisë së Jobalancit   (PEJBj) [€]":
            "excel_payment"
    }

    df = df.rename(
        columns=column_map
    )

    return df

def calculate_metrics(df):

    df["new_actual"] = (
        df["emi"]
        .abs()
    )


    df["old_predicted"] = (
        df["eknk"]
        .abs()
    )


    df["delta"] = (
        df["new_actual"]
        -
        df["old_predicted"]
    ).abs()


    df["percent_delta"] = (
        df["delta"]
        /
        df["old_predicted"]
    ) * 100


    df["payment"] = (
        df["ejb"]
        *
        df["price"]
    )

    return df

def transform(df):

    df = clean_columns(df)
    df = calculate_metrics(df)

    # DST AWARE DATETIME
    # Kosovo timezone
    df["datetime"] = (
        pd.to_datetime(
            df["date"].astype(str)
            + " "
            +
            (df["hour"] - 1).astype(str)
            + ":00"
        )
        .dt.tz_localize(
            TIMEZONE,
            ambiguous="NaT",
            nonexistent="shift_forward"
        )
    )

    return df

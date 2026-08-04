import pandas as pd

def clean_columns(df):

    df.columns = (
    df.columns
    .str.replace("\n", " ", regex=False)
    .str.strip()
)
    """
    Shkurtimi i emrave te kolonave
    """

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

    df["new_actual"]= (
        df["emi"]
        .abs()
    )

    df["old_predicted"]= (
            df["eknk"]
            .abs()
        )

    #DELTA
    df["delta"] = (
        df["new_actual"]
        -
        df["old_predicted"]
    ).abs()

    #%DELTA
    df["percent_delta"] = (
        df["delta"]
        /
        df["old_predicted"]
    )   * 100

    #PAYMEMT
    df["payment"] = (
        df["ejb"]
        *
        df["price"]
    )

    return df

def transform(df):
    
    df = clean_columns(df)
    df = calculate_metrics(df)

    return df



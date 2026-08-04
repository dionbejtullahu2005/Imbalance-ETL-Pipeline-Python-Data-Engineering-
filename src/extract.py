from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_FILE = (
    BASE_DIR
    / "data"
    / "Imbalanc_June_2026.xlsx"
)

def extract_excel():

    """
    Lexon fletet J dhe GJ
    """

    df_j = pd.read_excel(
        EXCEL_FILE,
        sheet_name="J",
        engine="openpyxl"
    )


    df_gj = pd.read_excel(
        EXCEL_FILE,
        sheet_name="GJ",
        header=None,
        engine="openpyxl"
    )


    df_gj = df_gj.iloc[:, [4, 5, 6, 7]]

    df_gj.columns = [
        "Nr. rendor",
        "Emërtimi - Përshkrimi",
        "Sasia",
        "Sqarime"
    ]

    df_gj = df_gj.dropna(
        subset=["Nr. rendor"]
    )


    df_gj = df_gj[
        df_gj["Nr. rendor"] != "Nr. rendor"
    ]


    df_gj["Nr. rendor"] = pd.to_numeric(
        df_gj["Nr. rendor"],
        errors="coerce"
    )


    df_gj = df_gj.dropna(
        subset=["Nr. rendor"]
    )


    df_gj["Nr. rendor"] = (
        df_gj["Nr. rendor"]
        .astype(int)
    )

    return df_j, df_gj

if __name__ == "__main__":
    j, gj = extract_excel()

    print("\n===== Sheet J =====")
    print(j.head())
    print("\nDimensionet:")
    print(j.shape)
    print("\nKolonat:")
    print(j.columns.tolist())

    print("\n===== Sheet GJ =====")
    print(gj.head())
    print("\nDimensionet:")
    print(gj.shape)
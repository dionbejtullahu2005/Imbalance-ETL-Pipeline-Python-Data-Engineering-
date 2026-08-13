from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

EXCEL_PATH = (
    BASE_DIR
    / "data"
    / "Imbalanc June 2026 (1).xlsx"
)


def extract_excel(excel_path=None):
    """Return ``(hourly, summary, prices)`` from a supported workbook."""
    source_path = Path(excel_path) if excel_path is not None else EXCEL_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {source_path}")
    workbook = pd.ExcelFile(source_path)
    required_sheets = {"imbalanc h", "Prices", "summary"}
    missing_sheets = sorted(required_sheets.difference(workbook.sheet_names))
    if missing_sheets:
        raise ValueError("Missing Excel sheets: " + ", ".join(missing_sheets))

    # ==========================================================
    # HOURLY DATA
    # ==========================================================

    df = pd.read_excel(
        source_path,
        sheet_name="imbalanc h"
    )

    # Normalizo emrat e kolonave
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ==========================================================
    # MBAN VETËM ORËT REALE
    # ==========================================================

    df = df[
        df["Supplier"].notna()
        &
        df["Date"].notna()
        &
        df["Hour"].notna()
    ].copy()

    # Vetëm EnerCo
    df = df[
        df["Supplier"]
        .astype(str)
        .str.strip()
        .eq("EnerCo")
    ].copy()

    # ==========================================================
    # DATA
    # ==========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Hour"] = pd.to_numeric(
        df["Hour"],
        errors="coerce"
    )

    # Largo rreshtat invalid
    df = df[
        df["Date"].notna()
        &
        df["Hour"].notna()
    ].copy()

    # ==========================================================
    # SORT
    # ==========================================================

    df = df.sort_values(
        ["Date", "Hour"]
    ).reset_index(drop=True)

    # ==========================================================
    # PRICES
    # ==========================================================

    prices = pd.read_excel(
        source_path,
        sheet_name="Prices"
    )

    prices.columns = (
        prices.columns
        .astype(str)
        .str.strip()
    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    summary = pd.read_excel(
        source_path,
        sheet_name="summary"
    )

    summary.columns = (
        summary.columns
        .astype(str)
        .str.strip()
    )

    return df, summary, prices

def extract_hourly_excel(excel_path):
    """
    Extract hourly EnerCo data from a monthly workbook.
    """

    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel workbook not found: {excel_path}"
        )

    workbook = pd.ExcelFile(
        excel_path
    )

    required_sheet = "imbalanc h"

    if required_sheet not in workbook.sheet_names:
        raise ValueError(
            f"Missing sheet: {required_sheet}"
        )

    df = pd.read_excel(
        excel_path,
        sheet_name=required_sheet,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required_columns = [
        "Supplier",
        "Date",
        "Hour",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing hourly columns: "
            + ", ".join(missing_columns)
        )

    df = df[
        df["Supplier"].notna()
        &
        df["Date"].notna()
        &
        df["Hour"].notna()
    ].copy()

    df = df[
        df["Supplier"]
        .astype(str)
        .str.strip()
        .eq("EnerCo")
    ].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce",
    )

    df["Hour"] = pd.to_numeric(
        df["Hour"],
        errors="coerce",
    )

    df = df[
        df["Date"].notna()
        &
        df["Hour"].between(1, 24)
    ].copy()

    return (
        df.sort_values(
            ["Date", "Hour"]
        )
        .reset_index(drop=True)
    )
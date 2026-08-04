from pathlib import Path
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def save_parquet(df, filename="imbalance.parquet"):
    path = OUTPUT_DIR / filename

    df.to_parquet(
        path,
        index=False
    )

    return path

def save_sqlite(
        df,
        filename="imbalance.db",
        table_name="hourly_imbalance"
):
    path = OUTPUT_DIR / filename

    connection = sqlite3.connect(
        path
    )

    df.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    return path
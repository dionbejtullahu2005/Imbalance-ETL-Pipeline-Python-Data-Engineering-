from src.extract import extract_excel
from src.transform import transform

from src.gj_calculation import calculate_gj_summary
import pandas as pd


def test_gj_calculation():

    df = pd.DataFrame({
        "emi": [1.0, 2.0], "eknk": [2.0, 3.0],
        "ejb": [0.5, -0.25], "payment": [5.0, -2.0],
    })
    df_gj = pd.DataFrame({"Nr. rendor": [7, 8, 10, 21], "Sasia": [3, 1, 20, 2]})

    result = calculate_gj_summary(df, df_gj)


    assert len(result) == 22

    assert "Python_Value" in result.columns


    assert result.loc[
        result["Nr"] == 1,
        "Python_Value"
    ].iloc[0] == df["emi"].sum()

from src.extract import extract_excel
from src.transform import transform

from src.gj_calculation import calculate_gj_summary


def test_gj_calculation():

    df_j, df_gj = extract_excel()


    df = transform(df_j)

    result = calculate_gj_summary(df, df_gj)


    assert len(result) == 22

    assert "Python_Value" in result.columns

    assert result.loc[
        result["Nr"] == 1,
        "Python_Value"
    ].iloc[0] == df["emi"].sum()

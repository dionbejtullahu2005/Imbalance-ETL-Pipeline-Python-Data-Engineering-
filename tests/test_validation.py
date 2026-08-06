from src.validate import validate_time_series

from src.extract import extract_excel
from src.transform import transform



def test_time_series():

    df_j, _ = extract_excel()

    df = transform(df_j)


    result = validate_time_series(df)

    assert result["total_rows"] == 720

    assert result["row_count_pass"]

    assert result["hours_complete"]

    assert result["duplicates_pass"]

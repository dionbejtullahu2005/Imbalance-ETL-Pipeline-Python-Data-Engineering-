import pandas as pd

from src.transform import transform


def test_transform_metrics():

    df = pd.DataFrame(
        {
            "Data": [
                "2026-06-01"
            ],
            "Ora": [
                1
            ],
            "Furnizuesi X Furnizimi (EMIFurnizuesiX) [MWh]": [
                -3.0
            ],
            "Konsumi i Nominuar i  Furnizuesi X (EKNKFurnizuesiX) [MWh]": [
                -4.0
            ],
            "Energjia e Jobalancit (EJBFFurnizuesiXj) [MWh]": [
                1.0
            ],
            "Çmimi i jobalancit [€/MWh]": [
                50
            ]
        }
    )


    result = transform(df)


    assert result["new_actual"].iloc[0] == 3.0

    assert result["old_predicted"].iloc[0] == 4.0

    assert result["delta"].iloc[0] == 1.0

    assert result["percent_delta"].iloc[0] == 25

    assert result["payment"].iloc[0] == 50
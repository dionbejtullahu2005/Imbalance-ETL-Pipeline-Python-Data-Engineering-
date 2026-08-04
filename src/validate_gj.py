import pandas as pd

def compare_gj(
        python_gj,
        excel_gj,
        tolerance=0.01
):
    result=[]        

    for _, row in python_gj.iterrows():
        nr = row['Nr']

        python_value = row["Python_Value"]
        excel_value = (
            excel_gj
            .loc[
                excel_gj["Nr. rendor"] == nr,
                "Sasia"
                ].values
        )

        if len(excel_value) == 0:
            excel_value = None
        else:
            excel_value = excel_value[0]

        if (
            python_value is not None
            and excel_value is not None
        ):
            diff = abs(
                python_value - excel_value
            )

            status = (
                "PASS"
                if diff <= tolerance
                else "FAIL"
            )
        else:
            diff = None
            status = "SKIP"

        result.append(
            {
                "Nr": nr,
                "Python": python_value,
                "Excel": excel_value,
                "Differnece": diff,
                "Status": status
            }
        )
    return pd.DataFrame(result)

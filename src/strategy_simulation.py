import pandas as pd

def simulate_nomination_strategies(df):

    results = []

    strategies = {
        "Base 0%": 0.00,
        "+2% Safety Margins": 0.02,
        "+5% Safety Margins": 0.05,
        "+10% Safety Margins": 0.10
    }

    for name, margin in strategies.items():

        temp = df.copy()

        #nominim i ri
        temp["simulated_nomination"] = (
            temp["old_predicted"]
            * (1 + margin)
        )

        #fabimi ne MWh
        temp["simulation_error_MWh"] = (
            temp["new_actual"]
            -
            temp["simulated_nomination"]
        ).abs()

        #kosto
        temp["simulation_cost_euro"] = (
            temp["simulation_error_MWh"]
            *
            temp["price"].abs()
        )

        results.append(
            {
                "Strategy": name,
                "Margin (%)": margin * 100,
                "MAE (MWh)": round(
                    temp["simulation_error_MWh"].mean(), 
                    6
                ),
                "Total Cost (eur)": round(
                    temp["simulation_cost_euro"].sum(), 
                    2
                ),
                "Average Cost eur/hour": round(
                    temp["simulation_cost_euro"].mean(), 
                    3
                )
            }
        )

    return pd.DataFrame(results)
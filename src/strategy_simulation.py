import pandas as pd


def simulate_nomination_strategies(df):

    results = []

    strategies = {
        "Base 0%": 0.00,
        "+2% Safety Margin": 0.02,
        "+5% Safety Margin": 0.05,
        "+10% Safety Margin": 0.10
    }

    for name, margin in strategies.items():

        temp = df.copy()

        # ======================================================
        # 1. SIMULATED NOMINATION
        # ======================================================

        temp["simulated_nomination"] = (
            temp["plan_mwh"]
            *
            (1 + margin)
        )

        # ======================================================
        # 2. SIMULATED IMBALANCE
        #
        # Formula reale nga Excel:
        #
        # Imbalance =
        # Plan
        # - Production
        # - Consumption
        # + Production Plan
        # + Uncover Sell
        # - Uncover Buy
        # ======================================================

        temp["simulated_imbalance"] = (
            temp["simulated_nomination"]
            - temp["production_mwh"]
            - temp["consumption_mwh"]
            + temp["plan_mwh3"]
        )

        # ======================================================
        # 3. ABSOLUTE IMBALANCE
        #
        # Për MAE përdorim madhësinë absolute të jobalancit.
        # ======================================================

        temp["simulation_error_MWh"] = (
            temp["simulated_imbalance"]
            .abs()
        )

        # ======================================================
        # 4. FINANCIAL COST
        #
        # Për krahasimin e riskut/kostos:
        # |imbalance| * |price|
        # ======================================================

        temp["simulation_cost_euro"] = (
            temp["simulation_error_MWh"]
            *
            temp["price"].abs()
        )

        # ======================================================
        # 5. RESULTS
        # ======================================================

        results.append(
            {
                "Strategy": name,

                "Margin (%)": (
                    margin * 100
                ),

                "MAE (MWh)": round(
                    temp[
                        "simulation_error_MWh"
                    ].mean(),
                    6
                ),

                "Total Cost (eur)": round(
                    temp[
                        "simulation_cost_euro"
                    ].sum(),
                    2
                ),

                "Average Cost eur/hour": round(
                    temp[
                        "simulation_cost_euro"
                    ].mean(),
                    3
                )
            }
        )

    return pd.DataFrame(
        results
    )
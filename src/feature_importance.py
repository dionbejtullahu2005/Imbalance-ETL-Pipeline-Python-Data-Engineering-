import pandas as pd
import matplotlib.pyplot as plt

from src.features import FEATURE_COLUMNS

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


REPORT_DIR = (
    BASE_DIR
    /
    "output"
    /
    "reports"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def extract_feature_importance(
    df_features,
    model,
    model_name
):

    features = [
        col for col in df_features.columns
        if col in FEATURE_COLUMNS
    ]


    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        result = pd.DataFrame(
            {
                "Feature": features,
                "Importance": importance
            }
        )


    elif hasattr(model, "coef_"):

        coefficients = model.coef_

        result = pd.DataFrame(
            {
                "Feature": features,
                "Coefficient": coefficients,
                "Importance": abs(coefficients)
            }
        )

        result = result.sort_values(
            "Importance",
            ascending=False
        )

        return result


    else:

        raise ValueError(
            "Model does not support feature importance"
        )


    result = result.sort_values(
        "Importance",
        ascending=False
    )


    return result

def plot_feature_importance(
        importance_df,
        model_name
):

    plt.figure(
        figsize=(10,6)
    )


    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )


    plt.gca().invert_yaxis()


    plt.title(
        f"Feature Importance - {model_name}"
    )


    plt.xlabel(
        "Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.grid(
        axis="x"
    )


    output_file = (
        REPORT_DIR
        /
        f"{model_name.lower().replace(' ','_')}_feature_importance.png"
    )


    plt.savefig(
        output_file,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Feature importance graph saved:",
        output_file
    )


    return output_file

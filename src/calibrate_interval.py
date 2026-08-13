import numpy as np
import pandas as pd

from src.evaluation import recursive_timeseries_residuals


def calibrate_conformal_interval(
    model,
    df,
    feature_columns,
    target_column="imbalance",
    confidence=0.90,
    splits=5,
    min_hourly_samples=30
):

    data = (
        df
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    required_columns = (
        feature_columns
        +
        [
            "datetime",
            "hour",
            target_column
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing calibration columns: "
            + ", ".join(missing_columns)
        )

    # Preserve the initial raw history. It contains the observations needed
    # to construct lag168 for the first recursive validation horizon.
    data = data.dropna(
        subset=[target_column, "hour", "datetime", "temperature"]
    ).copy()

    if len(data) < 100:
        raise ValueError(
            "Not enough historical rows "
            "for conformal calibration."
        )


    residual_df = recursive_timeseries_residuals(model, data, splits=splits)

    if residual_df.empty:
        raise ValueError(
            "No valid residuals generated."
        )


    # GLOBAL CONFORMAL RADIUS
    absolute_residuals = (
        residual_df[
            "absolute_residual"
        ]
        .to_numpy()
    )

    alpha = (
        1.0
        -
        confidence
    )

    n = len(
        absolute_residuals
    )

    global_quantile_level = (
        np.ceil(
            (n + 1)
            *
            (1 - alpha)
        )
        /
        n
    )

    global_quantile_level = min(
        global_quantile_level,
        1.0
    )

    global_radius = float(
        np.quantile(
            absolute_residuals,
            global_quantile_level,
            method="higher"
        )
    )


    # EXPECTED ERROR
    expected_error_mwh = float(
        np.mean(
            absolute_residuals
        )
    )

    median_error_mwh = float(
        np.median(
            absolute_residuals
        )
    )

    bias = float(
        residual_df[
            "signed_residual"
        ]
        .mean()
    )


    # HOURLY RADII
    hourly_radii = {}

    hourly_stats = []

    for hour in range(
        1,
        25
    ):

        hour_errors = (
            residual_df.loc[
                residual_df[
                    "hour"
                ]
                ==
                hour,
                "absolute_residual"
            ]
            .to_numpy()
        )

        count = len(
            hour_errors
        )

        if (
            count
            >=
            min_hourly_samples
        ):

            hourly_quantile_level = (
                np.ceil(
                    (count + 1)
                    *
                    (1 - alpha)
                )
                /
                count
            )

            hourly_quantile_level = min(
                hourly_quantile_level,
                1.0
            )

            radius = float(
                np.quantile(
                    hour_errors,
                    hourly_quantile_level,
                    method="higher"
                )
            )

            source = "hourly"

        else:

            radius = (
                global_radius
            )

            source = "global_fallback"


        hourly_radii[
            hour
        ] = radius


        hourly_stats.append(
            {
                "hour": hour,
                "samples": count,
                "radius": radius,
                "source": source
            }
        )


    hourly_stats_df = pd.DataFrame(
        hourly_stats
    )


    # HISTORICAL CONDITIONAL COVERAGE
    residual_df[
        "radius"
    ] = (
        residual_df[
            "hour"
        ]
        .map(
            hourly_radii
        )
    )

    residual_df[
        "covered"
    ] = (
        residual_df[
            "absolute_residual"
        ]
        <=
        residual_df[
            "radius"
        ]
    )

    conditional_coverage = (
        residual_df[
            "covered"
        ]
        .mean()
        *
        100
    )


    # PRINT RESULTS
    print()
    print("=" * 70)
    print(
        "HOURLY CONDITIONAL CONFORMAL CALIBRATION"
    )
    print("=" * 70)

    print(
        "Historical usable rows:",
        len(data)
    )

    print(
        "OOS residuals:",
        len(
            residual_df
        )
    )

    print(
        f"Confidence target: "
        f"{confidence * 100:.0f}%"
    )

    print(
        f"Global fallback radius: "
        f"{global_radius:.6f} MWh"
    )

    print(
        f"Expected OOS error: "
        f"{expected_error_mwh:.6f} MWh"
    )

    print(
        f"Median OOS error: "
        f"{median_error_mwh:.6f} MWh"
    )

    print(
        f"OOS residual bias: "
        f"{bias:.6f} MWh"
    )

    print(
        f"Conditional historical coverage: "
        f"{conditional_coverage:.2f}%"
    )

    print("\nHOURLY RADII")

    print(
        hourly_stats_df
        .to_string(
            index=False
        )
    )


    return {
        "global_radius":
            global_radius,

        "hourly_radii":
            hourly_radii,

        "expected_error_mwh":
            expected_error_mwh,

        "median_error_mwh":
            median_error_mwh,

        "bias":
            bias,

        "coverage":
            conditional_coverage,

        "residuals":
            residual_df,

        "hourly_stats":
            hourly_stats_df
    }

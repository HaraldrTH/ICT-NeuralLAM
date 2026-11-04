"""Compare wind speed and direction from paired point-level u10/v10 results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_lam.evaluation.normalization import (
    compute_derived_training_statistics,
)


DEFAULT_INPUT = Path(
    "evaluation_results/point_comparisons/grimstad/test/grimstad_point_cases.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "evaluation_results/point_comparisons/grimstad/test/figures"
)
CASE_COLUMNS = (
    "model_id",
    "checkpoint_id",
    "dataset",
    "split",
    "target_latitude",
    "target_longitude",
    "grid_y_index",
    "grid_x_index",
    "cell_latitude",
    "cell_longitude",
    "distance_to_target_km",
    "initialization_time_unix_s",
    "initialization_utc",
    "valid_time_unix_s",
    "valid_time_utc",
    "lead_hours",
)


def _wind_direction_from_components(u, v):
    """Return meteorological wind-from direction in degrees clockwise from north."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def _circular_absolute_difference(first, second):
    """Return the smallest absolute angular difference, in degrees."""
    return np.abs((first - second + 180.0) % 360.0 - 180.0)


def derive_wind_vector_cases(
    point_cases,
    calm_threshold=0.5,
    wind_speed_training_std=None,
):
    """Derive per-case wind speed and direction errors from paired u10/v10 rows."""
    if not np.isfinite(calm_threshold) or calm_threshold < 0:
        raise ValueError("calm_threshold must be a finite, non-negative value")

    required = {
        *CASE_COLUMNS,
        "variable",
        "unit",
        "prediction",
        "target",
        "persistence_prediction",
    }
    missing = required - set(point_cases.columns)
    if missing:
        raise ValueError(
            "Point-case CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    wind = point_cases[point_cases["variable"].isin(("u10", "v10"))].copy()
    if wind.empty:
        raise ValueError("Point-case CSV contains no u10/v10 rows")
    units = wind.groupby("variable")["unit"].agg(lambda values: set(values))
    for variable in ("u10", "v10"):
        if units.get(variable) != {"m/s"}:
            raise ValueError(f"{variable} must be present and use m/s units")

    if not np.isfinite(wind[["prediction", "target"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Wind component predictions and targets must be finite")

    index_columns = list(CASE_COLUMNS)
    if wind.duplicated(index_columns + ["variable"]).any():
        raise ValueError("Duplicate u10/v10 rows found for a model case")

    wide = wind.pivot(
        index=index_columns,
        columns="variable",
        values=["prediction", "target", "persistence_prediction"],
    )
    expected_columns = {
        (field, variable)
        for field in ("prediction", "target", "persistence_prediction")
        for variable in ("u10", "v10")
    }
    if set(wide.columns) != expected_columns:
        raise ValueError("Every model case must contain both u10 and v10 components")
    wide.columns = [f"{field}_{variable}" for field, variable in wide.columns]
    wide = wide.reset_index()

    shared_case_columns = [
        column for column in CASE_COLUMNS
        if column not in {"model_id", "checkpoint_id"}
    ]
    for component in ("u10", "v10"):
        target_wide = wide.pivot(
            index=shared_case_columns,
            columns="model_id",
            values=f"target_{component}",
        )
        if not np.allclose(
            target_wide.to_numpy(dtype=np.float64),
            target_wide.iloc[:, :1].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"Observed {component} targets differ between model caches"
            )

    prediction_speed = np.hypot(
        wide["prediction_u10"], wide["prediction_v10"]
    )
    observed_speed = np.hypot(wide["target_u10"], wide["target_v10"])
    persistence_speed = np.hypot(
        wide["persistence_prediction_u10"],
        wide["persistence_prediction_v10"],
    )
    if wind_speed_training_std is None:
        datasets = point_cases["dataset"].drop_duplicates().tolist()
        if len(datasets) != 1:
            raise ValueError("Point-case CSV must contain exactly one dataset")
        wind_speed_training_std = compute_derived_training_statistics(
            datasets[0]
        )["wind_speed_10m_interior"]["std"]
    if (
        not np.isfinite(wind_speed_training_std)
        or wind_speed_training_std <= 0
    ):
        raise ValueError("wind_speed_training_std must be finite and positive")
    prediction_direction = _wind_direction_from_components(
        wide["prediction_u10"].to_numpy(dtype=np.float64),
        wide["prediction_v10"].to_numpy(dtype=np.float64),
    )
    observed_direction = _wind_direction_from_components(
        wide["target_u10"].to_numpy(dtype=np.float64),
        wide["target_v10"].to_numpy(dtype=np.float64),
    )

    direction_valid = observed_speed.to_numpy() >= calm_threshold
    direction_error = np.full(len(wide), np.nan, dtype=np.float64)
    direction_error[direction_valid] = _circular_absolute_difference(
        prediction_direction[direction_valid],
        observed_direction[direction_valid],
    )
    prediction_calm = prediction_speed.to_numpy() < calm_threshold
    direction_error[direction_valid & prediction_calm] = 180.0
    persistence_direction = _wind_direction_from_components(
        wide["persistence_prediction_u10"].to_numpy(dtype=np.float64),
        wide["persistence_prediction_v10"].to_numpy(dtype=np.float64),
    )
    persistence_calm = persistence_speed.to_numpy() < calm_threshold
    persistence_direction_error = np.full(len(wide), np.nan, dtype=np.float64)
    persistence_direction_error[direction_valid] = _circular_absolute_difference(
        persistence_direction[direction_valid],
        observed_direction[direction_valid],
    )
    persistence_direction_error[direction_valid & persistence_calm] = 180.0

    result = wide[index_columns].copy()
    result["prediction_speed"] = prediction_speed
    result["observed_speed"] = observed_speed
    result["speed_error"] = prediction_speed - observed_speed
    result["absolute_speed_error"] = np.abs(result["speed_error"])
    result["squared_speed_error"] = np.square(result["speed_error"])
    result["persistence_speed"] = persistence_speed
    result["persistence_speed_error"] = persistence_speed - observed_speed
    result["persistence_absolute_speed_error"] = np.abs(
        result["persistence_speed_error"]
    )
    result["persistence_squared_speed_error"] = np.square(
        result["persistence_speed_error"]
    )
    result["wind_speed_training_std"] = wind_speed_training_std
    result["normalized_absolute_speed_error"] = (
        result["absolute_speed_error"] / wind_speed_training_std
    )
    result["prediction_direction_from_degrees"] = prediction_direction
    result["persistence_direction_from_degrees"] = persistence_direction
    result["observed_direction_from_degrees"] = observed_direction
    result["absolute_direction_error_degrees"] = direction_error
    result["squared_direction_error_degrees"] = np.square(direction_error)
    result["persistence_absolute_direction_error_degrees"] = (
        persistence_direction_error
    )
    result["persistence_squared_direction_error_degrees"] = np.square(
        persistence_direction_error
    )
    result["normalized_absolute_direction_error"] = direction_error / 180.0
    result["direction_error_normalization_degrees"] = 180.0
    result["observed_direction_valid"] = direction_valid
    result["prediction_is_calm"] = prediction_calm
    result["persistence_is_calm"] = persistence_calm
    result["calm_threshold_m_per_s"] = calm_threshold
    return result


def summarize_wind_vector_cases(cases):
    """Summarize speed and circular direction errors by model and forecast lead."""
    summary = (
        cases.groupby(["model_id", "lead_hours"], sort=True, as_index=False)
        .agg(
            speed_mae=("absolute_speed_error", "mean"),
            speed_mse=("squared_speed_error", "mean"),
            persistence_speed_mae=("persistence_absolute_speed_error", "mean"),
            persistence_speed_mse=("persistence_squared_speed_error", "mean"),
            normalized_speed_mae=("normalized_absolute_speed_error", "mean"),
            direction_mae_degrees=("absolute_direction_error_degrees", "mean"),
            direction_mse_degrees=("squared_direction_error_degrees", "mean"),
            normalized_direction_mae=(
                "normalized_absolute_direction_error",
                "mean",
            ),
            persistence_direction_mae_degrees=(
                "persistence_absolute_direction_error_degrees",
                "mean",
            ),
            persistence_direction_mse_degrees=(
                "persistence_squared_direction_error_degrees",
                "mean",
            ),
            sample_count=("absolute_speed_error", "count"),
            direction_sample_count=(
                "absolute_direction_error_degrees",
                "count",
            ),
        )
    )
    summary["speed_rmse"] = np.sqrt(summary.pop("speed_mse"))
    summary["persistence_speed_rmse"] = np.sqrt(
        summary.pop("persistence_speed_mse")
    )
    summary["speed_mae_skill_vs_persistence"] = (
        1.0 - summary["speed_mae"] / summary["persistence_speed_mae"]
    ).where(summary["persistence_speed_mae"] > 0)
    summary["direction_rmse_degrees"] = np.sqrt(
        summary.pop("direction_mse_degrees")
    )
    summary["persistence_direction_rmse_degrees"] = np.sqrt(
        summary.pop("persistence_direction_mse_degrees")
    )
    summary["direction_mae_skill_vs_persistence"] = (
        1.0
        - summary["direction_mae_degrees"]
        / summary["persistence_direction_mae_degrees"]
    ).where(summary["persistence_direction_mae_degrees"] > 0)
    return summary[
        [
            "model_id",
            "lead_hours",
            "speed_mae",
            "speed_rmse",
            "normalized_speed_mae",
            "persistence_speed_mae",
            "persistence_speed_rmse",
            "speed_mae_skill_vs_persistence",
            "direction_mae_degrees",
            "direction_rmse_degrees",
            "normalized_direction_mae",
            "persistence_direction_mae_degrees",
            "persistence_direction_rmse_degrees",
            "direction_mae_skill_vs_persistence",
            "sample_count",
            "direction_sample_count",
        ]
    ]


def plot_wind_vector_comparison(summary, cases, output_dir, calm_threshold):
    """Save unique PNG/PDF comparison plots for speed and direction errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_ids = summary["model_id"].drop_duplicates().tolist()
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = list(axes.flat)
    metrics = (
        ("speed_mae", "10 m wind-speed MAE", "MAE (m/s)"),
        ("speed_rmse", "10 m wind-speed RMSE", "RMSE (m/s)"),
        (
            "normalized_speed_mae",
            "Normalized 10 m wind-speed MAE",
            "MAE / training wind-speed standard deviation",
        ),
        (
            "speed_mae_skill_vs_persistence",
            "10 m wind-speed MAE skill vs persistence",
            "Skill (1 − model MAE / persistence MAE)",
        ),
        (
            "direction_mae_degrees",
            "10 m wind-direction MAE",
            "MAE (degrees)",
        ),
        (
            "direction_rmse_degrees",
            "10 m wind-direction RMSE",
            "RMSE (degrees)",
        ),
        (
            "normalized_direction_mae",
            "Normalized 10 m wind-direction MAE",
            "MAE / 180° maximum circular error",
        ),
        (
            "direction_mae_skill_vs_persistence",
            "10 m wind-direction MAE skill vs persistence",
            "Skill (1 − model MAE / persistence MAE)",
        ),
    )
    for model_id in model_ids:
        model_summary = summary[summary["model_id"] == model_id]
        for axis, (metric, _, _) in zip(axes, metrics):
            axis.plot(
                model_summary["lead_hours"],
                model_summary[metric],
                marker="o",
                linewidth=1.8,
                label=model_id,
            )

    for axis, (_, title, ylabel) in zip(axes, metrics):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Forecast lead (hours)")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    first = cases.iloc[0]
    figure.suptitle(
        "10 m wind-vector comparison at "
        f"{first['cell_latitude']:.2f}°N, {first['cell_longitude']:.2f}° "
        f"({first['distance_to_target_km']:.1f} km from target)\n"
        f"Direction scored when observed speed ≥ {calm_threshold:g} m/s; "
        "calm predictions receive 180° error"
    )
    figure.tight_layout()
    stem = "grimstad_wind_vector_comparison"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare Grimstad point wind speed and direction using paired "
            "10 m u/v component results."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calm-threshold",
        type=float,
        default=0.5,
        help=(
            "Observed wind speed threshold in m/s for direction scoring "
            "(default: 0.5)"
        ),
    )
    args = parser.parse_args(argv)
    if not args.input_csv.is_file():
        parser.error(f"Point-case CSV does not exist: {args.input_csv}")

    point_cases = pd.read_csv(args.input_csv)
    datasets = point_cases["dataset"].drop_duplicates().tolist()
    if len(datasets) != 1:
        parser.error("Input CSV must contain exactly one dataset")
    wind_speed_std = compute_derived_training_statistics(
        datasets[0]
    )["wind_speed_10m_interior"]["std"]
    wind_cases = derive_wind_vector_cases(
        point_cases,
        calm_threshold=args.calm_threshold,
        wind_speed_training_std=wind_speed_std,
    )
    summary = summarize_wind_vector_cases(wind_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "grimstad_wind_vector_cases.csv"
    summary_path = args.output_dir / "grimstad_wind_vector_metrics.csv"
    wind_cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)
    png_path, pdf_path = plot_wind_vector_comparison(
        summary,
        wind_cases,
        args.output_dir,
        args.calm_threshold,
    )
    print(f"Wrote {len(wind_cases)} paired wind cases to {cases_path}")
    print(f"Wrote per-lead wind-vector metrics to {summary_path}")
    print(f"Wrote comparison plots to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()

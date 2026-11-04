"""Extract and plot forecast performance at a nearest grid cell."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from neural_lam import constants, utils
from neural_lam.evaluation.forecast_cache import open_forecast_cache
from neural_lam.evaluation.normalization import (
    OCEAN_LAND_FRACTION_THRESHOLD,
    compute_derived_training_statistics,
)


DEFAULT_MODEL_IDS = (
    "1L-LAM",
    "Graphcast-LAM",
    "HI-LAM",
    "HI-LAM-Parallel",
)
DEFAULT_PLOT_VARIABLES = ("t2m", "d2m", "tp", "msl", "sst")
PLOT_METRICS = (
    ("normalized_mae", "Normalized MAE"),
    ("normalized_rmse", "Normalized RMSE"),
    ("mae_skill_vs_persistence", "MAE skill vs persistence"),
    ("rmse_skill_vs_persistence", "RMSE skill vs persistence"),
)
WIND_VECTOR_METRICS = PLOT_METRICS
WIND_CASE_COLUMNS = (
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


def nearest_grid_cell(latitude, longitude, target_latitude, target_longitude):
    """Return nearest grid indices, cell coordinates, and great-circle distance."""
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Cache latitude and longitude coordinates must be 1D")
    if not len(latitude) or not len(longitude):
        raise ValueError("Cache latitude and longitude coordinates cannot be empty")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Cache coordinates must be finite")
    if not -90 <= target_latitude <= 90 or not -180 <= target_longitude <= 180:
        raise ValueError("Target coordinates must be valid latitude/longitude values")

    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    target_latitude_rad = np.radians(target_latitude)
    delta_latitude = np.radians(lat_grid - target_latitude)
    delta_longitude = np.radians(lon_grid - target_longitude)
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(target_latitude_rad)
        * np.cos(np.radians(lat_grid))
        * np.sin(delta_longitude / 2) ** 2
    )
    distances = 2 * 6371.0088 * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )
    y_index, x_index = np.unravel_index(np.argmin(distances), distances.shape)
    return (
        int(y_index),
        int(x_index),
        float(lat_grid[y_index, x_index]),
        float(lon_grid[y_index, x_index]),
        float(distances[y_index, x_index]),
    )


def _find_model_caches(cache_dir, model_ids, split):
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Forecast cache directory does not exist: {cache_dir}")

    found = {model_id: [] for model_id in model_ids}
    incomplete = {model_id: [] for model_id in model_ids}
    for path in sorted(cache_dir.glob("*.zarr")):
        group = zarr.open_group(str(path), mode="r")
        model_id = group.attrs.get("model_id")
        if model_id not in found or group.attrs.get("split") != split:
            continue
        if group.attrs.get("complete") and group.attrs.get("status") == "complete":
            found[model_id].append(path)
        else:
            incomplete[model_id].append(path)

    selected = {}
    for model_id in model_ids:
        if len(found[model_id]) == 1:
            selected[model_id] = found[model_id][0]
            continue
        if len(found[model_id]) > 1:
            paths = ", ".join(str(path) for path in found[model_id])
            raise ValueError(
                f"Multiple complete {split} caches found for {model_id}: {paths}"
            )
        if incomplete[model_id]:
            raise RuntimeError(
                f"Only incomplete {split} caches found for {model_id}: "
                + ", ".join(str(path) for path in incomplete[model_id])
            )
        raise FileNotFoundError(
            f"No complete {split} cache found for model {model_id} in {cache_dir}"
        )
    return selected


def _load_training_std(dataset, variable_names, selected_variables):
    stats = utils.load_dataset_stats(dataset, "cpu")
    training_std = stats["data_std"].detach().cpu().numpy().astype(np.float64)
    if training_std.shape != (len(constants.param_names),):
        raise ValueError("Dataset training standard deviations have an unexpected shape")
    std_by_variable = {
        name: float(training_std[constants.param_names.index(name)])
        for name in variable_names
    }

    if "sst" in selected_variables:
        derived_stats = compute_derived_training_statistics(dataset)
        std_by_variable["sst"] = derived_stats["sst_ocean_dominant_interior"]["std"]
    return std_by_variable


def _utc_string(timestamp):
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def _point_summary(case_records, std_by_variable):
    frame = pd.DataFrame.from_records(case_records)
    group_columns = ["model_id", "lead_hours", "variable", "unit"]
    summary = (
        frame.groupby(group_columns, sort=True, as_index=False)
        .agg(
            mae=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
            bias=("error", "mean"),
            persistence_mae=("persistence_absolute_error", "mean"),
            persistence_mean_squared_error=("persistence_squared_error", "mean"),
            sample_count=("error", "count"),
        )
    )
    summary["rmse"] = np.sqrt(summary.pop("mean_squared_error"))
    summary["persistence_rmse"] = np.sqrt(
        summary.pop("persistence_mean_squared_error")
    )
    summary["normalization_std"] = summary["variable"].map(std_by_variable)
    summary["normalized_mae"] = summary["mae"] / summary["normalization_std"]
    summary["normalized_rmse"] = summary["rmse"] / summary["normalization_std"]
    summary["mae_skill_vs_persistence"] = (
        1.0 - summary["mae"] / summary["persistence_mae"]
    ).where(summary["persistence_mae"] > 0)
    summary["rmse_skill_vs_persistence"] = (
        1.0 - summary["rmse"] / summary["persistence_rmse"]
    ).where(summary["persistence_rmse"] > 0)
    return frame, summary


def _wind_direction_from_components(u, v):
    """Return meteorological wind-from direction clockwise from north."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def _circular_absolute_difference(first, second):
    """Return the smallest absolute angular difference, in degrees."""
    return np.abs((first - second + 180.0) % 360.0 - 180.0)


def derive_wind_vector_cases(
    point_cases,
    *,
    calm_threshold=0.5,
    wind_speed_training_std=None,
):
    """Derive speed and circular direction errors from paired u10/v10 cases."""
    if not np.isfinite(calm_threshold) or calm_threshold < 0:
        raise ValueError("calm_threshold must be a finite, non-negative value")

    required = {
        *WIND_CASE_COLUMNS,
        "variable",
        "unit",
        "prediction",
        "target",
        "persistence_prediction",
    }
    missing = required - set(point_cases.columns)
    if missing:
        raise ValueError(
            "Point-case data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    wind = point_cases[point_cases["variable"].isin(("u10", "v10"))].copy()
    if wind.empty:
        raise ValueError("Point cases contain no u10/v10 rows")
    units = wind.groupby("variable")["unit"].agg(lambda values: set(values))
    for variable in ("u10", "v10"):
        if units.get(variable) != {"m/s"}:
            raise ValueError(f"{variable} must be present and use m/s units")
    if not np.isfinite(
        wind[["prediction", "target"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Wind component predictions and targets must be finite")

    index_columns = list(WIND_CASE_COLUMNS)
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
        column for column in WIND_CASE_COLUMNS
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
            raise ValueError(f"Observed {component} targets differ between model caches")

    prediction_speed = np.hypot(wide["prediction_u10"], wide["prediction_v10"])
    observed_speed = np.hypot(wide["target_u10"], wide["target_v10"])
    persistence_speed = np.hypot(
        wide["persistence_prediction_u10"],
        wide["persistence_prediction_v10"],
    )
    if wind_speed_training_std is None:
        datasets = point_cases["dataset"].drop_duplicates().tolist()
        if len(datasets) != 1:
            raise ValueError("Point cases must contain exactly one dataset")
        wind_speed_training_std = compute_derived_training_statistics(
            datasets[0]
        )["wind_speed_10m_interior"]["std"]
    if not np.isfinite(wind_speed_training_std) or wind_speed_training_std <= 0:
        raise ValueError("wind_speed_training_std must be finite and positive")

    prediction_direction = _wind_direction_from_components(
        wide["prediction_u10"].to_numpy(dtype=np.float64),
        wide["prediction_v10"].to_numpy(dtype=np.float64),
    )
    observed_direction = _wind_direction_from_components(
        wide["target_u10"].to_numpy(dtype=np.float64),
        wide["target_v10"].to_numpy(dtype=np.float64),
    )
    persistence_direction = _wind_direction_from_components(
        wide["persistence_prediction_u10"].to_numpy(dtype=np.float64),
        wide["persistence_prediction_v10"].to_numpy(dtype=np.float64),
    )

    direction_valid = observed_speed.to_numpy() >= calm_threshold
    direction_error = np.full(len(wide), np.nan, dtype=np.float64)
    direction_error[direction_valid] = _circular_absolute_difference(
        prediction_direction[direction_valid],
        observed_direction[direction_valid],
    )
    prediction_calm = prediction_speed.to_numpy() < calm_threshold
    direction_error[direction_valid & prediction_calm] = 180.0

    persistence_direction_error = np.full(len(wide), np.nan, dtype=np.float64)
    persistence_direction_error[direction_valid] = _circular_absolute_difference(
        persistence_direction[direction_valid],
        observed_direction[direction_valid],
    )
    persistence_calm = persistence_speed.to_numpy() < calm_threshold
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
    result["prediction_direction_from_degrees"] = prediction_direction
    result["observed_direction_from_degrees"] = observed_direction
    result["persistence_direction_from_degrees"] = persistence_direction
    result["absolute_direction_error_degrees"] = direction_error
    result["squared_direction_error_degrees"] = np.square(direction_error)
    result["persistence_absolute_direction_error_degrees"] = (
        persistence_direction_error
    )
    result["persistence_squared_direction_error_degrees"] = np.square(
        persistence_direction_error
    )
    result["observed_direction_valid"] = direction_valid
    result["prediction_is_calm"] = prediction_calm
    result["persistence_is_calm"] = persistence_calm
    result["calm_threshold_m_per_s"] = calm_threshold
    return result


def summarize_wind_vector_cases(cases):
    """Summarize wind speed and direction using the standard point metrics."""
    metric_fields = (
        (
            "wind_speed_10m",
            "m/s",
            "wind_speed_training_std",
            "absolute_speed_error",
            "squared_speed_error",
            "persistence_absolute_speed_error",
            "persistence_squared_speed_error",
        ),
        (
            "wind_direction_10m",
            "degrees",
            None,
            "absolute_direction_error_degrees",
            "squared_direction_error_degrees",
            "persistence_absolute_direction_error_degrees",
            "persistence_squared_direction_error_degrees",
        ),
    )
    records = []
    for (
        variable,
        unit,
        normalization_column,
        error_column,
        squared_error_column,
        persistence_error_column,
        persistence_squared_error_column,
    ) in metric_fields:
        grouped = (
            cases.groupby(["model_id", "lead_hours"], sort=True, as_index=False)
            .agg(
                mae=(error_column, "mean"),
                mean_squared_error=(squared_error_column, "mean"),
                persistence_mae=(persistence_error_column, "mean"),
                persistence_mean_squared_error=(
                    persistence_squared_error_column,
                    "mean",
                ),
                sample_count=(error_column, "count"),
            )
        )
        grouped["rmse"] = np.sqrt(grouped.pop("mean_squared_error"))
        grouped["persistence_rmse"] = np.sqrt(
            grouped.pop("persistence_mean_squared_error")
        )
        normalization_std = (
            float(cases[normalization_column].iloc[0])
            if normalization_column
            else 180.0
        )
        grouped["variable"] = variable
        grouped["unit"] = unit
        grouped["normalization_std"] = normalization_std
        grouped["normalized_mae"] = grouped["mae"] / normalization_std
        grouped["normalized_rmse"] = grouped["rmse"] / normalization_std
        grouped["mae_skill_vs_persistence"] = (
            1.0 - grouped["mae"] / grouped["persistence_mae"]
        ).where(grouped["persistence_mae"] > 0)
        grouped["rmse_skill_vs_persistence"] = (
            1.0 - grouped["rmse"] / grouped["persistence_rmse"]
        ).where(grouped["persistence_rmse"] > 0)
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def summarize_model_comparisons(summary_frame):
    """Return macro normalized errors and pairwise relative MAE reductions."""
    model_ids = summary_frame["model_id"].drop_duplicates().tolist()
    if len(model_ids) < 2:
        raise ValueError("At least two models are required for pairwise comparisons")

    scores = summary_frame.pivot(
        index=["variable", "lead_hours"],
        columns="model_id",
        values="normalized_mae",
    )
    if scores[model_ids].isna().any().any():
        raise ValueError("Models must have matching variables and forecast leads")

    pairwise_records = []
    for baseline, compared in combinations(model_ids, 2):
        baseline_scores = scores[baseline]
        compared_scores = scores[compared]
        reductions = (
            100.0 * (1.0 - compared_scores / baseline_scores)
        ).where(baseline_scores > 0)
        pairwise_records.extend(
            {
                "baseline_model": baseline,
                "compared_model": compared,
                "variable": variable,
                "lead_hours": int(lead_hours),
                "baseline_normalized_mae": float(baseline_scores.loc[(variable, lead_hours)]),
                "compared_normalized_mae": float(compared_scores.loc[(variable, lead_hours)]),
                "relative_mae_reduction_pct": float(
                    reductions.loc[(variable, lead_hours)]
                ),
            }
            for variable, lead_hours in scores.index
        )

    overall_scores = summary_frame.groupby("model_id")["normalized_mae"].mean()
    overall_records = [
        {
            "baseline_model": baseline,
            "compared_model": compared,
            "baseline_mean_normalized_mae": float(overall_scores[baseline]),
            "compared_mean_normalized_mae": float(overall_scores[compared]),
            "relative_mae_reduction_pct": (
                float(100.0 * (1.0 - overall_scores[compared] / overall_scores[baseline]))
                if overall_scores[baseline] > 0
                else np.nan
            ),
            "aggregation": "equal mean across all included variables and leads",
        }
        for baseline, compared in combinations(model_ids, 2)
    ]
    macro_summary = (
        summary_frame.groupby(["model_id", "lead_hours"], sort=True, as_index=False)
        .agg(
            mean_normalized_mae=("normalized_mae", "mean"),
            variable_count=("variable", "nunique"),
        )
    )
    return (
        macro_summary,
        pd.DataFrame.from_records(pairwise_records),
        pd.DataFrame.from_records(overall_records),
    )


def plot_macro_normalized_mae(macro_summary, output_dir):
    """Plot equal-variable mean normalized MAE for direct model comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model_id in macro_summary["model_id"].drop_duplicates():
        model_rows = macro_summary[macro_summary["model_id"] == model_id]
        axis.plot(
            model_rows["lead_hours"],
            model_rows["mean_normalized_mae"],
            marker="o",
            linewidth=1.8,
            label=model_id,
        )
    axis.set_title("Grimstad equal-variable mean normalized MAE")
    axis.set_xlabel("Forecast lead (hours)")
    axis.set_ylabel("Mean across variables of MAE / training standard deviation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    stem = "grimstad_mean_normalized_mae_comparison"
    figure.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def extract_point_results(
    cache_paths,
    *,
    target_latitude,
    target_longitude,
    split,
    variables=None,
    training_std=None,
):
    """Read aligned caches and calculate point-level records and summaries."""
    if not cache_paths:
        raise ValueError("At least one forecast cache is required")

    cache_paths = {str(model_id): Path(path) for model_id, path in cache_paths.items()}
    if len(cache_paths) != len(set(cache_paths)):
        raise ValueError("Model identifiers must be unique")

    groups = {
        model_id: open_forecast_cache(path)
        for model_id, path in cache_paths.items()
    }
    reference_model = next(iter(groups))
    reference = groups[reference_model]
    reference_metadata = dict(reference.attrs)
    if reference_metadata.get("split") != split:
        raise ValueError(
            f"Requested split {split!r} does not match cache split "
            f"{reference_metadata.get('split')!r}"
        )
    dataset = reference_metadata.get("dataset")
    if not dataset:
        raise ValueError("Cache metadata is missing its dataset name")

    variable_names = tuple(reference_metadata.get("variable_names", ()))
    variable_units = tuple(reference_metadata.get("variable_units", ()))
    if len(variable_names) != len(variable_units):
        raise ValueError("Cache variable names and units have different lengths")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Cache variable names must be unique")
    selected_variables = tuple(variables) if variables else variable_names
    unknown_variables = set(selected_variables) - set(variable_names)
    if unknown_variables:
        raise ValueError(
            "Requested variables are absent from the caches: "
            + ", ".join(sorted(unknown_variables))
        )

    latitude = np.asarray(reference["latitude"][:])
    longitude = np.asarray(reference["longitude"][:])
    y_index, x_index, cell_lat, cell_lon, distance_km = nearest_grid_cell(
        latitude, longitude, target_latitude, target_longitude
    )
    lead_hours = np.asarray(reference["lead_hours"][:], dtype=np.int64)
    initializations = np.asarray(
        reference["initialization_time_unix_s"][:], dtype=np.int64
    )
    valid_times = np.asarray(reference["valid_time_unix_s"][:], dtype=np.int64)
    reference_shape = reference["prediction"].shape
    ny, nx = reference_shape[2:4]
    if latitude.shape != (ny,) or longitude.shape != (nx,):
        raise ValueError("Cache coordinate lengths do not match prediction grid")
    if y_index >= ny or x_index >= nx:
        raise ValueError("Nearest grid cell is outside the cached forecast grid")
    if bool(reference["boundary_mask"][y_index, x_index]):
        raise ValueError(
            "The nearest grid cell is boundary-forced and is excluded from "
            "evaluation metrics; choose another coordinate or boundary policy"
        )

    common_land_sea_fraction = float(reference["land_sea_mask"][y_index, x_index])
    selected_indices = [variable_names.index(name) for name in selected_variables]
    std_by_variable = (
        dict(training_std)
        if training_std is not None
        else _load_training_std(dataset, variable_names, selected_variables)
    )
    for variable in selected_variables:
        if variable not in std_by_variable:
            raise ValueError(f"No training normalization standard deviation for {variable}")
        value = std_by_variable[variable]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid training standard deviation for {variable}")

    case_records = []
    model_metadata = {}
    for model_id, group in groups.items():
        metadata = dict(group.attrs)
        if metadata.get("model_id") != model_id:
            raise ValueError(
                f"Cache model_id {metadata.get('model_id')!r} does not match "
                f"requested model {model_id!r}"
            )
        if metadata.get("dataset") != dataset or metadata.get("split") != split:
            raise ValueError("All caches must use the same dataset and split")
        if tuple(metadata.get("variable_names", ())) != variable_names:
            raise ValueError("All caches must use the same variable names and order")
        if tuple(metadata.get("variable_units", ())) != variable_units:
            raise ValueError("All caches must use the same variable units")
        if group["prediction"].shape != reference_shape:
            raise ValueError("All caches must have identical prediction shapes")
        for coordinate_name in ("latitude", "longitude", "lead_hours"):
            if not np.array_equal(
                group[coordinate_name][:], reference[coordinate_name][:]
            ):
                raise ValueError(
                    f"Cache coordinate {coordinate_name!r} differs between models"
                )
        if not np.array_equal(
            group["initialization_time_unix_s"][:], initializations
        ) or not np.array_equal(group["valid_time_unix_s"][:], valid_times):
            raise ValueError("All caches must have identical forecast timestamps")
        if bool(group["boundary_mask"][y_index, x_index]):
            raise ValueError(
                f"Nearest grid cell is boundary-forced in the {model_id} cache"
            )
        if not np.isclose(
            float(group["land_sea_mask"][y_index, x_index]),
            common_land_sea_fraction,
        ):
            raise ValueError("Land-sea mask differs between model caches")

        predictions = np.asarray(
            group["prediction"][:, :, y_index, x_index, :], dtype=np.float64
        )
        targets = np.asarray(
            group["target"][:, :, y_index, x_index, :], dtype=np.float64
        )
        latest = np.asarray(
            group["latest_observed_state"][:, y_index, x_index, :],
            dtype=np.float64,
        )
        if not (
            np.isfinite(predictions[:, :, selected_indices]).all()
            and np.isfinite(targets[:, :, selected_indices]).all()
            and np.isfinite(latest[:, selected_indices]).all()
        ):
            raise ValueError(f"Non-finite values found at the selected point for {model_id}")

        model_metadata[model_id] = {
            "checkpoint_id": metadata.get("checkpoint_id"),
            "graph": metadata.get("graph"),
        }
        for variable, variable_index in zip(selected_variables, selected_indices):
            if (
                variable == "sst"
                and common_land_sea_fraction >= OCEAN_LAND_FRACTION_THRESHOLD
            ):
                continue
            prediction = predictions[:, :, variable_index]
            target = targets[:, :, variable_index]
            persistence = np.broadcast_to(
                latest[:, variable_index, None], target.shape
            )
            error = prediction - target
            persistence_error = persistence - target
            unit = variable_units[variable_names.index(variable)]
            for init_index, initialization_time in enumerate(initializations):
                for lead_index, lead in enumerate(lead_hours):
                    valid_time = valid_times[init_index, lead_index]
                    case_records.append({
                        "model_id": model_id,
                        "checkpoint_id": metadata.get("checkpoint_id"),
                        "dataset": dataset,
                        "split": split,
                        "target_latitude": float(target_latitude),
                        "target_longitude": float(target_longitude),
                        "grid_y_index": y_index,
                        "grid_x_index": x_index,
                        "cell_latitude": cell_lat,
                        "cell_longitude": cell_lon,
                        "distance_to_target_km": distance_km,
                        "land_sea_fraction": common_land_sea_fraction,
                        "initialization_time_unix_s": int(initialization_time),
                        "initialization_utc": _utc_string(initialization_time),
                        "valid_time_unix_s": int(valid_time),
                        "valid_time_utc": _utc_string(valid_time),
                        "lead_hours": int(lead),
                        "variable": variable,
                        "unit": unit,
                        "prediction": float(prediction[init_index, lead_index]),
                        "target": float(target[init_index, lead_index]),
                        "error": float(error[init_index, lead_index]),
                        "absolute_error": float(abs(error[init_index, lead_index])),
                        "squared_error": float(error[init_index, lead_index] ** 2),
                        "persistence_prediction": float(
                            persistence[init_index, lead_index]
                        ),
                        "persistence_absolute_error": float(
                            abs(persistence_error[init_index, lead_index])
                        ),
                        "persistence_squared_error": float(
                            persistence_error[init_index, lead_index] ** 2
                        ),
                        "normalized_absolute_error": float(
                            abs(error[init_index, lead_index])
                            / std_by_variable[variable]
                        ),
                    })

    case_frame, summary_frame = _point_summary(case_records, std_by_variable)
    location = {
        "target_latitude": float(target_latitude),
        "target_longitude": float(target_longitude),
        "grid_y_index": y_index,
        "grid_x_index": x_index,
        "cell_latitude": cell_lat,
        "cell_longitude": cell_lon,
        "distance_to_target_km": distance_km,
        "land_sea_fraction": common_land_sea_fraction,
        "land_sea_class": (
            "land"
            if common_land_sea_fraction >= OCEAN_LAND_FRACTION_THRESHOLD
            else "sea"
        ),
        "dataset": dataset,
        "split": split,
        "lead_hours": lead_hours.tolist(),
        "models": model_metadata,
    }
    return case_frame, summary_frame, location


def plot_point_comparison(summary_frame, variables, output_dir, location):
    """Create one four-panel model-comparison figure for each variable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = summary_frame["model_id"].drop_duplicates().tolist()

    for variable in variables:
        variable_rows = summary_frame[summary_frame["variable"] == variable]
        if variable_rows.empty:
            raise ValueError(f"No point metrics available for variable {variable!r}")
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for axis, (metric, label) in zip(axes.flat, PLOT_METRICS):
            for model_id in model_ids:
                model_rows = variable_rows[variable_rows["model_id"] == model_id]
                axis.plot(
                    model_rows["lead_hours"],
                    model_rows[metric],
                    marker="o",
                    linewidth=1.8,
                    label=model_id,
                )
            axis.set_title(label)
            axis.set_xlabel("Forecast lead (hours)")
            axis.set_ylabel(
                "Unitless normalized error"
                if metric in {"normalized_mae", "normalized_rmse"}
                else "Skill (1 − model error / persistence error)"
            )
            axis.grid(True, alpha=0.3)
        axes.flat[0].legend()
        variable_title = {
            "wind_speed_10m": "10 m wind speed",
            "wind_direction_10m": "10 m wind direction",
        }.get(variable, variable)
        figure.suptitle(
            f"{variable_title} at nearest cell to "
            f"{location['target_latitude']:.4f}°N, "
            f"{location['target_longitude']:.4f}°E "
            f"(cell {location['cell_latitude']:.2f}°, "
            f"{location['cell_longitude']:.2f}°; "
            f"{location['distance_to_target_km']:.1f} km away)"
        )
        figure.tight_layout()
        stem = f"{variable}_point_comparison"
        figure.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
        figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)


def plot_wind_vector_comparison(summary_frame, cases, output_dir, calm_threshold):
    """Plot normalized errors and persistence skill for speed and direction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = summary_frame["model_id"].drop_duplicates().tolist()
    variable_columns = (
        ("wind_speed_10m", "10 m wind speed"),
        ("wind_direction_10m", "10 m wind direction"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    for column, (variable, variable_title) in enumerate(variable_columns):
        variable_rows = summary_frame[summary_frame["variable"] == variable]
        for row, (metric, metric_title) in enumerate(WIND_VECTOR_METRICS):
            axis = axes[row, column]
            for model_id in model_ids:
                model_rows = variable_rows[variable_rows["model_id"] == model_id]
                axis.plot(
                    model_rows["lead_hours"],
                    model_rows[metric],
                    marker="o",
                    linewidth=1.8,
                    label=model_id,
                )
            axis.set_title(f"{variable_title}: {metric_title}")
            axis.set_ylabel(
                "Unitless normalized error"
                if metric in {"normalized_mae", "normalized_rmse"}
                else "Skill (1 − model error / persistence error)"
            )
            axis.set_xlabel("Forecast lead (hours)")
            axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
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
            "Extract per-case forecast scores at the grid cell nearest a location "
            "and plot model comparisons."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default="evaluation_results/predictions",
        help="Directory containing completed per-model .zarr caches",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="Cache split to compare (default: test)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_IDS,
        help="Expected model_id values in the cache metadata",
    )
    parser.add_argument("--latitude", type=float, default=58.3405)
    parser.add_argument("--longitude", type=float, default=8.5934)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variables to extract (default: all variables in the cache)",
    )
    parser.add_argument(
        "--plot-variables",
        nargs="+",
        default=None,
        help=(
            "Variables to plot (default: available surface variables among "
            "t2m, d2m, tp, msl, sst; wind speed/direction are plotted when "
            "both u10 and v10 are extracted)"
        ),
    )
    parser.add_argument(
        "--calm-threshold",
        type=float,
        default=0.5,
        help=(
            "Observed wind speed threshold in m/s for direction scoring "
            "(default: 0.5)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/point_comparisons/grimstad/test",
        help="Directory for CSV tables and comparison figures",
    )
    args = parser.parse_args(argv)

    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    cache_paths = _find_model_caches(args.cache_dir, args.models, args.split)
    case_frame, summary_frame, location = extract_point_results(
        cache_paths,
        target_latitude=args.latitude,
        target_longitude=args.longitude,
        split=args.split,
        variables=args.variables,
    )
    available_variables = tuple(summary_frame["variable"].drop_duplicates())
    plot_variables = args.plot_variables
    if plot_variables is None:
        plot_variables = [
            variable for variable in DEFAULT_PLOT_VARIABLES
            if variable in available_variables
        ]
    missing_plot_variables = set(plot_variables) - set(available_variables)
    if missing_plot_variables:
        parser.error(
            "Plot variables are not present in extracted results: "
            + ", ".join(sorted(missing_plot_variables))
            + "; include them with --variables or remove them from --plot-variables"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_summary, pairwise_summary, pairwise_overall = (
        summarize_model_comparisons(summary_frame)
    )
    location["normalization"] = (
        "training-split standard deviation; SST uses interior "
        "ocean-dominant training cells"
    )
    case_path = output_dir / "grimstad_point_cases.csv"
    summary_path = output_dir / "grimstad_point_metrics.csv"
    metadata_path = output_dir / "grimstad_point_metadata.json"
    case_frame.to_csv(case_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    macro_path = output_dir / "grimstad_point_macro_metrics.csv"
    pairwise_path = output_dir / "grimstad_point_pairwise_metrics.csv"
    pairwise_overall_path = output_dir / "grimstad_point_pairwise_overall.csv"
    macro_summary.to_csv(macro_path, index=False)
    pairwise_summary.to_csv(pairwise_path, index=False)
    pairwise_overall.to_csv(pairwise_overall_path, index=False)
    pd.Series(location).to_json(metadata_path, indent=2)
    plot_point_comparison(
        summary_frame,
        plot_variables,
        output_dir / "figures",
        location,
    )
    plot_macro_normalized_mae(macro_summary, output_dir / "figures")
    if {"u10", "v10"}.issubset(set(case_frame["variable"])):
        datasets = case_frame["dataset"].drop_duplicates().tolist()
        if len(datasets) != 1:
            parser.error("Point cases must contain exactly one dataset for wind metrics")
        wind_speed_std = compute_derived_training_statistics(
            datasets[0]
        )["wind_speed_10m_interior"]["std"]
        wind_cases = derive_wind_vector_cases(
            case_frame,
            calm_threshold=args.calm_threshold,
            wind_speed_training_std=wind_speed_std,
        )
        wind_summary = summarize_wind_vector_cases(wind_cases)
        wind_cases_path = output_dir / "grimstad_wind_vector_cases.csv"
        wind_summary_path = output_dir / "grimstad_wind_vector_metrics.csv"
        wind_cases.to_csv(wind_cases_path, index=False)
        wind_summary.to_csv(wind_summary_path, index=False)
        wind_plot_paths = plot_wind_vector_comparison(
            wind_summary,
            wind_cases,
            output_dir / "figures",
            args.calm_threshold,
        )
        print(f"Wrote {len(wind_cases)} wind-vector cases to {wind_cases_path}")
        print(f"Wrote wind-vector metrics to {wind_summary_path}")
        print(f"Wrote wind-vector plots to {wind_plot_paths[0]} and {wind_plot_paths[1]}")
    elif {"u10", "v10"}.intersection(set(case_frame["variable"])):
        print(
            "Skipped wind-vector metrics: include both u10 and v10 with "
            "--variables to derive wind speed and direction."
        )
    print(f"Selected cell: ({location['cell_latitude']:.4f}, "
          f"{location['cell_longitude']:.4f}), "
          f"{location['distance_to_target_km']:.2f} km from target")
    print(f"Wrote {len(case_frame)} per-case records to {case_path}")
    print(f"Wrote {len(summary_frame)} summary records to {summary_path}")
    print(f"Wrote macro normalized metrics to {macro_path}")
    print(f"Wrote per-variable pairwise comparisons to {pairwise_path}")
    print(f"Wrote overall pairwise comparisons to {pairwise_overall_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote comparison figures to {output_dir / 'figures'}")


if __name__ == "__main__":
    main()

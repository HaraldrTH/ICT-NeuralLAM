"""Recompute evaluation metrics from a completed forecast cache."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from neural_lam import constants, utils
from neural_lam.evaluation.accumulators import StreamingMetricAccumulator
from neural_lam.evaluation.config import SUPPORTED_METRICS, load_evaluation_config
from neural_lam.evaluation.forecast_cache import open_forecast_cache
from neural_lam.evaluation.extreme_events import (
    ExtremeEventAccumulator,
    compute_training_event_thresholds,
)
from neural_lam.evaluation.metrics import VARIABLE_GROUPS, derive_10m_wind_speed
from neural_lam.evaluation.normalization import (
    OCEAN_LAND_FRACTION_THRESHOLD,
    compute_derived_training_statistics,
)
from neural_lam.evaluation.results_io import write_results, write_run_metadata
from neural_lam.evaluation.spatial_regions import (
    SpatialMetricAccumulator,
    build_configured_region_masks,
    build_spatial_region_masks,
)


def _validate_cache(group):
    required_arrays = (
        "prediction",
        "target",
        "latest_observed_state",
        "initialization",
        "initialization_time_unix_s",
        "valid_time_unix_s",
        "lead_hours",
        "latitude",
        "longitude",
        "land_sea_mask",
        "boundary_mask",
    )
    missing = [name for name in required_arrays if name not in group]
    if missing:
        raise ValueError(f"Forecast cache is missing arrays: {', '.join(missing)}")

    prediction = group["prediction"]
    target = group["target"]
    initial_state = group["latest_observed_state"]
    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("prediction and target must match (init, lead, y, x, variable)")
    if prediction.dtype != np.dtype("float32") or target.dtype != np.dtype("float32"):
        raise ValueError("prediction and target arrays must use float32")
    n_initializations, n_leads, ny, nx, n_variables = prediction.shape
    expected_dimensions = ["initialization", "lead", "y", "x", "variable"]
    if prediction.attrs.get("_ARRAY_DIMENSIONS") != expected_dimensions:
        raise ValueError("prediction dimension metadata is invalid")
    if target.attrs.get("_ARRAY_DIMENSIONS") != expected_dimensions:
        raise ValueError("target dimension metadata is invalid")
    if initial_state.shape != (n_initializations, ny, nx, n_variables):
        raise ValueError("latest_observed_state shape does not match forecast arrays")
    if initial_state.dtype != np.dtype("float32"):
        raise ValueError("latest_observed_state must use float32")
    if initial_state.attrs.get("_ARRAY_DIMENSIONS") != [
        "initialization", "y", "x", "variable"
    ]:
        raise ValueError("latest_observed_state dimension metadata is invalid")

    variable_names = tuple(group.attrs.get("variable_names", ()))
    variable_units = tuple(group.attrs.get("variable_units", ()))
    if len(variable_names) != n_variables or len(variable_units) != n_variables:
        raise ValueError("Cache variable names/units do not match array dimension")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Cache variable names must be unique")
    if tuple(prediction.attrs.get("units_by_variable", ())) != variable_units:
        raise ValueError("Prediction unit metadata does not match cache metadata")
    if tuple(target.attrs.get("units_by_variable", ())) != variable_units:
        raise ValueError("Target unit metadata does not match cache metadata")
    try:
        variable_indices = [constants.param_names.index(name) for name in variable_names]
    except ValueError as exc:
        raise ValueError("Cache contains a variable unknown to project constants") from exc
    if variable_indices != sorted(variable_indices):
        raise ValueError("Cache variable order does not match project constants")
    expected_units = tuple(constants.param_units[index] for index in variable_indices)
    if variable_units != expected_units:
        raise ValueError("Cache variable units do not match project constants")
    if group.attrs.get("prediction_units") != "physical":
        raise ValueError("Cache predictions are not marked as physical-unit values")
    if not group.attrs.get("model_id") or "checkpoint_id" not in group.attrs:
        raise ValueError("Cache provenance is missing model/checkpoint identifiers")
    if not group.attrs.get("boundary_cells_use_reference_truth", False):
        raise ValueError("Cache boundary-forcing provenance is missing")

    leads = np.asarray(group["lead_hours"][:], dtype=np.int64)
    if leads.shape != (n_leads,) or np.any(leads <= 0):
        raise ValueError("lead_hours does not match forecast lead dimension")
    if group["lead_hours"].attrs.get("_ARRAY_DIMENSIONS") != ["lead"]:
        raise ValueError("lead_hours dimension metadata is invalid")
    if np.any(np.diff(leads) <= 0):
        raise ValueError("lead_hours must be strictly increasing")

    initializations = np.asarray(
        group["initialization_time_unix_s"][:], dtype=np.int64
    )
    valid_times = np.asarray(group["valid_time_unix_s"][:], dtype=np.int64)
    if initializations.shape != (n_initializations,):
        raise ValueError("Initialization timestamps do not match forecast array")
    if valid_times.shape != (n_initializations, n_leads):
        raise ValueError("Valid timestamps do not match forecast dimensions")
    if group["initialization_time_unix_s"].attrs.get("_ARRAY_DIMENSIONS") != [
        "initialization"
    ]:
        raise ValueError("Initialization timestamp dimension metadata is invalid")
    if group["valid_time_unix_s"].attrs.get("_ARRAY_DIMENSIONS") != [
        "initialization", "lead"
    ]:
        raise ValueError("Valid timestamp dimension metadata is invalid")
    if np.any(np.diff(initializations) <= 0):
        raise ValueError("Initialization timestamps must be strictly increasing")
    if not np.array_equal(
        valid_times, initializations[:, None] + leads[None, :] * 3600
    ):
        raise ValueError("Valid timestamps are inconsistent with initialization and leads")
    if not np.array_equal(
        np.asarray(group["initialization"][:]),
        np.arange(n_initializations),
    ):
        raise ValueError("Initialization coordinate is not a contiguous index")

    if group["latitude"].shape != (ny,) or group["longitude"].shape != (nx,):
        raise ValueError("Latitude/longitude coordinate shapes do not match grid")
    expected_grid_dimensions = {
        "latitude": ["y"],
        "longitude": ["x"],
        "land_sea_mask": ["y", "x"],
        "boundary_mask": ["y", "x"],
    }
    for array_name, dimensions in expected_grid_dimensions.items():
        if group[array_name].attrs.get("_ARRAY_DIMENSIONS") != dimensions:
            raise ValueError(f"{array_name} dimension metadata is invalid")
    if group["land_sea_mask"].shape != (ny, nx):
        raise ValueError("land_sea_mask shape does not match forecast grid")
    if group["boundary_mask"].shape != (ny, nx):
        raise ValueError("boundary_mask shape does not match forecast grid")
    if not np.isfinite(group["land_sea_mask"][:]).all():
        raise ValueError("land_sea_mask contains non-finite values")

    return {
        "n_initializations": n_initializations,
        "n_leads": n_leads,
        "grid_shape": (ny, nx),
        "variable_names": variable_names,
        "variable_units": variable_units,
        "lead_hours": leads,
    }


def evaluate_forecast_cache(
    cache_path,
    *,
    training_std: Optional[Sequence[float]] = None,
    lead_hours: Optional[Sequence[int]] = None,
    variables: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    variable_groups=None,
    lead_groups=None,
    exclude_land_sst=True,
    spatial_regions=None,
    extreme_events=None,
):
    """Compute configured metrics from cached physical-unit arrays.

    Reads one initialization at a time to keep memory use bounded. Normalized
    metrics use the dataset's saved training standard deviations unless
    ``training_std`` is explicitly supplied.
    """
    group = open_forecast_cache(cache_path)
    cache_info = _validate_cache(group)
    variable_names = cache_info["variable_names"]
    variable_units = cache_info["variable_units"]
    cache_leads = cache_info["lead_hours"]

    requested_leads = tuple(
        int(value) for value in (lead_hours if lead_hours is not None else cache_leads)
    )
    if not requested_leads or len(set(requested_leads)) != len(requested_leads):
        raise ValueError("Requested leads must be non-empty and unique")
    missing_leads = set(requested_leads) - set(cache_leads.tolist())
    if missing_leads:
        raise ValueError(
            "Requested lead hours are absent from cache: "
            + ", ".join(str(value) for value in sorted(missing_leads))
        )
    lead_indices = np.asarray(
        [int(np.flatnonzero(cache_leads == lead)[0]) for lead in requested_leads],
        dtype=np.int64,
    )

    requested_variables = tuple(variables) if variables is not None else variable_names
    if not requested_variables or len(set(requested_variables)) != len(requested_variables):
        raise ValueError("Requested variables must be non-empty and unique")
    missing_variables = set(requested_variables) - set(variable_names)
    if missing_variables:
        raise ValueError(
            "Requested variables are absent from cache: "
            + ", ".join(sorted(missing_variables))
        )
    variable_indices = np.asarray(
        [variable_names.index(name) for name in requested_variables],
        dtype=np.int64,
    )
    selected_units = tuple(variable_units[index] for index in variable_indices)

    if training_std is None:
        dataset_name = group.attrs.get("dataset")
        if not dataset_name:
            raise ValueError("Cache metadata is missing dataset name for training stats")
        training_std = utils.load_dataset_stats(dataset_name, "cpu")["data_std"]
        if hasattr(training_std, "detach"):
            training_std = training_std.detach().cpu().numpy()
    training_std = np.asarray(training_std, dtype=np.float64).copy()
    if training_std.shape != (len(variable_names),):
        raise ValueError("Training standard deviations must cover all cached variables")
    if not np.isfinite(training_std).all() or np.any(training_std <= 0):
        raise ValueError("Training standard deviations must be finite and positive")
    normalization_audit = None
    has_sst = "sst" in requested_variables
    has_wind_components = {"u10", "v10"}.issubset(requested_variables)
    if (exclude_land_sst and has_sst) or has_wind_components:
        normalization_audit = compute_derived_training_statistics(
            str(group.attrs["dataset"])
        )
    if exclude_land_sst and has_sst:
        sst_index = requested_variables.index("sst")
        training_std[variable_indices[sst_index]] = normalization_audit[
            "sst_ocean_dominant_interior"
        ]["std"]
    selected_std = training_std[variable_indices]

    if metrics is not None:
        requested_metrics = tuple(metrics)
        unsupported = set(requested_metrics) - set(SUPPORTED_METRICS)
        if not requested_metrics or unsupported:
            raise ValueError(
                "Unsupported or empty metric selection: "
                + ", ".join(sorted(unsupported))
            )
    else:
        requested_metrics = (
            "mae",
            "rmse",
            "normalized_mae",
            "normalized_rmse",
            "persistence_mae",
            "persistence_rmse",
            "mae_skill_vs_persistence",
            "rmse_skill_vs_persistence",
        )

    accumulator = StreamingMetricAccumulator(
        requested_variables,
        requested_leads,
        training_std=selected_std,
        variable_units=selected_units,
    )
    boundary_mask = np.asarray(group["boundary_mask"][:], dtype=bool).reshape(-1)
    grid_shape = cache_info["grid_shape"]
    valid_mask = np.broadcast_to(
        (~boundary_mask).reshape(1, 1, -1, 1),
        (1, 1, boundary_mask.size, len(requested_variables)),
    ).copy()
    land_sea_mask = np.asarray(group["land_sea_mask"][:], dtype=np.float32)
    if exclude_land_sst and has_sst:
        ocean_dominant = (
            land_sea_mask < OCEAN_LAND_FRACTION_THRESHOLD
        ).reshape(-1)
        valid_mask[..., requested_variables.index("sst")] &= ocean_dominant.reshape(
            1, 1, -1
        )

    spatial_regions = spatial_regions or {"enabled": False}
    extreme_events = extreme_events or {"enabled": False}
    spatial_enabled = spatial_regions.get("enabled", False)
    events_enabled = extreme_events.get("enabled", False)
    if type(spatial_enabled) is not bool or type(events_enabled) is not bool:
        raise ValueError("Spatial and extreme-event enabled settings must be booleans")
    if spatial_enabled and spatial_regions.get("scope", "full_grid") != "full_grid":
        raise ValueError("Spatial evaluation currently supports only the full_grid scope")
    if events_enabled:
        if extreme_events.get("precipitation", False):
            raise ValueError(
                "Precipitation event evaluation is deferred until the tp "
                "accumulation interval has been verified"
            )
        lower_percentile = extreme_events.get("lower_percentile", 1.0)
        upper_percentile = extreme_events.get("upper_percentile", 99.0)
        if (
            isinstance(lower_percentile, bool)
            or isinstance(upper_percentile, bool)
            or not isinstance(lower_percentile, (int, float))
            or not isinstance(upper_percentile, (int, float))
            or not np.isfinite(lower_percentile)
            or not np.isfinite(upper_percentile)
            or not 0 < lower_percentile < 50 < upper_percentile < 100
        ):
            raise ValueError(
                "Extreme-event percentiles must satisfy "
                "0 < lower_percentile < 50 < upper_percentile < 100"
            )

    region_masks = None
    latitude = np.asarray(group["latitude"][:], dtype=np.float64)
    longitude = np.asarray(group["longitude"][:], dtype=np.float64)
    boundary_grid = np.asarray(group["boundary_mask"][:], dtype=bool)
    if spatial_enabled:
        region_definitions = spatial_regions.get("regions", {})
        if region_definitions:
            region_masks = build_configured_region_masks(
                latitude,
                longitude,
                land_sea_mask,
                region_definitions,
            )
        else:
            region_masks = build_spatial_region_masks(
                latitude,
                longitude,
                land_sea_mask,
                coastal_distance_km=spatial_regions.get(
                    "coastal_distance_km", 25.0
                ),
            )
    elif events_enabled:
        region_masks = {"all_grid": np.ones(grid_shape, dtype=bool)}

    spatial_accumulator = None
    if spatial_enabled:
        spatial_names = list(requested_variables)
        spatial_units = list(selected_units)
        spatial_std = list(selected_std)
        if has_wind_components:
            spatial_names.append("wind_speed_10m")
            spatial_units.append("m/s")
            spatial_std.append(
                normalization_audit["wind_speed_10m_interior"]["std"]
            )
        spatial_accumulator = SpatialMetricAccumulator(
            region_masks,
            latitude,
            boundary_grid,
            spatial_names,
            spatial_units,
            spatial_std,
            land_sea_mask=land_sea_mask,
            exclude_land_sst=exclude_land_sst,
        )

    event_accumulator = None
    event_threshold_info = None
    event_source_indices = {}
    if events_enabled:
        event_source_names = {
            "t2m": ("t2m",),
            "msl": ("msl",),
            "wind_speed_10m": ("u10", "v10"),
        }
        missing_event_variables = sorted({
            name
            for source_names in event_source_names.values()
            for name in source_names
            if name not in variable_names
        })
        if missing_event_variables:
            raise ValueError(
                "Cache is missing variables required for extreme-event analysis: "
                + ", ".join(missing_event_variables)
            )
        event_source_indices = {
            name: variable_names.index(name)
            for source_names in event_source_names.values()
            for name in source_names
        }
        lower_percentile = extreme_events.get("lower_percentile", 1.0)
        upper_percentile = extreme_events.get("upper_percentile", 99.0)
        event_threshold_info = compute_training_event_thresholds(
            str(group.attrs["dataset"]),
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        event_accumulator = ExtremeEventAccumulator(
            region_masks,
            latitude,
            boundary_grid,
            event_threshold_info["thresholds"],
        )

    wind_accumulator = None
    wind_component_indices = None
    if has_wind_components:
        wind_component_indices = (
            requested_variables.index("u10"),
            requested_variables.index("v10"),
        )
        wind_accumulator = StreamingMetricAccumulator(
            ["wind_speed_10m"],
            requested_leads,
            training_std=[
                normalization_audit["wind_speed_10m_interior"]["std"]
            ],
            variable_units=["m/s"],
        )

    for initialization_index in range(cache_info["n_initializations"]):
        prediction_all = np.asarray(
            group["prediction"][initialization_index:initialization_index + 1,
                                lead_indices, :, :, :],
            dtype=np.float32,
        )
        target_all = np.asarray(
            group["target"][initialization_index:initialization_index + 1,
                            lead_indices, :, :, :],
            dtype=np.float32,
        )
        prediction = prediction_all[..., variable_indices]
        target = target_all[..., variable_indices]
        latest = np.asarray(
            group["latest_observed_state"][initialization_index:
                                           initialization_index + 1],
            dtype=np.float32,
        )[..., variable_indices]

        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite prediction at initialization index {initialization_index}")
        if not np.isfinite(target).all():
            raise ValueError(f"Non-finite target at initialization index {initialization_index}")
        if not np.isfinite(latest).all():
            raise ValueError(
                f"Non-finite latest observed state at initialization index "
                f"{initialization_index}"
            )

        if spatial_accumulator is not None:
            spatial_prediction = prediction
            spatial_target = target
            if has_wind_components:
                spatial_prediction = np.concatenate(
                    (
                        prediction,
                        derive_10m_wind_speed(
                            prediction, requested_variables
                        )[..., None],
                    ),
                    axis=-1,
                )
                spatial_target = np.concatenate(
                    (
                        target,
                        derive_10m_wind_speed(
                            target, requested_variables
                        )[..., None],
                    ),
                    axis=-1,
                )
            spatial_accumulator.update(spatial_prediction, spatial_target)

        if event_accumulator is not None:
            event_prediction = {
                "t2m": prediction_all[..., event_source_indices["t2m"]],
                "msl": prediction_all[..., event_source_indices["msl"]],
                "wind_speed_10m": np.hypot(
                    prediction_all[..., event_source_indices["u10"]],
                    prediction_all[..., event_source_indices["v10"]],
                ),
            }
            event_target = {
                "t2m": target_all[..., event_source_indices["t2m"]],
                "msl": target_all[..., event_source_indices["msl"]],
                "wind_speed_10m": np.hypot(
                    target_all[..., event_source_indices["u10"]],
                    target_all[..., event_source_indices["v10"]],
                ),
            }
            event_accumulator.update(event_prediction, event_target)

        batch, n_selected_leads, ny, nx, n_selected_variables = prediction.shape
        prediction = prediction.reshape(
            batch, n_selected_leads, ny * nx, n_selected_variables
        )
        target = target.reshape(batch, n_selected_leads, ny * nx, n_selected_variables)
        latest = latest.reshape(batch, ny * nx, n_selected_variables)
        persistence = np.broadcast_to(
            latest[:, None, :, :], prediction.shape
        )
        accumulator.update(
            prediction,
            target,
            persistence_prediction=persistence,
            valid_mask=valid_mask,
        )
        if wind_accumulator is not None:
            wind_prediction = derive_10m_wind_speed(
                prediction, requested_variables
            )[..., None]
            wind_target = derive_10m_wind_speed(
                target, requested_variables
            )[..., None]
            wind_latest = derive_10m_wind_speed(
                latest, requested_variables
            )[..., None]
            wind_accumulator.update(
                wind_prediction,
                wind_target,
                persistence_prediction=np.broadcast_to(
                    wind_latest[:, None, :, :], wind_prediction.shape
                ),
                valid_mask=(~boundary_mask).reshape(1, 1, -1, 1),
            )

    selected_metrics = set(accumulator.compute())
    unavailable = set(requested_metrics) - selected_metrics
    if unavailable:
        raise ValueError(
            "Requested metrics are unavailable: " + ", ".join(sorted(unavailable))
        )
    records = accumulator.to_records(
        model_id=str(group.attrs["model_id"]),
        checkpoint_id=group.attrs.get("checkpoint_id"),
        metadata={
            "dataset": group.attrs.get("dataset"),
            "split": group.attrs.get("split"),
            "graph": group.attrs.get("graph"),
        },
        metric_names=requested_metrics,
    )
    model_id = str(group.attrs["model_id"])
    checkpoint_id = group.attrs.get("checkpoint_id")
    record_metadata = {
        "dataset": group.attrs.get("dataset"),
        "split": group.attrs.get("split"),
        "graph": group.attrs.get("graph"),
    }

    if wind_accumulator is not None:
        records.extend(wind_accumulator.to_records(
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            metadata={
                **record_metadata,
                "derived_from": "u10,v10",
                "normalization_std": normalization_audit[
                    "wind_speed_10m_interior"
                ]["std"],
            },
            metric_names=requested_metrics,
        ))

    if spatial_accumulator is not None:
        records.extend(spatial_accumulator.to_records(
            requested_leads,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            metadata=record_metadata,
        ))

    if event_accumulator is not None:
        records.extend(event_accumulator.to_records(
            requested_leads,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            metadata=record_metadata,
        ))

    variable_groups = VARIABLE_GROUPS if variable_groups is None else variable_groups
    metric_values = accumulator.compute()
    for group_name, members in variable_groups.items():
        member_indices = [
            requested_variables.index(name)
            for name in members
            if name in requested_variables
        ]
        if not member_indices:
            continue
        for metric_name in ("normalized_mae", "normalized_rmse"):
            if metric_name not in requested_metrics:
                continue
            aggregate = np.nanmean(metric_values[metric_name][:, member_indices], axis=1)
            for lead_index, lead in enumerate(requested_leads):
                sample_count = int(
                    accumulator._sample_count[lead_index, member_indices].sum()
                )
                weight_sum = float(
                    accumulator._weight_sum[lead_index, member_indices].sum()
                )
                records.append({
                    "model_id": model_id,
                    "checkpoint_id": checkpoint_id,
                    "lead_hours": lead,
                    "variable": f"group:{group_name}",
                    "unit": "1",
                    "metric": f"group_{metric_name}",
                    "value": (
                        float(aggregate[lead_index])
                        if np.isfinite(aggregate[lead_index]) else None
                    ),
                    "sample_count": sample_count,
                    "weight_sum": weight_sum,
                    **record_metadata,
                    "variable_group": group_name,
                    "group_variables": [
                        requested_variables[index] for index in member_indices
                    ],
                    "aggregation": "unweighted mean of per-variable normalized metrics",
                })

    lead_groups = lead_groups or {}
    for group_name, group_leads in lead_groups.items():
        member_leads = [
            lead for lead in group_leads if lead in requested_leads
        ]
        if not member_leads:
            continue
        member_indices = [requested_leads.index(lead) for lead in member_leads]
        for metric_name in requested_metrics:
            values = metric_values[metric_name]
            grouped_values = np.nanmean(values[member_indices, :], axis=0)
            for variable_index, variable_name in enumerate(requested_variables):
                sample_count = int(
                    accumulator._sample_count[
                        member_indices, variable_index
                    ].min()
                )
                weight_sum = float(
                    accumulator._weight_sum[
                        member_indices, variable_index
                    ].min()
                )
                value = grouped_values[variable_index]
                records.append({
                    "model_id": model_id,
                    "checkpoint_id": checkpoint_id,
                    "lead_hours": None,
                    "variable": variable_name,
                    "unit": (
                        variable_units[variable_indices[variable_index]]
                        if metric_name in {"mae", "rmse", "persistence_mae",
                                           "persistence_rmse"}
                        else "1"
                    ),
                    "metric": f"lead_group_mean_{metric_name}",
                    "value": float(value) if np.isfinite(value) else None,
                    "sample_count": sample_count,
                    "weight_sum": weight_sum,
                    **record_metadata,
                    "lead_group": group_name,
                    "lead_hours_included": member_leads,
                    "aggregation": "unweighted mean across configured lead times",
                })
    return records, {
        "cache_path": str(cache_path),
        "cache_metadata": dict(group.attrs),
        "n_initializations": cache_info["n_initializations"],
        "n_leads": len(requested_leads),
        "lead_hours": list(requested_leads),
        "variables": list(requested_variables),
        "metrics": list(requested_metrics),
        "variable_groups": {
            name: list(members) for name, members in variable_groups.items()
        },
        "lead_groups": {
            name: [lead for lead in group_leads if lead in requested_leads]
            for name, group_leads in lead_groups.items()
            if any(lead in requested_leads for lead in group_leads)
        },
        "exclude_land_sst": bool(exclude_land_sst),
        "normalization_audit": normalization_audit,
        "spatial_regions": {
            "enabled": spatial_enabled,
            "scope": spatial_regions.get("scope", "full_grid"),
            "coastal_distance_km": spatial_regions.get(
                "coastal_distance_km", 25.0
            ),
            "region_names": list(region_masks) if spatial_enabled else [],
            "region_definitions": spatial_regions.get("regions", {}),
            "distance_method": (
                "great-circle distance between a cell center and the nearest "
                "opposite land/sea-class cell center"
                if spatial_enabled else None
            ),
            "area_weighting": "cosine of latitude",
        },
        "extreme_events": {
            "enabled": events_enabled,
            "thresholds": event_threshold_info,
            "precipitation_included": False,
            "area_weighting": "cosine of latitude",
        },
        "normalization_policy": {
            "base_variables": (
                "dataset parameter_std; except SST when land-dominant cells are "
                "excluded"
            ),
            "sst": (
                "training-split standard deviation over interior cells with "
                f"land_sea_mask < {OCEAN_LAND_FRACTION_THRESHOLD}; evaluation "
                "uses the same ocean-dominant mask"
                if exclude_land_sst else "dataset parameter_std over all cells"
            ),
            "wind_speed_10m": (
                "training-split standard deviation of hypot(u10, v10) over "
                "interior cells"
            ),
            "group_aggregation": (
                "unweighted mean of normalized variable metrics"
            ),
            "lead_group_aggregation": (
                "unweighted mean across configured lead-time metrics"
            ),
        },
    }


def _slug(value):
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute forecast metrics from a completed Zarr cache."
    )
    parser.add_argument("--cache", required=True, help="Path to completed .zarr cache")
    parser.add_argument(
        "--config",
        required=True,
        help="Evaluation YAML used to select leads, variables, metrics, and output format",
    )
    args = parser.parse_args(argv)

    config = load_evaluation_config(args.config)
    group = open_forecast_cache(args.cache)
    metadata = dict(group.attrs)
    if metadata.get("dataset") != config.dataset or metadata.get("split") != config.split:
        raise ValueError("Evaluation config dataset/split does not match the cache")
    records, run_info = evaluate_forecast_cache(
        args.cache,
        lead_hours=config.lead_hours,
        variables=config.variables,
        metrics=config.metrics,
        variable_groups=config.variable_groups,
        lead_groups=config.lead_groups,
        exclude_land_sst=config.exclude_land_sst,
        spatial_regions=config.spatial_regions,
        extreme_events=config.extreme_events,
    )

    stem = (
        f"{_slug(str(metadata['model_id']))}_{_slug(str(metadata['split']))}_"
        f"{_slug(Path(str(metadata.get('checkpoint_id') or 'uncheckpointed')).stem)}"
        "_offline"
    )
    output_dir = Path(config.output_dir)
    results_path = write_results(
        records,
        output_dir / f"{stem}_metrics.{config.results_format}",
        file_format=config.results_format,
    )
    write_run_metadata(
        {
            **run_info,
            "config_path": str(args.config),
            "results_path": str(results_path),
            "evaluation_mode": "offline_from_forecast_cache",
        },
        output_dir / f"{stem}_metadata.json",
    )
    print(f"Wrote {len(records)} metric records to {results_path}")


if __name__ == "__main__":
    main()

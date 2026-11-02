"""YAML configuration loading for the reusable evaluation foundation."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str
    split: str
    lead_hours: Tuple[int, ...]
    variables: Optional[Tuple[str, ...]]
    metrics: Tuple[str, ...]
    step_length_hours: int
    batch_size: int
    num_workers: int
    output_dir: str
    results_format: str
    save_forecast_cache: bool
    prediction_store_dir: str
    variable_groups: Dict[str, Tuple[str, ...]]
    lead_groups: Dict[str, Tuple[int, ...]]
    exclude_land_sst: bool
    spatial_regions: Dict[str, object]
    extreme_events: Dict[str, object]


SUPPORTED_METRICS = (
    "mae",
    "rmse",
    "normalized_mae",
    "normalized_rmse",
    "persistence_mae",
    "persistence_rmse",
    "mae_skill_vs_persistence",
    "rmse_skill_vs_persistence",
)


def load_evaluation_config(path):
    """Load and validate a YAML evaluation configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Evaluation config must be a YAML mapping")

    required = ("dataset", "split", "lead_hours", "batch_size", "output_dir")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing evaluation config keys: {', '.join(missing)}")
    if data["split"] not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if not isinstance(data["lead_hours"], list) or not data["lead_hours"]:
        raise ValueError("lead_hours must be a non-empty YAML list")
    lead_hours = tuple(data["lead_hours"])
    if any(type(value) is not int or value <= 0 for value in lead_hours):
        raise ValueError("lead_hours entries must be positive integers")
    if len(set(lead_hours)) != len(lead_hours):
        raise ValueError("lead_hours entries must not contain duplicates")
    step_length_hours = data.get("step_length_hours", 6)
    if type(step_length_hours) is not int or step_length_hours <= 0:
        raise ValueError("step_length_hours must be a positive integer")
    if any(value % step_length_hours for value in lead_hours):
        raise ValueError("lead_hours must be multiples of step_length_hours")
    if type(data["batch_size"]) is not int or data["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    num_workers = data.get("num_workers", 0)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    variables = data.get("variables")
    if variables is not None and (not isinstance(variables, list)
                                  or not variables
                                  or any(not isinstance(value, str) or not value for value in variables)):
        raise ValueError("variables must be null or a non-empty list of names")
    metrics = data.get("metrics", ["mae", "rmse"])
    if (not isinstance(metrics, list) or not metrics
        or any(metric not in SUPPORTED_METRICS for metric in metrics)):
        raise ValueError("metrics must be a non-empty list of supported metric names: "
                         + ", ".join(SUPPORTED_METRICS))
    if len(set(metrics)) != len(metrics):
        raise ValueError("metrics entries must not contain duplicates")
    results_format = data.get("results_format", "csv")
    if results_format not in ("csv", "parquet"):
        raise ValueError("results_format must be 'csv' or 'parquet'")
    save_forecast_cache = data.get("save_forecast_cache", False)
    if type(save_forecast_cache) is not bool:
        raise ValueError("save_forecast_cache must be true or false")
    exclude_land_sst = data.get("exclude_land_sst", True)
    if type(exclude_land_sst) is not bool:
        raise ValueError("exclude_land_sst must be true or false")

    spatial_regions = data.get("spatial_regions", {})
    if not isinstance(spatial_regions, dict):
        raise ValueError("spatial_regions must be a YAML mapping")
    spatial_enabled = spatial_regions.get("enabled", False)
    if type(spatial_enabled) is not bool:
        raise ValueError("spatial_regions.enabled must be true or false")
    spatial_scope = spatial_regions.get("scope", "full_grid")
    if spatial_scope != "full_grid":
        raise ValueError("spatial_regions.scope currently supports only 'full_grid'")
    coastal_distance_km = spatial_regions.get("coastal_distance_km", 25.0)
    if (isinstance(coastal_distance_km, bool)
        or not isinstance(coastal_distance_km, (int, float))
        or not math.isfinite(coastal_distance_km)
        or coastal_distance_km <= 0):
        raise ValueError("spatial_regions.coastal_distance_km must be positive")
    raw_regions = spatial_regions.get("regions")
    if raw_regions is not None and (
        not isinstance(raw_regions, dict)
        or not raw_regions
        or any(not isinstance(name, str) or not name for name in raw_regions)
    ):
        raise ValueError("spatial_regions.regions must map names to definitions")
    region_definitions = {}
    region_fields = {
        "polygon_file",
        "bounds",
        "land_sea",
        "min_coast_distance_km",
        "max_coast_distance_km",
    }
    for name, definition in (raw_regions or {}).items():
        if not isinstance(definition, dict):
            raise ValueError(f"Spatial region {name!r} must be a YAML mapping")
        unknown_fields = set(definition) - region_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown fields for spatial region {name!r}: "
                + ", ".join(sorted(unknown_fields))
            )
        polygon_file = definition.get("polygon_file")
        if polygon_file is not None:
            if not isinstance(polygon_file, str) or not polygon_file:
                raise ValueError(
                    f"Spatial region {name!r} polygon_file must be a non-empty path"
                )
            polygon_path = Path(polygon_file)
            if not polygon_path.is_absolute():
                polygon_path = Path(path).resolve().parent / polygon_path
            definition = {
                **definition,
                "polygon_file": str(polygon_path.resolve()),
            }
        bounds = definition.get("bounds")
        if bounds is not None and (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in bounds
            )
        ):
            raise ValueError(
                f"Spatial region {name!r} bounds must be four finite numbers"
            )
        if "polygon_file" in definition and "bounds" in definition:
            raise ValueError(
                f"Spatial region {name!r} cannot specify both polygon_file and bounds"
            )
        land_sea = definition.get("land_sea", "all")
        if land_sea not in ("all", "land", "sea"):
            raise ValueError(
                f"Spatial region {name!r} land_sea must be all, land, or sea"
            )
        minimum_distance = definition.get("min_coast_distance_km")
        maximum_distance = definition.get("max_coast_distance_km")
        for field, value in (
            ("min_coast_distance_km", minimum_distance),
            ("max_coast_distance_km", maximum_distance),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Spatial region {name!r} {field} must be a finite "
                    "non-negative number"
                )
        if (
            minimum_distance is not None or maximum_distance is not None
        ) and land_sea == "all":
            raise ValueError(
                f"Spatial region {name!r} must select land or sea "
                "when using coast-distance filters"
            )
        if (
            minimum_distance is not None
            and maximum_distance is not None
            and minimum_distance >= maximum_distance
        ):
            raise ValueError(
                f"Spatial region {name!r} coast-distance bounds are invalid"
            )
        region_definitions[name] = definition

    extreme_events = data.get("extreme_events", {})
    if not isinstance(extreme_events, dict):
        raise ValueError("extreme_events must be a YAML mapping")
    extreme_enabled = extreme_events.get("enabled", False)
    if type(extreme_enabled) is not bool:
        raise ValueError("extreme_events.enabled must be true or false")
    lower_percentile = extreme_events.get("lower_percentile", 1.0)
    upper_percentile = extreme_events.get("upper_percentile", 99.0)
    for key, value in (
        ("lower_percentile", lower_percentile),
        ("upper_percentile", upper_percentile),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"extreme_events.{key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"extreme_events.{key} must be finite")
    if not 0 < lower_percentile < 50 < upper_percentile < 100:
        raise ValueError(
            "extreme-event percentiles must satisfy "
            "0 < lower_percentile < 50 < upper_percentile < 100"
        )
    include_precipitation = extreme_events.get("precipitation", False)
    if type(include_precipitation) is not bool:
        raise ValueError("extreme_events.precipitation must be true or false")
    if include_precipitation:
        raise ValueError(
            "Precipitation event evaluation is deferred until the tp "
            "accumulation interval has been verified"
        )

    from neural_lam.evaluation.metrics import VARIABLE_GROUPS

    raw_variable_groups = data.get(
        "variable_groups",
        {name: list(members) for name, members in VARIABLE_GROUPS.items()},
    )
    if not isinstance(raw_variable_groups, dict) or any(
        not isinstance(name, str) or not name
        or not isinstance(members, list) or not members
        or any(not isinstance(member, str) or not member for member in members)
        or len(set(members)) != len(members)
        for name, members in raw_variable_groups.items()
    ):
        raise ValueError("variable_groups must map names to non-empty variable lists")

    raw_lead_groups = data.get("lead_groups", {})
    if not isinstance(raw_lead_groups, dict) or any(
        not isinstance(name, str) or not name
        or not isinstance(leads, list) or not leads
        or any(type(lead) is not int or lead <= 0 for lead in leads)
        or len(set(leads)) != len(leads)
        for name, leads in raw_lead_groups.items()
    ):
        raise ValueError("lead_groups must map names to lists of positive unique hours")

    prediction_store_dir = data.get("prediction_store_dir",
                                    "evaluation_results/predictions")
    if not isinstance(prediction_store_dir, str) or not prediction_store_dir:
        raise ValueError("prediction_store_dir must be a non-empty path")

    return EvaluationConfig(
        dataset=data["dataset"],
        split=data["split"],
        lead_hours=lead_hours,
        variables=tuple(variables) if variables is not None else None,
        metrics=tuple(metrics),
        step_length_hours=step_length_hours,
        batch_size=data["batch_size"],
        num_workers=num_workers,
        output_dir=str(data["output_dir"]),
        results_format=results_format,
        save_forecast_cache=save_forecast_cache,
        prediction_store_dir=prediction_store_dir,
        variable_groups={
            name: tuple(members) for name, members in raw_variable_groups.items()
        },
        lead_groups={
            name: tuple(leads) for name, leads in raw_lead_groups.items()
        },
        exclude_land_sst=exclude_land_sst,
        spatial_regions={
            "enabled": spatial_enabled,
            "scope": spatial_scope,
            "coastal_distance_km": float(coastal_distance_km),
            "regions": region_definitions,
        },
        extreme_events={
            "enabled": extreme_enabled,
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "precipitation": False,
        },
    )

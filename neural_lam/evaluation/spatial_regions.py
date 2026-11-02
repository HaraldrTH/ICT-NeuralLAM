"""Geographic region masks and area-weighted regional metrics."""

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy
from shapely.geometry import box, shape
from shapely.ops import unary_union

from neural_lam.evaluation.normalization import OCEAN_LAND_FRACTION_THRESHOLD


EARTH_RADIUS_KM = 6371.0088


def _unit_sphere_coordinates(latitude, longitude):
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    latitude_grid, longitude_grid = np.meshgrid(
        latitude_radians, longitude_radians, indexing="ij"
    )
    return np.column_stack((
        np.cos(latitude_grid).ravel() * np.cos(longitude_grid).ravel(),
        np.cos(latitude_grid).ravel() * np.sin(longitude_grid).ravel(),
        np.sin(latitude_grid).ravel(),
    ))


def build_spatial_region_masks(
    latitude,
    longitude,
    land_sea_mask,
    *,
    coastal_distance_km=25.0,
    land_fraction_threshold=OCEAN_LAND_FRACTION_THRESHOLD,
):
    """Create full-grid land/sea and coastal-band masks.

    Coast distance is the great-circle distance between a cell center and the
    nearest opposite land/sea-class cell center. The land/sea classification
    uses the configured fractional-mask threshold.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    land_sea_mask = np.asarray(land_sea_mask, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional")
    if land_sea_mask.shape != (latitude.size, longitude.size):
        raise ValueError("land_sea_mask shape must match latitude/longitude")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("latitude and longitude must be finite")
    if not np.isfinite(land_sea_mask).all():
        raise ValueError("land_sea_mask must be finite")
    if np.any((land_sea_mask < 0) | (land_sea_mask > 1)):
        raise ValueError("land_sea_mask values must be between 0 and 1")
    if np.any(np.abs(latitude) > 90):
        raise ValueError("latitude values must be between -90 and 90 degrees")
    if not np.isfinite(coastal_distance_km) or coastal_distance_km <= 0:
        raise ValueError("coastal_distance_km must be finite and positive")
    if not 0 <= land_fraction_threshold <= 1:
        raise ValueError("land_fraction_threshold must be between 0 and 1")

    land = land_sea_mask >= land_fraction_threshold
    sea = ~land
    if not land.any() or not sea.any():
        raise ValueError("land_sea_mask must contain both land and sea cells")

    coordinates = _unit_sphere_coordinates(latitude, longitude)
    land_distances = np.full(land.size, np.inf, dtype=np.float64)
    sea_distances = np.full(sea.size, np.inf, dtype=np.float64)
    land_distances[land.ravel()] = _nearest_other_class_distance_km(
        coordinates[land.ravel()], coordinates[sea.ravel()]
    )
    sea_distances[sea.ravel()] = _nearest_other_class_distance_km(
        coordinates[sea.ravel()], coordinates[land.ravel()]
    )
    land_distances = land_distances.reshape(land.shape)
    sea_distances = sea_distances.reshape(sea.shape)

    return {
        "all_grid": np.ones(land.shape, dtype=bool),
        "coastal_land": land & (land_distances <= coastal_distance_km),
        "inland_land": land & (land_distances > coastal_distance_km),
        "nearshore_sea": sea & (sea_distances <= coastal_distance_km),
        "offshore_sea": sea & (sea_distances > coastal_distance_km),
    }


def build_configured_region_masks(
    latitude,
    longitude,
    land_sea_mask,
    region_definitions: Mapping[str, Mapping[str, object]],
    *,
    land_fraction_threshold=OCEAN_LAND_FRACTION_THRESHOLD,
):
    """Build named masks from polygon/bounds and optional land/coast filters.

    If ``regions`` are configured, only those names are returned. Polygon files
    may be GeoJSON geometries, Features, or FeatureCollections of polygons.
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    land_sea_mask = np.asarray(land_sea_mask, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional")
    if land_sea_mask.shape != (latitude.size, longitude.size):
        raise ValueError("land_sea_mask shape must match latitude/longitude")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("latitude and longitude must be finite")
    if np.any(np.abs(latitude) > 90):
        raise ValueError("latitude values must be between -90 and 90 degrees")
    if not np.isfinite(land_sea_mask).all() or np.any(
        (land_sea_mask < 0) | (land_sea_mask > 1)
    ):
        raise ValueError("land_sea_mask must contain finite fractions from 0 to 1")
    if (
        not np.isfinite(land_fraction_threshold)
        or not 0 <= land_fraction_threshold <= 1
    ):
        raise ValueError("land_fraction_threshold must be between 0 and 1")
    if not isinstance(region_definitions, Mapping) or not region_definitions:
        raise ValueError("Configured regions must be a non-empty mapping")

    land = land_sea_mask >= land_fraction_threshold
    sea = ~land
    coordinates = _unit_sphere_coordinates(latitude, longitude)
    land_distances = np.full(land.size, np.inf, dtype=np.float64)
    sea_distances = np.full(sea.size, np.inf, dtype=np.float64)
    if land.any() and sea.any():
        land_distances[land.ravel()] = _nearest_other_class_distance_km(
            coordinates[land.ravel()], coordinates[sea.ravel()]
        )
        sea_distances[sea.ravel()] = _nearest_other_class_distance_km(
            coordinates[sea.ravel()], coordinates[land.ravel()]
        )
    land_distances = land_distances.reshape(land.shape)
    sea_distances = sea_distances.reshape(sea.shape)

    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    masks = {}
    for name, definition in region_definitions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Configured region names must be non-empty strings")
        if not isinstance(definition, Mapping):
            raise ValueError(f"Region {name!r} must be a mapping")

        polygon_file = definition.get("polygon_file")
        bounds = definition.get("bounds")
        if polygon_file is not None and bounds is not None:
            raise ValueError(
                f"Region {name!r} cannot specify both polygon_file and bounds"
            )
        if polygon_file is not None:
            geometry = _load_polygon(polygon_file, name)
            mask = intersects_xy(geometry, longitude_grid, latitude_grid)
        elif bounds is not None:
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    for value in bounds
                )
            ):
                raise ValueError(
                    f"Region {name!r} bounds must be "
                    "[min_longitude, min_latitude, max_longitude, max_latitude]"
                )
            min_lon, min_lat, max_lon, max_lat = map(float, bounds)
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError(f"Region {name!r} bounds must have positive area")
            if min_lat < -90 or max_lat > 90:
                raise ValueError(f"Region {name!r} latitude bounds must be in [-90, 90]")
            geometry = box(min_lon, min_lat, max_lon, max_lat)
            mask = intersects_xy(geometry, longitude_grid, latitude_grid)
        else:
            mask = np.ones(land.shape, dtype=bool)

        land_sea = definition.get("land_sea", "all")
        if land_sea not in ("all", "land", "sea"):
            raise ValueError(
                f"Region {name!r} land_sea must be 'all', 'land', or 'sea'"
            )
        if land_sea == "land":
            mask &= land
            distances = land_distances
        elif land_sea == "sea":
            mask &= sea
            distances = sea_distances
        else:
            distances = None

        minimum_distance = definition.get("min_coast_distance_km")
        maximum_distance = definition.get("max_coast_distance_km")
        if minimum_distance is not None or maximum_distance is not None:
            if distances is None:
                raise ValueError(
                    f"Region {name!r} must set land_sea to land or sea "
                    "when using coast-distance filters"
                )
            minimum_distance = (
                0.0 if minimum_distance is None else float(minimum_distance)
            )
            maximum_distance = (
                np.inf if maximum_distance is None else float(maximum_distance)
            )
            if (
                not np.isfinite(minimum_distance)
                or minimum_distance < 0
                or np.isnan(maximum_distance)
                or maximum_distance < 0
                or maximum_distance <= minimum_distance
            ):
                raise ValueError(
                    f"Region {name!r} coast-distance bounds are invalid"
                )
            mask &= distances >= minimum_distance
            mask &= distances <= maximum_distance

        if not mask.any():
            raise ValueError(
                f"Region {name!r} selects no grid-cell centers; check its "
                "polygon/bounds and filters"
            )
        masks[name] = np.asarray(mask, dtype=bool)
    return masks


def _load_polygon(polygon_file, region_name):
    path = Path(polygon_file)
    try:
        with path.open("r", encoding="utf-8") as file:
            geojson = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read GeoJSON polygon for region {region_name!r}: {path}"
        ) from exc

    if geojson.get("type") == "Feature":
        geometries = [shape(geojson["geometry"])]
    elif geojson.get("type") == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in geojson.get("features", [])
            if feature.get("geometry") is not None
        ]
    else:
        geometries = [shape(geojson)]
    if not geometries:
        raise ValueError(f"GeoJSON for region {region_name!r} contains no geometry")
    if any(
        geometry.is_empty
        or geometry.geom_type not in ("Polygon", "MultiPolygon")
        or not geometry.is_valid
        for geometry in geometries
    ):
        raise ValueError(
            f"GeoJSON for region {region_name!r} must contain valid polygons"
        )
    return unary_union(geometries)


def _nearest_other_class_distance_km(query_coordinates, reference_coordinates):
    chord_distances, _ = cKDTree(reference_coordinates).query(query_coordinates)
    return 2 * EARTH_RADIUS_KM * np.arcsin(
        np.clip(chord_distances / 2, 0, 1)
    )


class SpatialMetricAccumulator:
    """Accumulate area-weighted MAE/RMSE for regions and forecast leads."""

    def __init__(
        self,
        region_masks: Mapping[str, np.ndarray],
        latitude,
        boundary_mask,
        variable_names: Sequence[str],
        variable_units: Sequence[str],
        training_std: Sequence[float],
        *,
        land_sea_mask=None,
        exclude_land_sst=True,
    ):
        self.region_masks = {
            name: np.asarray(mask, dtype=bool)
            for name, mask in region_masks.items()
        }
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.boundary_mask = np.asarray(boundary_mask, dtype=bool)
        self.land_sea_mask = (
            None if land_sea_mask is None
            else np.asarray(land_sea_mask, dtype=np.float64)
        )
        self.variable_names = tuple(variable_names)
        self.variable_units = tuple(variable_units)
        self.training_std = np.asarray(training_std, dtype=np.float64)
        self.exclude_land_sst = exclude_land_sst
        if not self.region_masks:
            raise ValueError("At least one region mask is required")
        if len(self.variable_names) != len(self.variable_units):
            raise ValueError("Each regional variable must have one unit")
        if self.training_std.shape != (len(self.variable_names),):
            raise ValueError("training_std must contain one value per regional variable")
        if not np.isfinite(self.training_std).all() or np.any(self.training_std <= 0):
            raise ValueError("Regional training standard deviations must be positive")
        if self.boundary_mask.ndim != 2 or self.boundary_mask.shape[0] != self.latitude.size:
            raise ValueError("boundary_mask shape does not match regional grid")
        expected_shape = self.boundary_mask.shape
        if any(mask.shape != expected_shape for mask in self.region_masks.values()):
            raise ValueError("Region mask shapes do not match one another")
        if self.land_sea_mask is not None and self.land_sea_mask.shape != expected_shape:
            raise ValueError("land_sea_mask shape does not match regional grid")
        if self.exclude_land_sst and "sst" in self.variable_names:
            if self.land_sea_mask is None:
                raise ValueError("land_sea_mask is required to exclude land SST")

        self._absolute_error_sum = None
        self._squared_error_sum = None
        self._weight_sum = None
        self._sample_count = None

    def update(self, prediction, target):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("regional arrays must match (init, lead, y, x, variable)")
        if prediction.shape[0] != 1:
            raise ValueError("regional updates must contain one initialization")
        if prediction.shape[2:] != (
            self.boundary_mask.shape[0],
            self.boundary_mask.shape[1],
            len(self.variable_names),
        ):
            raise ValueError("regional arrays do not match configured grid/variables")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("regional prediction and target values must be finite")

        n_leads = prediction.shape[1]
        shape = (len(self.region_masks), n_leads, len(self.variable_names))
        if self._absolute_error_sum is None:
            self._absolute_error_sum = np.zeros(shape, dtype=np.float64)
            self._squared_error_sum = np.zeros(shape, dtype=np.float64)
            self._weight_sum = np.zeros(shape, dtype=np.float64)
            self._sample_count = np.zeros(shape, dtype=np.int64)
        elif self._absolute_error_sum.shape != shape:
            raise ValueError("regional update lead count changed")

        errors = prediction[0].astype(np.float64) - target[0].astype(np.float64)
        area_weights = np.broadcast_to(
            np.cos(np.deg2rad(self.latitude))[:, None],
            self.boundary_mask.shape,
        )
        interior = ~self.boundary_mask
        for region_index, region_mask in enumerate(self.region_masks.values()):
            valid = region_mask & interior
            for variable_index, variable_name in enumerate(self.variable_names):
                variable_valid = valid
                if self.exclude_land_sst and variable_name == "sst":
                    variable_valid = valid & (
                        self.land_sea_mask < OCEAN_LAND_FRACTION_THRESHOLD
                    )
                weights = area_weights * variable_valid
                self._absolute_error_sum[region_index, :, variable_index] += (
                    np.sum(
                        np.abs(errors[..., variable_index]) * weights[None, :, :],
                        axis=(1, 2),
                    )
                )
                self._squared_error_sum[region_index, :, variable_index] += (
                    np.sum(
                        np.square(errors[..., variable_index]) * weights[None, :, :],
                        axis=(1, 2),
                    )
                )
                self._weight_sum[region_index, :, variable_index] += weights.sum()
                self._sample_count[region_index, :, variable_index] += int(
                    variable_valid.sum()
                )

    def to_records(
        self,
        lead_hours,
        *,
        model_id,
        checkpoint_id,
        metadata,
    ):
        if self._absolute_error_sum is None:
            raise ValueError("No regional metric samples were accumulated")
        records = []
        for region_index, region_name in enumerate(self.region_masks):
            for lead_index, lead in enumerate(lead_hours):
                for variable_index, variable_name in enumerate(self.variable_names):
                    weight_sum = self._weight_sum[
                        region_index, lead_index, variable_index
                    ]
                    count = int(self._sample_count[
                        region_index, lead_index, variable_index
                    ])
                    mae = (
                        self._absolute_error_sum[
                            region_index, lead_index, variable_index
                        ] / weight_sum
                        if weight_sum > 0 else np.nan
                    )
                    mse = (
                        self._squared_error_sum[
                            region_index, lead_index, variable_index
                        ] / weight_sum
                        if weight_sum > 0 else np.nan
                    )
                    values = {
                        "regional_mae": mae,
                        "regional_rmse": np.sqrt(mse) if np.isfinite(mse) else np.nan,
                        "regional_normalized_mae": mae / self.training_std[variable_index],
                        "regional_normalized_rmse": (
                            np.sqrt(mse) / self.training_std[variable_index]
                            if np.isfinite(mse) else np.nan
                        ),
                    }
                    for metric_name, value in values.items():
                        records.append({
                            "model_id": model_id,
                            "checkpoint_id": checkpoint_id,
                            "lead_hours": int(lead),
                            "variable": variable_name,
                            "unit": (
                                self.variable_units[variable_index]
                                if metric_name in {"regional_mae", "regional_rmse"}
                                else "1"
                            ),
                            "metric": metric_name,
                            "value": float(value) if np.isfinite(value) else None,
                            "sample_count": count,
                            "weight_sum": float(weight_sum),
                            **metadata,
                            "region": region_name,
                            "area_weighting": "cosine of latitude",
                        })
        return records

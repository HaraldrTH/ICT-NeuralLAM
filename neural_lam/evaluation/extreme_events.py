"""Training-derived thresholds and event verification for cached forecasts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import numpy as np

from neural_lam import constants
from neural_lam.evaluation.metrics import derive_10m_wind_speed
from neural_lam.weather_dataset import open_store


_EVENT_SOURCE_VARIABLES = {
    "t2m": ("t2m",),
    "msl": ("msl",),
    "wind_speed_10m": ("u10", "v10"),
}
_EVENT_DIRECTIONS = {
    "t2m": ("upper", "lower"),
    "msl": ("lower",),
    "wind_speed_10m": ("upper",),
}


def compute_training_event_thresholds(
    dataset_name,
    *,
    lower_percentile=1.0,
    upper_percentile=99.0,
    block_size=64,
):
    """Compute empirical training-split percentiles without holding all data in RAM."""
    if not 0 < lower_percentile < 50 < upper_percentile < 100:
        raise ValueError(
            "Percentiles must satisfy 0 < lower < 50 < upper < 100"
        )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    static_dir = Path("data") / dataset_name / "static"
    border_mask = np.load(static_dir / "border_mask.npy").astype(bool)
    if border_mask.ndim != 2 or not (~border_mask).any():
        raise ValueError("border_mask must contain interior grid cells")
    interior = ~border_mask
    dataset = open_store(dataset_name)
    try:
        if dataset["state"].dims != ("time", "y", "x", "var"):
            raise ValueError(f"Unexpected ERA5 state dimensions: {dataset['state'].dims}")
        if dataset["state"].shape[1:3] != interior.shape:
            raise ValueError("border_mask and training-data grids do not match")
        store_variables = list(dataset.attrs["state_vars"])
        missing = sorted({
            name for source_names in _EVENT_SOURCE_VARIABLES.values()
            for name in source_names if name not in store_variables
        })
        if missing:
            raise ValueError(
                "Training store is missing extreme-event variables: "
                + ", ".join(missing)
            )
        variable_indices = {
            name: store_variables.index(name)
            for source_names in _EVENT_SOURCE_VARIABLES.values()
            for name in source_names
        }
        times = dataset.time.values
        start, end = (
            np.datetime64(value, "ns")
            for value in constants.split_times["train"]
        )
        training_indices = np.flatnonzero((times >= start) & (times <= end))
        if not len(training_indices):
            raise ValueError(f"No training times found for dataset {dataset_name}")
        if not np.all(np.diff(training_indices) == 1):
            raise ValueError("Training split timestamps are not contiguous")

        sample_count = int(len(training_indices) * interior.sum())
        if sample_count <= 0:
            raise ValueError("Training data has no interior samples")

        with TemporaryDirectory(prefix="neural_lam_event_thresholds_") as temp_dir:
            memmaps = {}
            try:
                memmaps = {
                    name: np.memmap(
                        Path(temp_dir) / f"{name}.bin",
                        mode="w+",
                        dtype=np.float32,
                        shape=(sample_count,),
                    )
                    for name in _EVENT_SOURCE_VARIABLES
                }
                offset = 0
                source_indices = sorted(variable_indices.values())
                source_positions = {
                    name: source_indices.index(index)
                    for name, index in variable_indices.items()
                }
                for block_start in range(0, len(training_indices), block_size):
                    time_indices = training_indices[block_start:block_start + block_size]
                    block = np.asarray(
                        dataset["state"].isel(
                            time=time_indices, var=source_indices
                        ).values,
                        dtype=np.float32,
                    )
                    if not np.isfinite(block).all():
                        raise ValueError(
                            "Training data contains non-finite event values"
                        )
                    block_count = len(time_indices) * int(interior.sum())
                    block_slice = slice(offset, offset + block_count)
                    for name in ("t2m", "msl"):
                        values = block[..., source_positions[name]][:, interior]
                        memmaps[name][block_slice] = values.reshape(-1)
                    wind_speed = np.hypot(
                        block[..., source_positions["u10"]],
                        block[..., source_positions["v10"]],
                    )[:, interior]
                    memmaps["wind_speed_10m"][block_slice] = wind_speed.reshape(-1)
                    offset += block_count
                if offset != sample_count:
                    raise RuntimeError(
                        "Training threshold sample count was inconsistent"
                    )

                thresholds = {}
                for name, mapped_values in memmaps.items():
                    mapped_values.flush()
                    lower, upper = np.quantile(
                        mapped_values,
                        [lower_percentile / 100, upper_percentile / 100],
                        method="linear",
                        overwrite_input=True,
                    )
                    thresholds[name] = {
                        "lower": float(lower),
                        "upper": float(upper),
                        "lower_percentile": float(lower_percentile),
                        "upper_percentile": float(upper_percentile),
                        "sample_count": sample_count,
                        "units": "m/s" if name == "wind_speed_10m" else (
                            "K" if name == "t2m" else "Pa"
                        ),
                        "derived_from": (
                            ["u10", "v10"] if name == "wind_speed_10m" else [name]
                        ),
                    }
            finally:
                for mapped_values in memmaps.values():
                    if mapped_values._mmap is not None:
                        mapped_values._mmap.close()
        return {
            "dataset": dataset_name,
            "split": "train",
            "scope": "all interior grid cells and training timestamps",
            "thresholds": thresholds,
        }
    finally:
        dataset.close()


class ExtremeEventAccumulator:
    """Accumulate area-weighted contingency scores and errors during events."""

    def __init__(
        self,
        region_masks: Mapping[str, np.ndarray],
        latitude,
        boundary_mask,
        thresholds: Mapping[str, Mapping[str, object]],
    ):
        self.region_masks = {
            name: np.asarray(mask, dtype=bool)
            for name, mask in region_masks.items()
        }
        if not self.region_masks:
            raise ValueError("At least one region mask is required")
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.boundary_mask = np.asarray(boundary_mask, dtype=bool)
        self.thresholds = thresholds
        if self.boundary_mask.ndim != 2 or self.boundary_mask.shape[0] != self.latitude.size:
            raise ValueError("boundary_mask shape does not match event grid")
        if any(mask.shape != self.boundary_mask.shape for mask in self.region_masks.values()):
            raise ValueError("Region mask shapes do not match event grid")
        self._states = {}

    def update(self, predictions, targets):
        if set(predictions) != set(targets):
            raise ValueError("Event prediction and target variables must match")
        for variable_name, prediction in predictions.items():
            if variable_name not in self.thresholds:
                raise ValueError(f"Missing event threshold for {variable_name}")
            prediction = np.asarray(prediction)
            target = np.asarray(targets[variable_name])
            if prediction.shape != target.shape or prediction.ndim != 4:
                raise ValueError("Event arrays must match (init, lead, y, x)")
            if prediction.shape[0] != 1 or prediction.shape[2:] != self.boundary_mask.shape:
                raise ValueError("Event update must contain one initialization on the cache grid")
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                raise ValueError("Event prediction and target values must be finite")

            for direction in _EVENT_DIRECTIONS[variable_name]:
                threshold_info = self.thresholds[variable_name]
                threshold = float(threshold_info[direction])
                observed = target[0] >= threshold if direction == "upper" else target[0] <= threshold
                forecast = (
                    prediction[0] >= threshold
                    if direction == "upper"
                    else prediction[0] <= threshold
                )
                errors = prediction[0].astype(np.float64) - target[0].astype(np.float64)
                for region_name, region_mask in self.region_masks.items():
                    valid = region_mask & ~self.boundary_mask
                    weights = np.cos(np.deg2rad(self.latitude))[:, None] * valid
                    valid_3d = valid[None, :, :]
                    observed_valid = observed & valid_3d
                    forecast_valid = forecast & valid_3d
                    state_key = (region_name, variable_name, direction)
                    state = self._states.setdefault(
                        state_key,
                        {
                            "sample_count": np.zeros(
                                prediction.shape[1], dtype=np.int64
                            ),
                            "observed_event_count": np.zeros(
                                prediction.shape[1], dtype=np.int64
                            ),
                            "forecast_event_count": np.zeros(
                                prediction.shape[1], dtype=np.int64
                            ),
                            "true_positive_weight": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                            "false_positive_weight": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                            "false_negative_weight": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                            "extreme_absolute_error_sum": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                            "extreme_squared_error_sum": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                            "observed_event_weight": np.zeros(
                                prediction.shape[1], dtype=np.float64
                            ),
                        },
                    )
                    state["sample_count"] += int(valid.sum())
                    state["observed_event_count"] += observed_valid.sum(
                        axis=(1, 2)
                    )
                    state["forecast_event_count"] += forecast_valid.sum(
                        axis=(1, 2)
                    )
                    true_positive = observed_valid & forecast_valid
                    false_positive = ~observed_valid & forecast_valid & valid_3d
                    false_negative = observed_valid & ~forecast_valid
                    state["true_positive_weight"] += float(
                        np.sum(true_positive * weights[None, :, :])
                    )
                    state["false_positive_weight"] += float(
                        np.sum(false_positive * weights[None, :, :])
                    )
                    state["false_negative_weight"] += float(
                        np.sum(false_negative * weights[None, :, :])
                    )
                    state["extreme_absolute_error_sum"] += np.sum(
                        np.abs(errors) * observed_valid * weights[None, :, :],
                        axis=(1, 2),
                    )
                    state["extreme_squared_error_sum"] += np.sum(
                        np.square(errors) * observed_valid * weights[None, :, :],
                        axis=(1, 2),
                    )
                    state["observed_event_weight"] += np.sum(
                        observed_valid * weights[None, :, :], axis=(1, 2)
                    )

    def to_records(
        self,
        lead_hours: Sequence[int],
        *,
        model_id,
        checkpoint_id,
        metadata,
    ):
        if not self._states:
            raise ValueError("No event metrics were accumulated")
        records = []
        for (region_name, variable_name, direction), state in self._states.items():
            threshold_info = self.thresholds[variable_name]
            threshold = float(threshold_info[direction])
            percentile = float(threshold_info[f"{direction}_percentile"])
            for lead_index, lead in enumerate(lead_hours):
                true_positive = state["true_positive_weight"][lead_index]
                false_positive = state["false_positive_weight"][lead_index]
                false_negative = state["false_negative_weight"][lead_index]
                observed_weight = state["observed_event_weight"][lead_index]
                absolute_error_sum = state["extreme_absolute_error_sum"][lead_index]
                squared_error_sum = state["extreme_squared_error_sum"][lead_index]
                metric_values = {
                    "event_pod": _divide(true_positive, true_positive + false_negative),
                    "event_precision": _divide(
                        true_positive, true_positive + false_positive
                    ),
                    "event_far": _divide(
                        false_positive, true_positive + false_positive
                    ),
                    "event_csi": _divide(
                        true_positive,
                        true_positive + false_positive + false_negative,
                    ),
                    "event_frequency_bias": _divide(
                        true_positive + false_positive,
                        true_positive + false_negative,
                    ),
                    "extreme_mae": _divide(absolute_error_sum, observed_weight),
                    "extreme_rmse": (
                        np.sqrt(_divide(squared_error_sum, observed_weight))
                        if observed_weight > 0 else np.nan
                    ),
                }
                for metric_name, value in metric_values.items():
                    is_error = metric_name in {"extreme_mae", "extreme_rmse"}
                    records.append({
                        "model_id": model_id,
                        "checkpoint_id": checkpoint_id,
                        "lead_hours": int(lead),
                        "variable": variable_name,
                        "unit": threshold_info["units"] if is_error else "1",
                        "metric": metric_name,
                        "value": float(value) if np.isfinite(value) else None,
                        "sample_count": int(state["sample_count"][lead_index]),
                        **metadata,
                        "region": region_name,
                        "event_type": direction,
                        "threshold": threshold,
                        "threshold_units": threshold_info["units"],
                        "training_percentile": percentile,
                        "observed_event_count": int(
                            state["observed_event_count"][lead_index]
                        ),
                        "forecast_event_count": int(
                            state["forecast_event_count"][lead_index]
                        ),
                        "true_positive_area_weight": true_positive,
                        "false_positive_area_weight": false_positive,
                        "false_negative_area_weight": false_negative,
                        "event_error_weight": float(observed_weight),
                        "area_weighting": "cosine of latitude",
                        "threshold_scope": (
                            "training percentiles over all interior grid cells"
                        ),
                    })
        return records


def _divide(numerator, denominator):
    if denominator <= 0:
        return np.nan
    return numerator / denominator

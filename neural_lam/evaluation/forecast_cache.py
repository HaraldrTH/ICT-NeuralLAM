"""Appendable, self-describing Zarr cache for forecast evaluation."""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import zarr
from numcodecs import Blosc


def _numpy(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return result


def _create_array(group, name, shape, chunks, dtype, compressor, data=None):
    create_array = getattr(group, "create_array", None)
    create = create_array if create_array is not None else group.create_dataset
    kwargs = {"chunks": chunks, "compressor": compressor}
    if data is not None:
        kwargs["data"] = data
    else:
        kwargs["shape"] = shape
        kwargs["dtype"] = dtype
    return create(name, **kwargs)


def _write_coordinate(group, name, values, dimension, compressor):
    values = np.asarray(values)
    chunks = tuple(max(1, int(size)) for size in values.shape)
    array = _create_array(
        group, name, values.shape, chunks, values.dtype, compressor,
        data=values,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = [dimension]
    return array


class ZarrForecastCache:
    """Append one batch of forecasts at a time to a non-overwriting Zarr store."""

    def __init__(
        self,
        path,
        *,
        model_id: str,
        checkpoint_id: Optional[str],
        dataset: str,
        split: str,
        graph: Optional[str],
        config_path: Optional[str],
        step_length_hours: int,
        lead_hours: Sequence[int],
        variable_names: Sequence[str],
        variable_units: Sequence[str],
        latitude,
        longitude,
        land_sea_mask,
        boundary_mask,
        subset: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.lead_hours = np.asarray(lead_hours, dtype=np.int32)
        self.variable_names = tuple(variable_names)
        self.variable_units = tuple(variable_units)
        if not len(self.lead_hours) or np.any(self.lead_hours <= 0):
            raise ValueError("lead_hours must contain positive values")
        if len(set(self.lead_hours.tolist())) != len(self.lead_hours):
            raise ValueError("lead_hours must not contain duplicates")
        if len(self.variable_names) != len(self.variable_units):
            raise ValueError("Each variable must have one unit")

        latitude = np.asarray(latitude, dtype=np.float32)
        longitude = np.asarray(longitude, dtype=np.float32)
        land_sea_mask = np.asarray(land_sea_mask, dtype=np.float32)
        boundary_mask = np.asarray(boundary_mask, dtype=bool)

        expected_mask_shape = (len(latitude), len(longitude))
        if land_sea_mask.shape != expected_mask_shape:
            raise ValueError("land_sea_mask shape does not match coordinates")
        if boundary_mask.shape != expected_mask_shape:
            raise ValueError("boundary_mask shape does not match coordinates")

        self.grid_shape = expected_mask_shape
        self.n_variables = len(self.variable_names)
        self.n_leads = len(self.lead_hours)
        self.compressor = Blosc(cname="zstd",
                                clevel=3,
                                shuffle=Blosc.BITSHUFFLE)

        # Force Zarr v2 format. This works with the project's Zarr 2 pin and
        # with Zarr 3 installations that can read/write v2 stores.
        if int(zarr.__version__.split(".", maxsplit=1)[0]) >= 3:
            self.group = zarr.open_group(
                str(self.path),
                mode="w-",
                zarr_format=2,
            )
        else:
            self.group = zarr.open_group(str(self.path), mode="w-")

        self.group.attrs.update({
            "schema_version": 1,
            "status": "writing",
            "complete": False,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "dataset": dataset,
            "split": split,
            "graph": graph,
            "config_path": config_path,
            "subset": bool(subset),
            "step_length_hours": int(step_length_hours),
            "variable_names": list(self.variable_names),
            "variable_units": list(self.variable_units),
            "prediction_units": "physical",
            "boundary_cells_use_reference_truth": True,
        })

        ny, nx = self.grid_shape
        nvars = self.n_variables
        nleads = self.n_leads
        float_chunks = self.compressor

        _write_coordinate(
            self.group, "initialization", np.empty(0, dtype=np.int64),
            "initialization", float_chunks,
        )
        _write_coordinate(
            self.group, "lead_hours", self.lead_hours, "lead", float_chunks
        )
        _write_coordinate(
            self.group, "y", np.arange(ny, dtype=np.int32), "y", float_chunks
        )
        _write_coordinate(
            self.group, "x", np.arange(nx, dtype=np.int32), "x", float_chunks
        )
        _write_coordinate(
            self.group, "latitude", latitude, "y", float_chunks
        )
        _write_coordinate(
            self.group, "longitude", longitude, "x", float_chunks
        )

        initialization_times = _create_array(
            self.group, "initialization_time_unix_s", (0,), (256,),
            np.int64, self.compressor,
        )
        initialization_times.attrs["_ARRAY_DIMENSIONS"] = ["initialization"]

        valid_times = _create_array(
            self.group, "valid_time_unix_s", (0, nleads), (1, nleads),
            np.int64, self.compressor,
        )
        valid_times.attrs["_ARRAY_DIMENSIONS"] = ["initialization", "lead"]

        prediction_shape = (0, nleads, ny, nx, nvars)
        prediction_chunks = (1, 1, ny, nx, nvars)
        self.predictions = _create_array(
            self.group, "prediction", prediction_shape, prediction_chunks,
            np.float32, self.compressor,
        )
        self.targets = _create_array(
            self.group, "target", prediction_shape, prediction_chunks,
            np.float32, self.compressor,
        )
        for array in (self.predictions, self.targets):
            array.attrs["_ARRAY_DIMENSIONS"] = [
                "initialization", "lead", "y", "x", "variable"
            ]
            array.attrs["units_by_variable"] = list(self.variable_units)

        initial_shape = (0, ny, nx, nvars)
        initial_chunks = (1, ny, nx, nvars)
        self.initial_state = _create_array(
            self.group, "latest_observed_state", initial_shape, initial_chunks,
            np.float32, self.compressor,
        )
        self.initial_state.attrs["_ARRAY_DIMENSIONS"] = [
            "initialization", "y", "x", "variable"
        ]
        self.initial_state.attrs["units_by_variable"] = list(self.variable_units)

        for name, values, dtype in (
            ("land_sea_mask", land_sea_mask, np.float32),
            ("boundary_mask", boundary_mask, np.bool_),
        ):
            values = values.astype(dtype, copy=False)
            chunks = tuple(max(1, int(size)) for size in values.shape)
            array = _create_array(
                self.group, name, values.shape, chunks, values.dtype,
                self.compressor, data=values,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

        self.initialization_indices = self.group["initialization"]
        self.initialization_times = initialization_times
        self.valid_times = valid_times

    def append(self, prediction, target, latest_observation, metadata: Mapping):
        """Append physical-unit data and its initialization/valid timestamps."""
        prediction = _numpy(prediction, "prediction").astype(np.float32, copy=False)
        target = _numpy(target, "target").astype(np.float32, copy=False)
        latest_observation = _numpy(
            latest_observation, "latest_observation"
        ).astype(np.float32, copy=False)

        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must match with shape (batch, lead, grid, variable)"
            )
        batch_size, nleads, ngrid, nvars = prediction.shape
        ny, nx = self.grid_shape
        if nleads != self.n_leads or nvars != self.n_variables:
            raise ValueError("Forecast dimensions do not match cache configuration")
        if ngrid != ny * nx:
            raise ValueError("Forecast grid size does not match cache coordinates")
        if latest_observation.shape != (batch_size, ngrid, nvars):
            raise ValueError("latest_observation shape does not match forecast")

        try:
            initialization_times = _numpy(
                metadata["initialization_time_unix_s"],
                "initialization_time_unix_s",
            ).astype(np.int64, copy=False).reshape(-1)
            valid_times = _numpy(
                metadata["valid_time_unix_s"],
                "valid_time_unix_s",
            ).astype(np.int64, copy=False)
        except KeyError as exc:
            raise ValueError(f"Missing timestamp metadata field: {exc.args[0]}") from exc

        if initialization_times.shape != (batch_size,):
            raise ValueError("Expected one initialization timestamp per forecast")
        if valid_times.shape != (batch_size, nleads):
            raise ValueError("Expected one valid timestamp per forecast lead")

        expected_valid_times = (
            initialization_times[:, None]
            + self.lead_hours[None, :] * 3600
        )
        if not np.array_equal(valid_times, expected_valid_times):
            raise ValueError(
                "Valid timestamps do not match initialization times and lead_hours"
            )

        if not np.isfinite(prediction).all():
            raise ValueError("prediction contains non-finite values")
        if not np.isfinite(target).all():
            raise ValueError("target contains non-finite values")
        if not np.isfinite(latest_observation).all():
            raise ValueError("latest_observation contains non-finite values")

        current_count = self.predictions.shape[0]
        if current_count and initialization_times[0] <= self.initialization_times[-1]:
            raise ValueError("Initialization timestamps must append in increasing order")
        if np.any(np.diff(initialization_times) <= 0):
            raise ValueError("Batch initialization timestamps must be strictly increasing")

        new_count = current_count + batch_size
        new_prediction_shape = (new_count, *self.predictions.shape[1:])
        new_initial_shape = (new_count, *self.initial_state.shape[1:])

        # Resize/write each dataset; status remains "writing" if interrupted.
        self.predictions.resize(new_prediction_shape)
        self.targets.resize(new_prediction_shape)
        self.initial_state.resize(new_initial_shape)
        self.initialization_indices.resize((new_count,))
        self.initialization_times.resize((new_count,))
        self.valid_times.resize((new_count, self.n_leads))

        prediction_grid = prediction.reshape(
            batch_size, nleads, ny, nx, nvars
        )
        target_grid = target.reshape(batch_size, nleads, ny, nx, nvars)
        initial_grid = latest_observation.reshape(batch_size, ny, nx, nvars)

        selection = slice(current_count, new_count)
        self.predictions[selection] = prediction_grid
        self.targets[selection] = target_grid
        self.initial_state[selection] = initial_grid
        self.initialization_indices[selection] = np.arange(
            current_count, new_count, dtype=np.int64
        )
        self.initialization_times[selection] = initialization_times
        self.valid_times[selection] = valid_times
        self.group.attrs["initializations_written"] = new_count

    def finish(self):
        """Mark the store complete after all evaluation batches were written."""
        count = self.predictions.shape[0]
        if count == 0:
            raise RuntimeError("Cannot mark an empty forecast cache complete")
        if any(array.shape[0] != count for array in (
            self.targets,
            self.initial_state,
            self.initialization_times,
            self.valid_times,
        )):
            raise RuntimeError("Forecast cache arrays have inconsistent lengths")
        self.group.attrs["initializations_written"] = count
        self.group.attrs["status"] = "complete"
        self.group.attrs["complete"] = True


def open_forecast_cache(path, require_complete=True):
    """Open a cache read-only, rejecting incomplete stores by default."""
    group = zarr.open_group(str(Path(path)), mode="r")
    if require_complete and not group.attrs.get("complete", False):
        raise RuntimeError(
            f"Forecast cache is incomplete: {path} "
            f"(status={group.attrs.get('status', 'unknown')})"
        )
    return group
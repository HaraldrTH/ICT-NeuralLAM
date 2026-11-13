"""Training-split statistics for derived evaluation variables."""

from pathlib import Path

import numpy as np

from neural_lam import constants, utils
from neural_lam.weather_dataset import open_store


OCEAN_LAND_FRACTION_THRESHOLD = 0.5
WIND_COMPONENT_PAIRS = {
    "10m": ("u10", "v10"),
    "500hPa": ("u_500", "v_500"),
    "700hPa": ("u_700", "v_700"),
    "850hPa": ("u_850", "v_850"),
    "925hPa": ("u_925", "v_925"),
}

def compute_derived_training_statistics(dataset_name, block_size=64):
    """Compute ocean-only SST and interior 10 m wind-speed training statistics.

    The SST mask uses ocean-dominant grid cells (land-sea fraction < 0.5).
    Wind speed is computed from physical-unit u10/v10 over interior cells.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    static_dir = Path("data") / dataset_name / "static"
    border_mask = np.load(static_dir / "border_mask.npy").astype(bool)
    if border_mask.ndim != 2:
        raise ValueError("border_mask must be a two-dimensional grid")
    interior = ~border_mask

    sst_index = constants.param_names.index("sst")
    wind_vars = tuple(dict.fromkeys(name for pair in WIND_COMPONENT_PAIRS.values() for name in pair))
    wind_var_indices = {name: constants.param_names.index(name) for name in wind_vars}
    selected_var_indices = [sst_index, *(wind_var_indices[name] for name in wind_vars)]
    block_columns = {name: index + 1 for index, name in enumerate(wind_vars)}
    wind_totals = {level: [0, 0.0, 0.0] for level in WIND_COMPONENT_PAIRS}

    stats = utils.load_dataset_stats(dataset_name, "cpu")
    data_mean = stats["data_mean"].detach().cpu().numpy()
    data_std = stats["data_std"].detach().cpu().numpy()
    ds = open_store(dataset_name)
    try:
        if ds["state"].dims != ("time", "y", "x", "var"):
            raise ValueError(f"Unexpected ERA5 state dimensions: {ds['state'].dims}")
        if ds.lsm.shape != interior.shape:
            raise ValueError("land-sea mask and border mask grids do not match")
        times = ds.time.values
        start, end = (np.datetime64(value, "ns") for value in constants.split_times["train"])
        training_indices = np.flatnonzero((times >= start) & (times <= end))
        if not len(training_indices):
            raise ValueError(f"No training times found for dataset {dataset_name}")
        if not np.all(np.diff(training_indices) == 1):
            raise ValueError("Training split timestamps are not contiguous")

        ocean_mask = ds.lsm.values < OCEAN_LAND_FRACTION_THRESHOLD
        valid_sst_mask = ocean_mask & interior
        if not valid_sst_mask.any() or not interior.any():
            raise ValueError("Training statistics masks contain no usable cells")

        sst_count = 0
        sst_sum = 0.0
        sst_squared_sum = 0.0
        wind_count = 0
        wind_sum = 0.0
        wind_squared_sum = 0.0
        for offset in range(0, len(training_indices), block_size):
            indices = training_indices[offset:offset + block_size]
            block = ds["state"].isel(time=indices, var=selected_var_indices).values
            block = np.asarray(block, dtype=np.float64)

            sst_values = block[..., 0][:, valid_sst_mask]
            if not np.isfinite(sst_values).all():
                raise ValueError("Training data contains non-finite SST values")

            sst_count += sst_values.size
            sst_sum += sst_values.sum(dtype=np.float64)
            sst_squared_sum += np.square(sst_values).sum(dtype=np.float64)

            for level, (u_name, v_name) in WIND_COMPONENT_PAIRS.items():
                speed = np.hypot(block[..., block_columns[u_name]],
                                 block[..., block_columns[v_name]])[:, interior]
                if not np.isfinite(speed).all():
                    raise ValueError(f"Training data contains non-finite wind speed at {level}")

                totals = wind_totals[level]
                totals[0] += speed.size
                totals[1] += speed.sum(dtype=np.float64)
                totals[2] += np.square(speed).sum(dtype=np.float64)

        def summarize(count, value_sum, squared_sum):
            mean = value_sum / count
            variance = max(squared_sum / count - mean**2, 0.0)
            return {"mean": float(mean), "std": float(np.sqrt(variance)), "sample_count": int(count)}

        sst_index = constants.param_names.index("sst")
        return {
            "dataset": dataset_name,
            "split": "train",
            "stats_scope": "training split, all times",
            "sst_ocean_dominant_interior": {
                **summarize(sst_count, sst_sum, sst_squared_sum),
                "land_sea_fraction_rule": (f"land_sea_mask < {OCEAN_LAND_FRACTION_THRESHOLD}"),
                "stored_all_grid_mean": float(data_mean[sst_index]),
                "stored_all_grid_std": float(data_std[sst_index]),
                "excludes_boundary_cells": True,
            },
            **{f"wind_speed_{level}_interior": {
                **summarize(*totals),
                "derived_from": list(WIND_COMPONENT_PAIRS[level]),
                "excludes_boundary_cells": True,
            }for level, totals in wind_totals.items()
            },
            "boundary_cells_excluded": True,
        }
    finally:
        ds.close()

import os
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch

from neural_lam.weather_dataset import open_store, toa_solar_radiation
from neural_lam import constants

def main():
    parser = ArgumentParser(description='Training arguments')
    parser.add_argument('--dataset', type=str, default="era5",
        help='Dataset to compute weights for (default: era5)')
    parser.add_argument('--step_length', type=int, default=6,
        help='Step length in hours to consider single time step (default: 6)')
    parser.add_argument('--block_size', type=int, default=100,
        help='Number of time steps to load at a time (default: 100)')
    args = parser.parse_args()

    assert args.step_length % constants.data_step_hours == 0, \
            f"Step length must be a multiple of {constants.data_step_hours} h"
    step = args.step_length // constants.data_step_hours # In data time steps

    static_dir_path = os.path.join("data", args.dataset, "static")

    # Parameter weights, as specified in constants
    print("Saving parameter weights...")
    np.save(os.path.join(static_dir_path, 'parameter_weights.npy'),
            np.array(constants.param_weights, dtype=np.float32))

    ds = open_store(args.dataset)
    times = ds.time.values
    store_vars = list(ds.attrs["state_vars"])
    var_index = np.array([store_vars.index(name) for name in constants.param_names])
    lon_grid, lat_grid = np.meshgrid(ds.longitude.values, ds.latitude.values)
    grid_lat, grid_lon = lat_grid.flatten(), lon_grid.flatten()

    # Statistics are computed over training split only
    split_first, split_last = (np.datetime64(t, "ns")
            for t in constants.split_times["train"])
    train_indices = np.nonzero((times >= split_first) & (times <= split_last))[0]
    train_start, train_end = train_indices[0], train_indices[-1] + 1
    assert len(train_indices) == train_end - train_start, \
            "Training split is not contiguous in data"

    # Accumulate sums in float64
    n_vals = 0
    var_sum = np.zeros(len(var_index))
    var_sq_sum = np.zeros(len(var_index))
    n_diffs = 0
    diff_sum = np.zeros(len(var_index))
    diff_sq_sum = np.zeros(len(var_index))
    flux_sum = 0.
    flux_sq_sum = 0.

    print("Computing mean and std.-dev. for parameters, one-step differences "\
            "and flux...")
    for block_start in tqdm(range(train_start, train_end, args.block_size)):
        block_end = min(block_start + args.block_size, train_end)
        # Load extra steps at end for differences
        load_end = min(block_end + step, train_end)
        block = ds["state"].isel(time=slice(block_start, load_end)).values[...,
                var_index].astype(np.float64) # (N_t, N_y, N_x, d_features)
        block = block.reshape(block.shape[0], -1, block.shape[-1]
                ) # (N_t, N_grid, d_features)
        block_times = times[block_start:load_end]
        n_own = block_end - block_start # Steps belonging to this block

        var_sum += block[:n_own].sum(axis=(0,1))
        var_sq_sum += (block[:n_own]**2).sum(axis=(0,1))
        n_vals += n_own*block.shape[1]

        # Differences between steps step_length apart, only where contiguous
        diff_starts = np.arange(min(n_own, block.shape[0] - step))
        contiguous = (block_times[diff_starts + step] - block_times[diff_starts] ==
                np.timedelta64(args.step_length, "h"))
        diff_starts = diff_starts[contiguous]
        diffs = block[diff_starts + step] - block[diff_starts]
        diff_sum += diffs.sum(axis=(0,1))
        diff_sq_sum += (diffs**2).sum(axis=(0,1))
        n_diffs += diffs.shape[0]*diffs.shape[1]

        flux = toa_solar_radiation(block_times[:n_own], grid_lat, grid_lon
                ).astype(np.float64)
        flux_sum += flux.sum()
        flux_sq_sum += (flux**2).sum()

    mean = var_sum/n_vals
    std = np.sqrt(var_sq_sum/n_vals - mean**2)

    flux_mean = flux_sum/n_vals # n_vals is number of steps * grid points
    flux_std = np.sqrt(flux_sq_sum/n_vals - flux_mean**2)

    # Differences are of standardized data, i.e. raw differences divided by std
    raw_diff_mean = diff_sum/n_diffs
    raw_diff_std = np.sqrt(diff_sq_sum/n_diffs - raw_diff_mean**2)
    diff_mean = raw_diff_mean/std
    diff_std = raw_diff_std/std

    print("Saving mean, std.-dev, flux_stats...")
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    torch.save(to_tensor(mean), os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(to_tensor(std), os.path.join(static_dir_path, "parameter_std.pt"))
    torch.save(to_tensor(np.array([flux_mean, flux_std])),
            os.path.join(static_dir_path, "flux_stats.pt"))

    print("Saving one-step difference mean and std.-dev...")
    torch.save(to_tensor(diff_mean), os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(to_tensor(diff_std), os.path.join(static_dir_path, "diff_std.pt"))

if __name__ == "__main__":
    main()

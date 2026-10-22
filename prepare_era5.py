import os
import glob
from argparse import ArgumentParser
import numpy as np
import xarray as xr
import cfgrib

from neural_lam import constants

SURFACE_VARS = ["msl", "u10", "v10", "t2m", "d2m", "sst"]
PRESSURE_VARS = ["z", "t", "u", "v", "q"]
PRESSURE_LEVELS = [500, 700, 850, 925]
SST_LAND_VALUE = 271.46 # K, value ERA5 uses for SST under sea ice

def state_var_names():
    """
    Names of all state variables written to the zarr store, in order
    """
    return SURFACE_VARS + ["tp"] + [f"{var}_{level}" for var in PRESSURE_VARS
            for level in PRESSURE_LEVELS]

def open_single_level(path):
    """
    Open single level GRIB file, returns instantaneous surface fields and
    precipitation (as DataArray indexed by valid time)
    """
    surface, tp = None, None
    for ds in cfgrib.open_datasets(path):
        if "msl" in ds:
            surface = ds
        elif "tp" in ds:
            tp = ds
        # Wave fields (swh, mwd, mwp) are on a coarser grid and are not used
    assert surface is not None and tp is not None, \
            f"Could not find surface fields and tp in {path}"

    # tp is stored as (base time, step), re-index by valid time
    tp_values = tp["tp"].values # (N_base, N_step, N_lat, N_lon)
    valid_times = tp["valid_time"].values.flatten()
    tp_da = xr.DataArray(tp_values.reshape(-1, *tp_values.shape[2:]),
            dims=("time", "latitude", "longitude"),
            coords={"time": valid_times, "latitude": tp.latitude,
                "longitude": tp.longitude})
    tp_da = tp_da.dropna("time", how="all")
    return surface, tp_da

def save_static_files(static_dir_path, latitude, longitude, args):
    """
    Save grid coordinates, border mask and (optionally) surface geopotential
    latitude: (N_y,) ascending, longitude: (N_x,)
    """
    print("Saving static files...")
    lon_grid, lat_grid = np.meshgrid(longitude, latitude) # (N_y, N_x)
    np.save(os.path.join(static_dir_path, "nwp_xy.npy"),
            np.stack((lon_grid, lat_grid)).astype(np.float32)) # (2, N_y, N_x)

    border_mask = np.ones(lon_grid.shape, dtype=bool)
    border_mask[args.border_width:-args.border_width,
            args.border_width:-args.border_width] = False
    np.save(os.path.join(static_dir_path, "border_mask.npy"), border_mask) # (N_y, N_x)

    if args.orography:
        orography = xr.open_dataset(args.orography)
        geopotential = orography["z"].squeeze().values
        if orography.latitude.values[0] > orography.latitude.values[-1]:
            geopotential = geopotential[::-1]
        assert geopotential.shape == lon_grid.shape, \
                "Orography grid does not match data grid"
        np.save(os.path.join(static_dir_path, "surface_geopotential.npy"),
                geopotential.astype(np.float32)) # (N_y, N_x)
    else:
        print("No orography given, skipping surface_geopotential.npy")

def main():
    parser = ArgumentParser(description='Convert ERA5 GRIB files to zarr store')
    parser.add_argument('--grib_dir', type=str, default="data/era5grib",
        help='Directory with single_level/ and pressure/<year>/ GRIB files '
        '(default: data/era5grib)')
    parser.add_argument('--dataset', type=str, default="era5",
        help='Name of dataset to create in data directory (default: era5)')
    parser.add_argument('--orography', type=str,
        help='Optional GRIB/netCDF file with surface geopotential (z) on the '
        'same grid (default: None)')
    parser.add_argument('--border_width', type=int, default=5,
        help='Number of grid cells along each edge used as boundary (default: 5)')
    parser.add_argument('--block_size', type=int, default=124,
        help='Number of time steps to convert at a time (default: 124)')
    parser.add_argument('--static_only', action='store_true',
        help='Only create static files, reading the grid from an existing zarr '
        'store instead of GRIB files')
    args = parser.parse_args()

    dataset_dir = os.path.join("data", args.dataset)
    static_dir_path = os.path.join(dataset_dir, "static")
    os.makedirs(static_dir_path, exist_ok=True)
    store_path = os.path.join(dataset_dir, constants.zarr_name)

    if args.static_only:
        store = xr.open_zarr(store_path)
        save_static_files(static_dir_path, store.latitude.values,
                store.longitude.values, args)
        return

    print("Opening single level data...")
    surface, tp = open_single_level(os.path.join(args.grib_dir, "single_level",
        "data.grib"))
    latitude = surface.latitude.values
    longitude = surface.longitude.values
    flip_lat = latitude[0] > latitude[-1] # Store latitude ascending (S -> N)
    if flip_lat:
        latitude = latitude[::-1]

    var_names = state_var_names()
    pressure_paths = sorted(glob.glob(os.path.join(args.grib_dir, "pressure", "*",
        "data.grib")))
    assert pressure_paths, "No pressure level files found"

    first_write = True
    last_time = None
    for pressure_path in pressure_paths:
        print(f"Converting {pressure_path}...")
        pressure = xr.open_dataset(pressure_path, engine="cfgrib")
        assert np.allclose(pressure.latitude.values[::-1] if flip_lat else
                pressure.latitude.values, latitude) and np.allclose(
                pressure.longitude.values, longitude), \
                f"Grid in {pressure_path} does not match single level grid"

        file_times = pressure.time.values
        for block_start in range(0, len(file_times), args.block_size):
            times = file_times[block_start:block_start+args.block_size]
            if last_time is not None:
                assert times[0] > last_time, "Pressure level files overlap in time"
            last_time = times[-1]

            surface_block = surface.sel(time=times)
            fields = []
            for var in SURFACE_VARS:
                values = surface_block[var].values
                if var == "sst":
                    # SST is undefined over land
                    values = np.nan_to_num(values, nan=SST_LAND_VALUE)
                fields.append(values)
            fields.append(tp.sel(time=times).values)
            pressure_block = pressure.sel(time=times)
            for var in PRESSURE_VARS:
                for level in PRESSURE_LEVELS:
                    fields.append(pressure_block[var].sel(isobaricInhPa=level).values)

            state = np.stack(fields, axis=-1).astype(np.float32) # (N_t, N_y, N_x, d)
            if flip_lat:
                state = state[:, ::-1]
            assert not np.isnan(state).any(), \
                    f"NaN in converted data for times {times[0]} - {times[-1]}"

            ds_out = xr.Dataset({"state": (("time", "y", "x", "var"), state)},
                    coords={"time": times})
            # Set on every write, appending replaces the attributes of the store
            ds_out.attrs["state_vars"] = var_names

            if first_write:
                lsm = surface["lsm"].isel(time=0).values
                if flip_lat:
                    lsm = lsm[::-1]
                ds_out["lsm"] = (("y", "x"), lsm.astype(np.float32))
                ds_out["latitude"] = (("y",), latitude)
                ds_out["longitude"] = (("x",), longitude)
                ds_out.to_zarr(store_path, mode="w", encoding={"state": {
                    "chunks": (1, *state.shape[1:])}})
                first_write = False
            else:
                ds_out.to_zarr(store_path, append_dim="time")

            print(f"\t{times[0]} - {times[-1]}")

    # Check that time series is contiguous
    all_times = xr.open_zarr(store_path).time.values
    steps = np.unique(np.diff(all_times))
    if len(steps) > 1 or steps[0] != np.timedelta64(constants.data_step_hours, "h"):
        print(f"Warning: time steps in data are not all {constants.data_step_hours} h, "
                f"found {steps}")

    save_static_files(static_dir_path, latitude, longitude, args)

if __name__ == "__main__":
    main()

import os
import glob
import urllib.request
import datetime as dt
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import eccodes
import numcodecs

from neural_lam import constants

# GRIB short names of the single level variables in constants.param_names
SINGLE_LEVEL_SHORT_NAMES = {
    'msl_meanSea_0_instant': 'msl',
    't_heightAboveGround_2_instant': '2t',
    'd_heightAboveGround_2_instant': '2d',
    'u_heightAboveGround_10_instant': '10u',
    'v_heightAboveGround_10_instant': '10v',
}
LAND_SEA_MASK_KEY = ('lsm', 0)

# Public copy of ERA5 (0.25 degree), used only for the static surface geopotential
WB2_ERA5_URL = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")

SOLAR_CONSTANT = 1361. # W/m^2

def grib_key(param_name):
    """
    Map name in constants.param_names to (shortName, level) of GRIB message
    """
    if param_name in SINGLE_LEVEL_SHORT_NAMES:
        return (SINGLE_LEVEL_SHORT_NAMES[param_name], 0)

    short_name, level_type, level, _ = param_name.split('_')
    assert level_type == "isobaricInhPa", f"Unknown variable: {param_name}"
    return (short_name, int(level))

def iterate_grib(path, headers_only=False):
    """
    Iterate over handles of all messages in GRIB file
    """
    with open(path, "rb") as grib_file:
        while True:
            gid = eccodes.codes_grib_new_from_file(grib_file,
                    headers_only=headers_only)
            if gid is None:
                break
            try:
                yield gid
            finally:
                eccodes.codes_release(gid)

def message_key(gid):
    return (eccodes.codes_get(gid, "shortName"), eccodes.codes_get(gid, "level"))

def message_time(gid):
    """
    Valid time of GRIB message
    """
    date = eccodes.codes_get(gid, "validityDate") # yyyymmdd
    time = eccodes.codes_get(gid, "validityTime") # hhmm
    return dt.datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")

def message_grid(gid):
    """
    Returns values on grid, shape (N_y, N_x), with y (lat) in ascending order
    """
    assert eccodes.codes_get(gid, "gridType") == "regular_ll", "Not a lat-lon grid"
    assert eccodes.codes_get(gid, "numberOfMissing") == 0, "Missing values in field"
    values = eccodes.codes_get_values(gid).reshape(
            eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni"))
    if not eccodes.codes_get(gid, "jScansPositively"):
        values = values[::-1] # Stored north to south
    return values

def scan_times(grib_paths, keys):
    """
    Find the time steps available for all given variables, and the grid coordinates

    Returns sorted list of datetimes, lats (N_y,) and lons (N_x,)
    """
    key_times = {key: set() for key in keys}
    lats, lons = None, None
    for path in grib_paths:
        for gid in tqdm(iterate_grib(path, headers_only=True), desc=f"Scanning {path}"):
            key = message_key(gid)
            if not key in key_times:
                continue
            key_times[key].add(message_time(gid))

            if lats is None:
                lats = np.sort(eccodes.codes_get_array(gid, "distinctLatitudes"))
                lons = eccodes.codes_get_array(gid, "distinctLongitudes")

    times = sorted(set.union(*key_times.values()))
    for key, var_times in key_times.items():
        assert len(var_times) == len(times), (f"Variable {key} only available at "
                f"{len(var_times)} of {len(times)} time steps")

    step = dt.timedelta(hours=constants.raw_step_hours)
    assert all(t2 - t1 == step for t1, t2 in zip(times[:-1], times[1:])), (
            f"Time steps are not all {constants.raw_step_hours} h apart")

    return times, lats, lons

def load_fields(grib_paths, keys, times, grid_shape):
    """
    Load all fields into one array of shape (N_t, N_y, N_x, d_features)
    """
    key_index = {key: i for i, key in enumerate(keys)}
    time_index = {time: i for i, time in enumerate(times)}

    fields = np.zeros((len(times), *grid_shape, len(keys)), dtype=np.float32)
    loaded = np.zeros((len(times), len(keys)), dtype=bool)
    for path in grib_paths:
        for gid in tqdm(iterate_grib(path), desc=f"Loading {path}"):
            key = message_key(gid)
            if not key in key_index:
                continue
            time_i, key_i = time_index[message_time(gid)], key_index[key]
            fields[time_i, :, :, key_i] = message_grid(gid)
            loaded[time_i, key_i] = True

    assert loaded.all(), "Some fields were not loaded"
    return fields

def load_land_sea_mask(grib_path):
    """
    Land-sea mask is constant in time, load from first message. Shape (N_y, N_x)
    """
    for gid in iterate_grib(grib_path):
        if message_key(gid) == LAND_SEA_MASK_KEY:
            return message_grid(gid).astype(np.float32)
    assert False, f"No land-sea mask in {grib_path}"

def download_surface_geopotential(lats, lons):
    """
    Download ERA5 surface geopotential from WeatherBench 2 and cut out grid.
    Returns array of shape (N_y, N_x)
    """
    def load_zarr_array(name, shape):
        chunk_name = ".".join("0" for _ in shape) # Arrays are stored as single chunks
        with urllib.request.urlopen(f"{WB2_ERA5_URL}/{name}/{chunk_name}") as response:
            decoded = numcodecs.Blosc().decode(response.read())
        return np.frombuffer(decoded, dtype="<f4").reshape(shape)

    global_lats = load_zarr_array("latitude", (721,))
    global_lons = load_zarr_array("longitude", (1440,))
    global_geopotential = load_zarr_array("geopotential_at_surface", (721, 1440))

    # Global grid is 0.25 degree, from lat 90 to -90 and lon 0 to 359.75
    lat_i = np.rint((90. - lats)/0.25).astype(int)
    lon_i = np.rint((lons % 360.)/0.25).astype(int)
    assert np.allclose(global_lats[lat_i], lats) and np.allclose(
            global_lons[lon_i], lons % 360.), "Grid does not match global ERA5 grid"

    return global_geopotential[np.ix_(lat_i, lon_i)]

def toa_solar_flux(times, lats, lons):
    """
    Instantaneous downwelling shortwave flux at top of atmosphere (W/m^2),
    computed from solar geometry (Spencer, 1971).

    Returns array of shape (N_t, N_y, N_x)
    """
    hour = np.array([t.hour + t.minute/60. for t in times]) # (N_t,)
    day_of_year = np.array([t.timetuple().tm_yday for t in times]) # (N_t,)
    day_angle = 2*np.pi*(day_of_year - 1 + hour/24.)/365. # (N_t,)

    # Correction for varying sun-earth distance
    dist_factor = 1.00011 + 0.034221*np.cos(day_angle) + 0.00128*np.sin(day_angle) +\
        0.000719*np.cos(2*day_angle) + 0.000077*np.sin(2*day_angle)
    declination = 0.006918 - 0.399912*np.cos(day_angle) + 0.070257*np.sin(day_angle) -\
        0.006758*np.cos(2*day_angle) + 0.000907*np.sin(2*day_angle) -\
        0.002697*np.cos(3*day_angle) + 0.00148*np.sin(3*day_angle)
    eq_of_time = 229.18*(0.000075 + 0.001868*np.cos(day_angle) -
        0.032077*np.sin(day_angle) - 0.014615*np.cos(2*day_angle) -
        0.040849*np.sin(2*day_angle)) # Minutes

    solar_time = (hour + eq_of_time/60.)[:,np.newaxis] + lons/15. # (N_t, N_x), hours
    hour_angle = np.deg2rad(15.*(solar_time - 12.))[:,np.newaxis] # (N_t, 1, N_x)

    lats_rad = np.deg2rad(lats)[np.newaxis,:,np.newaxis] # (1, N_y, 1)
    declination = declination[:,np.newaxis,np.newaxis] # (N_t, 1, 1)
    cos_zenith = np.sin(lats_rad)*np.sin(declination) +\
        np.cos(lats_rad)*np.cos(declination)*np.cos(hour_angle) # (N_t, N_y, N_x)

    flux = SOLAR_CONSTANT*dist_factor[:,np.newaxis,np.newaxis]*np.maximum(cos_zenith, 0.)
    return flux.astype(np.float32)

def main():
    parser = ArgumentParser(description='Convert ERA5 GRIB files to Neural-LAM dataset')
    parser.add_argument('--dataset', type=str, default="ERA5_nordic",
        help='Dataset directory with GRIB files in single_level and pressure '
            'sub-directories. Samples and static files are also saved here '
            '(default: ERA5_nordic)')
    parser.add_argument('--stride', type=int, default=10,
        help='Number of time steps between the start of consecutive samples '
            '(default: 10)')
    parser.add_argument('--border_width', type=int, default=10,
        help='Width of boundary forcing border, in grid cells (default: 10)')
    parser.add_argument('--val_start', type=str, default="2025-01-01",
        help='First date of validation period, earlier data is used for training '
            '(default: 2025-01-01)')
    parser.add_argument('--test_start', type=str, default="2026-01-01",
        help='First date of test period (default: 2026-01-01)')
    parser.add_argument('--overwrite', type=int, default=0,
        help='If existing sample files should be removed (default: 0 (false))')
    args = parser.parse_args()

    dataset_dir_path = os.path.join("data", args.dataset)
    static_dir_path = os.path.join(dataset_dir_path, "static")
    samples_dir_path = os.path.join(dataset_dir_path, "samples")

    single_level_path = os.path.join(dataset_dir_path, "single_level", "data.grib")
    grib_paths = [single_level_path] + sorted(glob.glob(os.path.join(
        dataset_dir_path, "pressure", "*", "data.grib")))

    # Make sure that no old samples are mixed with the new ones
    splits = ("train", "val", "test")
    for split in splits:
        old_paths = glob.glob(os.path.join(samples_dir_path, split, "*.npy"))
        if old_paths:
            assert args.overwrite, (f"Samples already exist in "
                f"{os.path.join(samples_dir_path, split)}, use --overwrite 1 to replace")
            for path in old_paths:
                os.remove(path)
        os.makedirs(os.path.join(samples_dir_path, split), exist_ok=True)
    os.makedirs(static_dir_path, exist_ok=True)

    # -- Load data --
    keys = [grib_key(name) for name in constants.param_names]
    times, lats, lons = scan_times(grib_paths, keys)
    grid_shape = (len(lats), len(lons)) # (N_y, N_x)
    assert grid_shape == constants.grid_shape, (
            f"Grid shape {grid_shape} does not match constants.grid_shape")
    print(f"Found {len(times)} time steps from {times[0]} to {times[-1]}, "
            f"on grid of shape {grid_shape}")

    fields = load_fields(grib_paths, keys, times,
            grid_shape) # (N_t, N_y, N_x, d_features)
    assert np.isfinite(fields).all(), "Non-finite values in data"

    # -- Static files --
    print("Saving static files...")
    grid_xy = np.stack(np.meshgrid(lons, lats)).astype(np.float32) # (2, N_y, N_x)
    np.save(os.path.join(static_dir_path, "nwp_xy.npy"), grid_xy)

    geopotential_path = os.path.join(static_dir_path, "surface_geopotential.npy")
    if not os.path.isfile(geopotential_path):
        print("Downloading surface geopotential...")
        np.save(geopotential_path, download_surface_geopotential(lats, lons))

    border_mask = np.ones(grid_shape, dtype=bool) # (N_y, N_x)
    border_mask[args.border_width:-args.border_width,
            args.border_width:-args.border_width] = False
    np.save(os.path.join(static_dir_path, "border_mask.npy"), border_mask)

    # -- Samples --
    water_cover = 1. - load_land_sea_mask(single_level_path) # (N_y, N_x)
    flux = toa_solar_flux(times, lats, lons) # (N_t, N_y, N_x)

    val_start = dt.datetime.strptime(args.val_start, "%Y-%m-%d")
    test_start = dt.datetime.strptime(args.test_start, "%Y-%m-%d")
    split_limits = {
        "train": (times[0], val_start),
        "val": (val_start, test_start),
        "test": (test_start, times[-1] + dt.timedelta(hours=constants.raw_step_hours)),
    }

    times_np = np.array(times)
    for split in splits:
        split_start, split_end = split_limits[split]
        # Only use samples that lie entirely within the period of the split
        split_time_i = np.nonzero((times_np >= split_start) & (times_np < split_end))[0]
        n_split_times = len(split_time_i)
        sample_starts = split_time_i[:max(n_split_times -
            constants.sample_length_raw + 1, 0):args.stride]

        split_dir_path = os.path.join(samples_dir_path, split)
        for start_i in tqdm(sample_starts, desc=f"Saving {split} samples"):
            sample_slice = slice(start_i, start_i + constants.sample_length_raw)
            # First time step is at time stamp + offset
            sample_time = times[start_i] - dt.timedelta(
                    hours=constants.sample_offset_hours)
            time_str = sample_time.strftime("%Y%m%d%H")

            np.save(os.path.join(split_dir_path, f"nwp_{time_str}_mbr000.npy"),
                    fields[sample_slice]) # (N_t', N_y, N_x, d_features)
            np.save(os.path.join(split_dir_path,
                f"nwp_toa_downwelling_shortwave_flux_{time_str}.npy"),
                flux[sample_slice]) # (N_t', N_y, N_x)
            np.save(os.path.join(split_dir_path, f"wtr_{time_str}.npy"),
                    water_cover) # (N_y, N_x)

if __name__ == "__main__":
    main()

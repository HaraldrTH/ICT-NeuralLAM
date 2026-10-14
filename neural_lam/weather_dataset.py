import os
import torch
import numpy as np
import xarray as xr

from neural_lam import utils, constants

SOLAR_CONSTANT = 1361. # W/m^2

def open_store(dataset_name):
    """
    Open zarr store of dataset lazily (no data is loaded)
    """
    store_path = os.path.join("data", dataset_name, constants.zarr_name)
    return xr.open_dataset(store_path, engine="zarr", chunks=None)

def toa_solar_radiation(times, lat, lon):
    """
    Instantaneous downwelling shortwave flux at top of atmosphere (W/m^2)
    times: (N_t,) datetime64, UTC
    lat, lon: (N_grid,) in degrees
    returns: (N_t, N_grid)
    """
    year_start = times.astype("datetime64[Y]")
    days_in_year = ((year_start + np.timedelta64(1, "Y")).astype("datetime64[D]")
            - year_start.astype("datetime64[D]")) / np.timedelta64(1, "D")
    day_of_year = (times.astype("datetime64[D]") - year_start.astype("datetime64[D]")
            ) / np.timedelta64(1, "D") # 0-based
    hour = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")

    # Fractional year, eq. of time and declination (Spencer, 1971)
    gamma = 2*np.pi/days_in_year * (day_of_year + (hour - 12)/24) # (N_t,)
    eq_time = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
            - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma)) # minutes
    declination = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma)) # radians
    distance_factor = (1.000110 + 0.034221*np.cos(gamma) + 0.001280*np.sin(gamma)
            + 0.000719*np.cos(2*gamma) + 0.000077*np.sin(2*gamma))

    solar_time = hour[:,None]*60 + eq_time[:,None] + 4*lon[None,:] # (N_t, N_grid)
    hour_angle = np.deg2rad(solar_time/4 - 180)
    lat_rad = np.deg2rad(lat)[None,:]
    cos_zenith = (np.sin(lat_rad)*np.sin(declination[:,None]) +
            np.cos(lat_rad)*np.cos(declination[:,None])*np.cos(hour_angle))

    return (SOLAR_CONSTANT*distance_factor[:,None]*np.clip(cos_zenith, 0, None)
            ).astype(np.float32)

def datetime_features(times):
    """
    Time of day and year, encoded as sin/cos and rescaled to [0,1]
    times: (N_t,) datetime64, UTC
    returns: (N_t, 4)
    """
    hour = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")
    year_start = times.astype("datetime64[Y]")
    year_length = ((year_start + np.timedelta64(1, "Y")).astype("datetime64[D]")
            - year_start.astype("datetime64[D]"))
    year_fraction = (times - year_start.astype("datetime64[D]")) / year_length

    hour_angle = torch.tensor((hour/12)*np.pi, dtype=torch.float32)
    year_angle = torch.tensor(year_fraction*2*np.pi, dtype=torch.float32)
    datetime_forcing = torch.stack((
        torch.sin(hour_angle),
        torch.cos(hour_angle),
        torch.sin(year_angle),
        torch.cos(year_angle),
        ), dim=1) # (N_t, 4)
    return (datetime_forcing + 1)/2 # Rescale to [0,1]

class WeatherDataset(torch.utils.data.Dataset):
    """
    ERA5 data read from zarr store data/<dataset_name>/era5.zarr

    For our dataset:
    N_t = pred_length + 2 (= 21 by default)
    N_x = 115
    N_y = 106
    N_grid = 106x115 = 12190
    d_features = len(constants.param_names) (= 27)
    d_forcing = 5

    subsample_step: number of data time steps (6 h) per model time step
    """
    def __init__(self, dataset_name, pred_length=19, split="train", subsample_step=1,
            standardize=True, subset=False, control_only=False):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.dataset_name = dataset_name
        self.sample_length = pred_length + 2 # 2 init states
        self.subsample_step = subsample_step
        self._ds = None # Opened lazily in each worker process

        ds = open_store(dataset_name)
        self.times = ds.time.values # (N_t_total,)

        store_vars = list(ds.attrs["state_vars"])
        missing_vars = [name for name in constants.param_names
                if name not in store_vars]
        assert not missing_vars, f"Variables not in zarr store: {missing_vars}"
        self.var_index = np.array([store_vars.index(name)
            for name in constants.param_names])

        lon_grid, lat_grid = np.meshgrid(ds.longitude.values, ds.latitude.values)
        self.grid_lat = lat_grid.flatten() # (N_grid,)
        self.grid_lon = lon_grid.flatten() # (N_grid,)

        # Open water, static over time but given as batch static feature
        self.open_water = torch.tensor(1. - ds.lsm.values,
                dtype=torch.float32).flatten().unsqueeze(-1) # (N_grid, 1)
        ds.close()

        # Find all start indices where the full sample lies within the split
        # and the time steps are contiguous
        split_first, split_last = (np.datetime64(t, "ns")
                for t in constants.split_times[split])
        span = (self.sample_length - 1)*subsample_step
        start_candidates = np.arange(max(len(self.times) - span, 0))
        start_times = self.times[start_candidates]
        end_times = self.times[start_candidates + span]
        valid = (start_times >= split_first) & (end_times <= split_last) &\
                (end_times - start_times ==
                        np.timedelta64(span*constants.data_step_hours, "h"))
        if split != "train":
            # Only start forecasts at fixed times of day in val/test
            init_times = self.times[np.minimum(start_candidates + subsample_step,
                len(self.times) - 1)]
            init_hours = (init_times - init_times.astype("datetime64[D]")
                    ) // np.timedelta64(1, "h")
            valid &= np.isin(init_hours, constants.eval_init_hours)
        self.start_indices = start_candidates[valid]

        if subset:
            self.start_indices = self.start_indices[:50] # Limit to 50 samples
        assert len(self.start_indices) > 0, \
                f"No samples of length {self.sample_length} in {split} split"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std, self.flux_mean, self.flux_std =\
                ds_stats["data_mean"], ds_stats["data_std"], ds_stats["flux_mean"], \
                ds_stats["flux_std"]

    def __getstate__(self):
        # Do not pickle open zarr store when sending dataset to worker processes
        state = self.__dict__.copy()
        state["_ds"] = None
        return state

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        if self._ds is None:
            self._ds = open_store(self.dataset_name)

        # === Sample ===
        start_index = self.start_indices[idx]
        time_slice = slice(start_index,
                start_index + (self.sample_length - 1)*self.subsample_step + 1,
                self.subsample_step)
        sample_times = self.times[time_slice] # (N_t,)

        sample = self._ds["state"].isel(time=time_slice).values[...,
                self.var_index] # (N_t, N_y, N_x, d_features)
        sample = torch.tensor(sample, dtype=torch.float32).flatten(1,2
                ) # (N_t, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean)/self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2] # (2, N_grid, d_features)
        target_states = sample[2:] # (sample_length-2, N_grid, d_features)

        # === Static batch features ===
        static_features = self.open_water # (N_grid, 1)

        # === Forcing features ===
        flux = torch.tensor(toa_solar_radiation(sample_times, self.grid_lat,
            self.grid_lon)).unsqueeze(-1) # (N_t, N_grid, 1)
        if self.standardize:
            flux = (flux - self.flux_mean)/self.flux_std

        datetime_forcing = datetime_features(sample_times).unsqueeze(1).expand(-1,
                flux.shape[1], -1) # (N_t, N_grid, 4)

        # Put forcing features together
        forcing_features = torch.cat((flux, datetime_forcing),
                dim=-1) # (N_t, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat((
            forcing_features[:-2],
            forcing_features[1:-1],
            forcing_features[2:],
            ), dim=2) # (N_t-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        return init_states, target_states, static_features, forcing_windowed

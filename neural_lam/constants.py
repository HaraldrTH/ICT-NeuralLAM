import cartopy
import numpy as np

wandb_project = "neural-lam"

# Log prediction error for these lead times
val_step_log_errors = np.array([1, 2, 3, 5, 10, 15, 19])

# Data
zarr_name = "era5.zarr" # Name of zarr store in data/<dataset>/
data_step_hours = 6 # Hours between time steps in data
# Val and test are each a full year, so all seasons are covered, ending at the
# last time in the data (2026-09-18T12). Update if more data is downloaded.
split_times = { # (first, last) time of each split, inclusive
    "train": ("2022-01-01T00", "2024-09-17T18"),
    "val": ("2024-09-18T00", "2025-09-17T18"),
    "test": ("2025-09-18T00", "2026-09-18T12"),
}
eval_init_hours = (0, 12) # Hours of day to start forecasts from in val/test

# Variables, (name in zarr store, unit, loss weight)
# Loss weights loosely follow GraphCast: t2m 1.0, other surface fields 0.1,
# pressure level fields 0.1 * level/1000
param_table = [
    ("msl", "Pa", 0.1),
    ("u10", "m/s", 0.1),
    ("v10", "m/s", 0.1),
    ("t2m", "K", 1.0),
    ("d2m", "K", 0.1),
    ("sst", "K", 0.1),
    ("tp", "m", 0.1),
] + [
    (f"{var}_{level}", unit, 0.1*level/1000)
    for var, unit in (
        ("z", "m$^2$/s$^2$"),
        ("t", "K"),
        ("u", "m/s"),
        ("v", "m/s"),
        ("q", "kg/kg"),
    )
    for level in (500, 700, 850, 925)
]

param_names = [name for name, _, _ in param_table]
param_names_short = param_names
param_units = [unit for _, unit, _ in param_table]
param_weights = [weight for _, _, weight in param_table]

# Projection and grid
grid_shape = (106, 115) # (y, x), (latitude, longitude)

grid_limits = [ # In projection, outer edges of grid cells
    2.875, # min x (longitude)
    31.625, # max x
    54.375, # min y (latitude)
    80.875, # max y
]

# Data is on regular lat-lon grid
data_proj = cartopy.crs.PlateCarree()

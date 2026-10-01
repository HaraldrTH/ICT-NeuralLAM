import cartopy
import numpy as np

wandb_project = "neural-lam"

seconds_in_year = 365*24*60*60 # Ignoring leap years

# Log prediction error for these lead times
val_step_log_errors = np.array([1, 2, 3, 5, 10, 15, 19])

# Time structure of the sample files
sample_length_raw = 21 # N_t', number of time steps stored in each sample file
raw_step_hours = 6 # Hours between consecutive time steps in the sample files
sample_offset_hours = 0 # Hours from time stamp in file name to first time step

# Variable names
pressure_levels = (500, 700, 850, 925)
param_names = [
    'msl_meanSea_0_instant',
    't_heightAboveGround_2_instant',
    'd_heightAboveGround_2_instant',
    'u_heightAboveGround_10_instant',
    'v_heightAboveGround_10_instant',
] + [f'{var}_isobaricInhPa_{level}_instant'
    for var in ('z', 't', 'u', 'v', 'q') for level in pressure_levels]

param_names_short = [
    'msl_0',
    't_2',
    'd_2',
    'u_10',
    'v_10',
] + [f'{var}_{level}'
    for var in ('z', 't', 'u', 'v', 'q') for level in pressure_levels]

param_units = [
    'Pa',
    'K',
    'K',
    'm/s',
    'm/s',
] + [unit
    for unit in (
        'm\\textsuperscript{2}/s\\textsuperscript{2}',
        'K',
        'm/s',
        'm/s',
        'kg/kg',
    ) for _ in pressure_levels]

# Projection and grid
# TODO Do not hard code this, make part of static dataset files
grid_shape = (106, 115) # (y, x) = (lat, lon)

grid_limits = [ # In projection (outer edges of the 0.25 degree grid cells)
    2.875, # min x (lon)
    31.625, # max x (lon)
    54.375, # min y (lat)
    80.875, # max y (lat)
]

# Create projection (the grid is a regular lat-lon grid)
map_proj = cartopy.crs.PlateCarree()

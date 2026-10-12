import argparse
import xarray as xr   # pip install xarray cfgrib
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--in_path", default="data/single_level/data.grib")
parser.add_argument("-o","--out_path", default="data/single_level/data.csv")
parser.add_argument("-a", "--read_all_samples", action="store_true")

args = parser.parse_args()

df = pd.DataFrame()

# Scalar coords cfgrib adds that we don't want as CSV columns
EXTRA = ["number", "step", "surface", "valid_time",
         "heightAboveGround", "meanSea", "depthBelowLandLayer"]

# Atmospheric, instantaneous: u10, v10, d2m, t2m, msl, sst, lsm
atm = xr.open_dataset(args.in_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stream": "oper", "stepType": "instant"}})
# Wave model: mwd, mwp, swh (0.5° grid)
wave = xr.open_dataset(args.in_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stream": "wave"}})

atm = atm.drop_vars(EXTRA, errors="ignore")
wave = wave.drop_vars(EXTRA, errors="ignore")

# Put wave data on the atmospheric grid (nearest 0.5° cell)
wave_on_atm = wave.reindex_like(atm, method="nearest", tolerance=0.25)

ds = xr.merge([atm, wave_on_atm])

df = ds.to_dataframe()

if not args.read_all_samples:
    df = df.head(2)

df.to_csv(args.out_path)
"""
Convert downloaded IFS runs to a forecast cache that the evaluation code reads.

Reads data/ifs/nordic/ifs_YYYYMMDDHH.npz (written by ifs/download_ifs.py) and
writes data/ifs/ifs_hres_<split>.zarr, with the same layout as the model
caches in evaluation_results/predictions:

    prediction             IFS forecast, ERA5 variable names and units
    target                 ERA5 at the valid time (data/era5/era5.zarr)
    latest_observed_state  ERA5 at the run start (used for persistence)

Conversions to match the ERA5 variables:
    z_<level>   geopotential height (gh) * g
    tp          ERA5 tp at 6 h samples is a 1 h accumulation. IFS tp is
                accumulated from the run start, so the mean hourly rate over the
                last 3 h is used: (tp(L) - tp(L-3)) / 3. This is an
                approximation; treat tp scores with care.
    sst         not in the open data, so not included
Cells in the boundary mask are set to ERA5, as is done for the models; these
cells are excluded from all scores.

Variables that are missing from any downloaded run are left out of the cache.

Usage (from the repository root):
    python ifs/preprocess_ifs.py                 # test split
    python ifs/preprocess_ifs.py --overwrite     # replace an existing store
"""

import glob
import os
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_lam import constants  # noqa: E402
from neural_lam.evaluation.forecast_cache import ZarrForecastCache  # noqa: E402

GRAVITY = 9.80665  # m/s^2, ECMWF value
SST_LAND_VALUE = 271.46  # K, same fill value as prepare_era5.py
TP_OFFSET_HOURS = 3

# ERA5 variable -> IFS open data field
FIELD_NAMES = {
    "msl": "msl",
    "u10": "10u",
    "v10": "10v",
    "t2m": "2t",
    "d2m": "2d",
    "sst": "sst",
    "tp": "tp",
    **{f"{var}_{level}": f"{'gh' if var == 'z' else var}_{level}"
        for var in ("z", "t", "u", "v", "q") for level in (500, 700, 850, 925)},
}


def parse_args():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ifs_dir", default=os.path.join("data", "ifs"),
            help="Directory with nordic/*.npz from download_ifs.py")
    parser.add_argument("--dataset", default="era5",
            help="Dataset in data/ used as truth (default: era5)")
    parser.add_argument("--split", default="test", choices=list(constants.split_times),
            help="Only use runs starting in this split (default: test)")
    parser.add_argument("--out", default=None,
            help="Output store (default: <ifs_dir>/ifs_hres_<split>.zarr)")
    parser.add_argument("--variables", nargs="+", default=None,
            help="ERA5 variables to include (default: all available in IFS)")
    parser.add_argument("--exclude_tp", action="store_true",
            help="Leave tp out of the cache (its accumulation period differs from ERA5)")
    parser.add_argument("--overwrite", action="store_true",
            help="Replace the output store if it exists")
    return parser.parse_args()


def field_steps(run, field):
    return run[f"{field}__steps"].astype(np.int64)


def field_at(run, field, hours):
    """Field values (len(hours), N_y, N_x) at the given steps, None if missing."""
    steps = field_steps(run, field)
    index = [np.flatnonzero(steps == h) for h in hours]
    if any(len(i) == 0 for i in index):
        return None
    return run[field][[int(i[0]) for i in index]]


def convert(run, variable, leads):
    """IFS forecast of one ERA5 variable at the leads, in ERA5 units."""
    field = FIELD_NAMES[variable]
    if variable == "tp":
        current = field_at(run, field, leads)
        previous = field_at(run, field, [lead - TP_OFFSET_HOURS for lead in leads])
        if current is None or previous is None:
            return None
        return np.clip((current - previous) / TP_OFFSET_HOURS, 0.0, None)

    values = field_at(run, field, leads)
    if values is None:
        return None
    if variable.startswith("z_"):
        values = values * GRAVITY
    if variable == "sst":
        values = np.where(np.isfinite(values), values, SST_LAND_VALUE)
    return values


def main():
    args = parse_args()
    out_path = args.out or os.path.join(args.ifs_dir, f"ifs_hres_{args.split}.zarr")
    if os.path.exists(out_path):
        if not args.overwrite:
            raise SystemExit(f"{out_path} exists, pass --overwrite to replace it")
        shutil.rmtree(out_path)

    run_paths = sorted(glob.glob(os.path.join(args.ifs_dir, "nordic", "ifs_*.npz")))
    if not run_paths:
        raise SystemExit(f"No runs in {args.ifs_dir}/nordic, run download_ifs.py first")

    # Truth data
    data_dir = os.path.join("data", args.dataset)
    era5 = xr.open_zarr(os.path.join(data_dir, constants.zarr_name))
    era5_vars = list(era5.attrs["state_vars"])
    era5_times = pd.DatetimeIndex(era5.time.values)
    time_index = {int(t.value // 10**9): i for i, t in enumerate(era5_times)}
    latitude = era5.latitude.values
    longitude = era5.longitude.values
    land_sea_mask = era5.lsm.values
    boundary_mask = np.load(os.path.join(data_dir, "static", "border_mask.npy"))

    split_start, split_end = (pd.Timestamp(t, tz="UTC")
            for t in constants.split_times[args.split])

    # Select runs in the split and work out leads and variables
    runs = []
    for path in run_paths:
        with np.load(path) as run:
            init = int(run["init_time_unix_s"])
            fields = set(run["field_names"].tolist())
            steps = field_steps(run, "msl") if "msl" in fields else None
        if split_start <= pd.Timestamp(init, unit="s", tz="UTC") <= split_end:
            runs.append((init, path, fields, steps))
    if not runs:
        raise SystemExit(f"No downloaded runs start in the {args.split} split")

    leads = [int(h) for h in runs[0][3] if h > 0 and h % constants.data_step_hours == 0]
    common_fields = set.intersection(*(fields for _, _, fields, _ in runs))
    candidates = args.variables or constants.param_names
    variables = [v for v in constants.param_names if v in candidates
            and FIELD_NAMES[v] in common_fields and v in era5_vars
            and not (v == "tp" and args.exclude_tp)]
    skipped_vars = [v for v in candidates if v not in variables]
    if skipped_vars:
        print(f"Not included: {', '.join(skipped_vars)}")
    var_index = [era5_vars.index(v) for v in variables]
    units = [constants.param_units[constants.param_names.index(v)] for v in variables]
    print(f"{len(runs)} runs in {args.split} split, leads {leads[0]}-{leads[-1]} h, "
            f"{len(variables)} variables")

    cache = ZarrForecastCache(
        out_path,
        model_id="IFS-HRES",
        checkpoint_id="ecmwf-opendata-ifs-0p25",
        dataset=args.dataset,
        split=args.split,
        graph=None,
        config_path=None,
        step_length_hours=constants.data_step_hours,
        lead_hours=leads,
        variable_names=variables,
        variable_units=units,
        latitude=latitude,
        longitude=longitude,
        land_sea_mask=land_sea_mask,
        boundary_mask=boundary_mask,
    )
    cache.group.attrs.update({
        "source": "ECMWF open data, IFS HRES 0.25 deg (ecmwf-opendata)",
        "truth": f"data/{args.dataset}/{constants.zarr_name}",
        "tp_definition": "mean hourly rate over the last 3 h, (tp(L) - tp(L-3)) / 3",
        "z_definition": f"gh * {GRAVITY}",
    })

    lead_seconds = np.asarray(leads, dtype=np.int64) * 3600
    written, skipped = 0, []
    for init, path, _, _ in runs:
        stamp = os.path.basename(path)[4:-4]
        valid = init + lead_seconds
        if init not in time_index or any(int(t) not in time_index for t in valid):
            skipped.append(f"{stamp} (valid times outside {args.dataset} data)")
            continue

        with np.load(path) as run:
            if not (np.allclose(run["latitude"], latitude, atol=1e-3)
                    and np.allclose(run["longitude"], longitude, atol=1e-3)):
                raise ValueError(f"Grid of {path} does not match {args.dataset}")
            fields = [convert(run, v, leads) for v in variables]
        if any(f is None for f in fields):
            skipped.append(f"{stamp} (missing lead times)")
            continue
        prediction = np.stack(fields, axis=-1).astype(np.float32)  # (L, N_y, N_x, d)

        target = era5.state.isel(time=[time_index[int(t)] for t in valid]).values
        target = target[..., var_index].astype(np.float32)
        latest = era5.state.isel(time=time_index[init]).values[..., var_index]
        prediction[:, boundary_mask] = target[:, boundary_mask]

        if not np.isfinite(prediction).all():
            skipped.append(f"{stamp} (non-finite IFS values)")
            continue

        n_leads, ny, nx, nvars = prediction.shape
        cache.append(
            prediction.reshape(1, n_leads, ny * nx, nvars),
            target.reshape(1, n_leads, ny * nx, nvars),
            latest.astype(np.float32).reshape(1, ny * nx, nvars),
            {"initialization_time_unix_s": np.array([init]),
                "valid_time_unix_s": valid[None, :]},
        )
        written += 1
        if written % 50 == 0:
            print(f"\t{written} runs written", flush=True)

    for message in skipped:
        print(f"Skipped {message}")
    cache.finish()
    print(f"Wrote {written} runs to {out_path}")


if __name__ == "__main__":
    main()

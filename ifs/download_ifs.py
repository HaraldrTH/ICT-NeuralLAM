"""
Download ECMWF IFS HRES forecasts (open data, 0.25 deg) for the ERA5 Nordic grid.

Forecasts are read from the ECMWF open data archive, which keeps past runs.
It is mirrored on Google Cloud (default) and AWS with the same file layout;
use --source to pick one (the AWS bucket often answers 503 SlowDown). Only the
needed GRIB messages are fetched, using the byte offsets in the .index file next
to each forecast step file. The fields are cropped to the ERA5
grid in data/era5/static/nwp_xy.npy and each run is saved as one compressed
.npz file:

    data/ifs/nordic/ifs_YYYYMMDDHH.npz

The global GRIB file of a run (about 350 MB for leads up to 114 h) is deleted
after cropping unless --keep_grib is given. The script can be stopped (Ctrl-C)
and restarted at any time: runs that already have an .npz are skipped, and the
steps of an unfinished run are kept in data/ifs/grib/ifs_YYYYMMDDHH/ so only
the missing steps are downloaded again.

The open data has no sea surface temperature (sst), so IFS is not scored on sst.

Usage (from the repository root):
    python ifs/download_ifs.py                        # test split, 00/06/12/18 UTC runs
    python ifs/download_ifs.py --start 2025-09-18 --end 2025-10-18
"""

import json
import os
import random
import shutil
import sys
import threading
import time
import warnings
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_lam import constants  # noqa: E402

SOURCES = {
    "google": "https://storage.googleapis.com/ecmwf-open-data",
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
}
# Throttling and server errors are retried with backoff, up to MAX_WAIT_S each
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_WAIT_S = 120
# Set on Ctrl-C, so the worker threads stop between requests
STOP = threading.Event()
# 06/18 UTC runs were published as stream "scda" before this date, "oper" after
SCDA_END = datetime(2026, 5, 12, tzinfo=timezone.utc)

# Open data names (shortName in GRIB) of the fields needed for the ERA5 variables
SURFACE_PARAMS = ["msl", "10u", "10v", "2t", "2d", "tp"]
PRESSURE_PARAMS = ["gh", "t", "u", "v", "q"]
PRESSURE_LEVELS = [500, 700, 850, 925]
# ERA5 tp is a 1 h accumulation, so tp is also fetched 3 h before each lead
# (the finest open data step) to compute a mean hourly rate in preprocessing
TP_OFFSET_HOURS = 3


def parse_args():
    test_start, test_end = constants.split_times["test"]
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=test_start[:10],
            help="First run date, YYYY-MM-DD (default: start of test split)")
    parser.add_argument("--end", default=test_end[:10],
            help="Last run date, YYYY-MM-DD, inclusive (default: end of test split)")
    parser.add_argument("--init_hours", type=int, nargs="+", default=[0, 6, 12, 18],
            help="Run hours (UTC), all reach 144 h (default: 0 6 12 18, as "
            "eval_init_hours in constants.py)")
    parser.add_argument("--max_lead", type=int, default=114,
            help="Longest lead time in hours (default: 114, as the model caches)")
    parser.add_argument("--step_hours", type=int, default=constants.data_step_hours,
            help="Hours between lead times (default: 6)")
    parser.add_argument("--every_n_days", type=int, default=1,
            help="Only download every n-th day, to reduce download size")
    parser.add_argument("--out_dir", default=os.path.join("data", "ifs"),
            help="Output directory (default: data/ifs)")
    parser.add_argument("--static_dir", default=os.path.join("data", "era5", "static"),
            help="Directory with nwp_xy.npy for the target grid")
    parser.add_argument("--source", choices=sorted(SOURCES), default="google",
            help="Open data mirror (default: google)")
    parser.add_argument("--connections", type=int, default=8,
            help="Parallel HTTP requests per run (default: 8)")
    parser.add_argument("--keep_grib", action="store_true",
            help="Keep the global GRIB files in <out_dir>/grib")
    return parser.parse_args()


def target_grid(static_dir):
    """Ascending latitudes and longitudes of the ERA5 grid."""
    xy = np.load(os.path.join(static_dir, "nwp_xy.npy"))  # (2, N_y, N_x): lon, lat
    return xy[1, :, 0].astype(np.float64), xy[0, 0, :].astype(np.float64)


def match_indices(source, target, name):
    """Index in source of each target coordinate (exact grid points)."""
    index = np.abs(source[None, :] - target[:, None]).argmin(axis=1)
    if not np.allclose(source[index], target, atol=1e-3):
        raise ValueError(f"IFS {name} grid does not contain the ERA5 {name} points")
    return index


def get(session, url, byte_range=None, retries=10):
    """GET with retries on throttling and network errors. None if not found."""
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    status = None
    for attempt in range(retries):
        wait = min(MAX_WAIT_S, 2 ** attempt)
        try:
            response = session.get(url, headers=headers, timeout=60)
            status = response.status_code
            if status in (403, 404):
                return None
            if status in (200, 206):
                return response.content
            if status not in RETRY_STATUS:
                break
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = min(MAX_WAIT_S, max(wait, int(retry_after)))
        except requests.RequestException:
            pass
        if STOP.is_set():
            break
        # Jitter, so the parallel requests do not all retry at the same moment
        time.sleep(wait * random.uniform(0.5, 1.5))
    raise RuntimeError(f"Failed to download {url} (last status {status})")


def step_url(base_url, init, step):
    stream = "scda" if init.hour in (6, 18) and init < SCDA_END else "oper"
    stamp = init.strftime("%Y%m%d%H%M%S")
    return (f"{base_url}/{init:%Y%m%d}/{init:%H}z/ifs/0p25/{stream}/"
            f"{stamp}-{step}h-{stream}-fc")


def wanted_messages(index_text, params, levels):
    """(offset, length, name) of the wanted messages in a step file."""
    found = []
    for line in index_text.splitlines():
        entry = json.loads(line)
        param = entry["param"]
        if entry.get("levtype") == "sfc" and param in params["sfc"]:
            found.append((entry["_offset"], entry["_length"], param))
        elif (entry.get("levtype") == "pl" and param in params["pl"]
                and int(entry["levelist"]) in levels):
            found.append((entry["_offset"], entry["_length"],
                f"{param}_{entry['levelist']}"))
    return found


def crop_grib(path, target_lat, target_lon):
    """Read a GRIB file and crop all fields. Returns {name: (steps, array)}."""
    import cfgrib

    fields = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    for ds in datasets:
        lon = ((ds.longitude.values + 180.0) % 360.0) - 180.0
        ilat = match_indices(ds.latitude.values, target_lat, "latitude")
        ilon = match_indices(lon, target_lon, "longitude")
        for var in ds.data_vars.values():
            short_name = var.attrs.get("GRIB_shortName", var.name)
            if "step" not in var.dims:
                var = var.expand_dims("step")
            steps = np.atleast_1d(var["step"].values / np.timedelta64(1, "h"))
            levels = (np.atleast_1d(var["isobaricInhPa"].values)
                    if "isobaricInhPa" in var.coords else [None])
            for level in levels:
                level_var = var
                if level is not None and "isobaricInhPa" in var.dims:
                    level_var = var.sel(isobaricInhPa=level)
                values = level_var.transpose("step", "latitude", "longitude").values
                values = values[:, ilat][:, :, ilon].astype(np.float32)
                name = short_name if level is None else f"{short_name}_{int(level)}"
                fields[name] = (steps.astype(np.int32), values)
        ds.close()
    return fields


def download_run(init, args, lat, lon, leads, pool, session):
    stamp = init.strftime("%Y%m%d%H")
    out_path = os.path.join(args.out_dir, "nordic", f"ifs_{stamp}.npz")
    if os.path.exists(out_path):
        return f"{stamp}: exists, skipped"

    tp_steps = sorted({lead - TP_OFFSET_HOURS for lead in leads} - set(leads))
    requests_by_step = {lead: {"sfc": SURFACE_PARAMS, "pl": PRESSURE_PARAMS}
            for lead in leads}
    for step in tp_steps:
        requests_by_step[step] = {"sfc": ["tp"], "pl": []}

    # Each step is saved to its own file in step_dir as soon as it is complete,
    # so an interrupted or failed run continues from the steps it already has
    grib_dir = os.path.join(args.out_dir, "grib")
    step_dir = os.path.join(grib_dir, f"ifs_{stamp}")
    os.makedirs(step_dir, exist_ok=True)
    step_path = lambda step: os.path.join(step_dir, f"step_{step:03d}.grib2")
    base_url = SOURCES[args.source]

    def download_step(step):
        """Fetch the wanted messages of one step. Returns the number fetched."""
        path = step_path(step)
        if os.path.exists(path):
            return 0
        url = step_url(base_url, init, step)
        index = get(session, url + ".index")
        if index is None:
            raise RuntimeError(f"step {step} h not in archive")
        messages = wanted_messages(index.decode(), requests_by_step[step],
                PRESSURE_LEVELS)
        part_path = path + ".part"
        with open(part_path, "wb") as out:
            for offset, length, _ in messages:
                if STOP.is_set():
                    raise RuntimeError("stopped")
                data = get(session, url + ".grib2", (offset, offset + length - 1))
                if data is None or len(data) != length:
                    raise RuntimeError(f"incomplete message from {url}.grib2")
                out.write(data)
        os.replace(part_path, path)
        return len(messages)

    n_fetched = sum(pool.map(download_step, sorted(requests_by_step)))

    grib_path = os.path.join(grib_dir, f"ifs_{stamp}.grib2")
    try:
        with open(grib_path, "wb") as out:
            for step in sorted(requests_by_step):
                with open(step_path(step), "rb") as step_file:
                    shutil.copyfileobj(step_file, out)
        fields = crop_grib(grib_path, lat, lon)
        expected = set(SURFACE_PARAMS) | {f"{p}_{level}" for p in PRESSURE_PARAMS
                for level in PRESSURE_LEVELS}
        missing = expected - set(fields)

        arrays = {"latitude": lat.astype(np.float32),
                "longitude": lon.astype(np.float32),
                "init_time_unix_s": np.int64(init.timestamp()),
                "field_names": np.array(sorted(fields)),
                "missing": np.array(sorted(missing), dtype=str)}
        for name, (steps, values) in fields.items():
            arrays[name] = values
            arrays[f"{name}__steps"] = steps
        tmp_path = out_path[:-len(".npz")] + ".tmp.npz"
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_path)
        shutil.rmtree(step_dir)
    finally:
        if not args.keep_grib and os.path.exists(grib_path):
            os.remove(grib_path)

    note = f" (not in open data: {', '.join(sorted(missing))})" if missing else ""
    return f"{stamp}: {len(fields)} fields, {n_fetched} messages fetched{note}"


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "nordic"), exist_ok=True)
    lat, lon = target_grid(args.static_dir)
    leads = list(range(args.step_hours, args.max_lead + 1, args.step_hours))

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    inits = []
    day = start
    while day <= end:
        inits += [day + timedelta(hours=h) for h in sorted(args.init_hours)]
        day += timedelta(days=args.every_n_days)
    print(f"Downloading {len(inits)} IFS runs, leads {leads[0]}-{leads[-1]} h, "
            f"from {args.source} to {args.out_dir}/nordic")

    failed = []
    session = requests.Session()
    # One pooled connection per worker thread
    adapter = requests.adapters.HTTPAdapter(pool_connections=1,
            pool_maxsize=args.connections)
    session.mount("https://", adapter)
    with session, ThreadPoolExecutor(max_workers=args.connections) as pool:
        for init in inits:
            try:
                print(download_run(init, args, lat, lon, leads, pool, session),
                        flush=True)
            except KeyboardInterrupt:
                # Let the workers drop their unfinished steps and exit quickly
                STOP.set()
                pool.shutdown(wait=True, cancel_futures=True)
                print(f"{init:%Y%m%d%H}: stopped, re-run the script to resume")
                sys.exit(130)
            except Exception as exc:
                failed.append(init)
                print(f"{init:%Y%m%d%H}: FAILED ({exc})", flush=True)
                # Back off before the next run in case the source is throttling
                time.sleep(MAX_WAIT_S)

    if failed:
        print(f"{len(failed)} runs failed, re-run the script to retry them:")
        print(" ".join(f"{init:%Y%m%d%H}" for init in failed))


if __name__ == "__main__":
    main()

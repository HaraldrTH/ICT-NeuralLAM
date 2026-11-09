"""
Score IFS HRES and the trained models on the same forecasts and plot them.

All systems are scored against ERA5 with the offline evaluator
(neural_lam.evaluation.offline_metrics), restricted to what they have in
common: the same initialization times, the same variables and the same lead
times. Runs that failed to download, or run hours left out with
download_ifs.py --init_hours, are then also left out for the models.

Outputs in --output_dir (default evaluation_results/metrics/ifs_comparison):
    <model>_<split>_matched_metrics.csv   one per system, offline_metrics format
    comparison_metrics.csv                all systems in one table
    comparison_metadata.json              caches, initializations, variables
    <metric>_by_lead.png                  per-variable curves vs lead time

Usage (from the repository root):
    python ifs/evaluate_ifs.py
    python ifs/evaluate_ifs.py --model_caches evaluation_results/predictions/HI-LAM_test_...zarr
"""

import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_lam import constants  # noqa: E402
from neural_lam.evaluation.config import load_evaluation_config  # noqa: E402
from neural_lam.evaluation.forecast_cache import open_forecast_cache  # noqa: E402
from neural_lam.evaluation.offline_metrics import (  # noqa: E402
    _slug,
    evaluate_forecast_cache,
)
from neural_lam.evaluation.results_io import (  # noqa: E402
    write_results,
    write_run_metadata,
)

PLOT_METRICS = {
    "normalized_rmse": "Normalized RMSE",
    "normalized_mae": "Normalized MAE",
}


def parse_args():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ifs_cache", default=None,
            help="IFS store from preprocess_ifs.py (default: data/ifs/ifs_hres_<split>.zarr)")
    parser.add_argument("--model_caches", nargs="*", default=None,
            help="Model caches to compare with. Default: every complete cache "
            "for the split in --prediction_dir")
    parser.add_argument("--prediction_dir", default=None,
            help="Where to look for model caches (default: prediction_store_dir "
            "of the config)")
    parser.add_argument("--config", default="evaluation_configs/evaluation.yaml",
            help="Evaluation YAML (leads, metrics, regions, extremes)")
    parser.add_argument("--output_dir", default=None,
            help="Default: <config output_dir>/ifs_comparison")
    parser.add_argument("--plot_variables", nargs="+", default=None,
            help="Variables to plot (default: all compared variables)")
    parser.add_argument("--no_plots", action="store_true", help="Only write tables")
    return parser.parse_args()


def find_model_caches(prediction_dir, split):
    """Complete caches for the split directly in prediction_dir, one per model."""
    by_model = {}
    for path in sorted(glob.glob(os.path.join(prediction_dir, "*.zarr"))):
        group = open_forecast_cache(path, require_complete=False)
        if group.attrs.get("split") != split or not group.attrs.get("complete"):
            continue
        by_model.setdefault(group.attrs["model_id"], []).append(path)
    duplicates = {m: p for m, p in by_model.items() if len(p) > 1}
    if duplicates:
        raise SystemExit("Several caches for the same model, pass --model_caches: "
                + "; ".join(f"{m}: {', '.join(p)}" for m, p in duplicates.items()))
    return [paths[0] for paths in by_model.values()]


def cache_summary(path):
    group = open_forecast_cache(path)
    return {
        "path": str(path),
        "model_id": str(group.attrs["model_id"]),
        "split": group.attrs.get("split"),
        "dataset": group.attrs.get("dataset"),
        "init_times": set(group["initialization_time_unix_s"][:].tolist()),
        "variables": list(group.attrs["variable_names"]),
        "leads": set(int(h) for h in group["lead_hours"][:]),
    }


def plot_by_lead(frame, metric, label, variables, output_dir):
    import matplotlib.pyplot as plt

    rows = frame[(frame["metric"] == metric) & frame["lead_hours"].notna()]
    for column in ("region", "variable_group", "event_type"):
        if column in rows:
            rows = rows[rows[column].isna()]
    variables = [v for v in variables if v in set(rows["variable"])]
    if not variables:
        return None

    n_cols = min(4, len(variables))
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.8 * n_rows),
            sharex=True, squeeze=False)
    models = list(dict.fromkeys(rows["model_id"]))
    for ax, variable in zip(axes.flat, variables):
        for model in models:
            line = rows[(rows["variable"] == variable) & (rows["model_id"] == model)]
            line = line.sort_values("lead_hours")
            style = {"color": "black", "linestyle": "--", "linewidth": 2} \
                if model.startswith("IFS") else {}
            ax.plot(line["lead_hours"], line["value"], marker="o", markersize=3,
                    label=model, **style)
        ax.set_title(variable)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Lead time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel(label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models),
            bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(output_dir) / f"{metric}_by_lead.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    config = load_evaluation_config(args.config)
    ifs_cache = args.ifs_cache or os.path.join("data", "ifs",
            f"ifs_hres_{config.split}.zarr")
    model_caches = args.model_caches
    if model_caches is None:
        prediction_dir = args.prediction_dir or getattr(
                config, "prediction_store_dir", None) or "evaluation_results/predictions"
        model_caches = find_model_caches(prediction_dir, config.split)
    if not model_caches:
        print("No model caches found, scoring IFS only")

    caches = [cache_summary(ifs_cache)] + [cache_summary(p) for p in model_caches]
    for cache in caches:
        if cache["dataset"] != config.dataset or cache["split"] != config.split:
            raise SystemExit(f"{cache['path']} is {cache['dataset']}/{cache['split']}, "
                    f"config is {config.dataset}/{config.split}")

    init_times = sorted(set.intersection(*(c["init_times"] for c in caches)))
    if not init_times:
        raise SystemExit("The caches have no initialization times in common")
    wanted = config.variables or constants.param_names
    variables = [v for v in constants.param_names if v in wanted
            and all(v in c["variables"] for c in caches)]
    leads = [h for h in config.lead_hours if all(h in c["leads"] for c in caches)]
    dropped_leads = sorted(set(config.lead_hours) - set(leads))

    hours = sorted({pd.Timestamp(t, unit="s").hour for t in init_times})
    print(f"Comparing {len(caches)} systems on {len(init_times)} common initializations "
            f"(hours {hours}, {pd.Timestamp(init_times[0], unit='s')} - "
            f"{pd.Timestamp(init_times[-1], unit='s')})")
    print(f"Variables: {', '.join(variables)}")
    print(f"Leads: {leads}" + (f" (not in all caches: {dropped_leads})"
            if dropped_leads else ""))

    output_dir = Path(args.output_dir or Path(config.output_dir) / "ifs_comparison")
    frames, run_infos = [], {}
    for cache in caches:
        print(f"Scoring {cache['model_id']} ({cache['path']})...", flush=True)
        records, info = evaluate_forecast_cache(
            cache["path"],
            lead_hours=leads,
            variables=variables,
            metrics=config.metrics,
            variable_groups=config.variable_groups,
            lead_groups=config.lead_groups,
            exclude_land_sst=config.exclude_land_sst,
            spatial_regions=config.spatial_regions,
            extreme_events=config.extreme_events,
            initialization_times_unix_s=init_times,
        )
        stem = f"{_slug(cache['model_id'])}_{_slug(config.split)}_matched"
        write_results(records, output_dir / f"{stem}_metrics.{config.results_format}",
                file_format=config.results_format)
        frames.append(pd.DataFrame.from_records(records))
        info.pop("cache_metadata", None)
        run_infos[cache["model_id"]] = info

    frame = pd.concat(frames, ignore_index=True)
    results_path = write_results(frame.to_dict("records"),
            output_dir / f"comparison_metrics.{config.results_format}",
            file_format=config.results_format)
    write_run_metadata({
        "config_path": args.config,
        "caches": {c["model_id"]: c["path"] for c in caches},
        "n_initializations": len(init_times),
        "initialization_hours_utc": hours,
        "first_initialization_unix_s": init_times[0],
        "last_initialization_unix_s": init_times[-1],
        "variables": variables,
        "lead_hours": leads,
        "runs": run_infos,
    }, output_dir / "comparison_metadata.json")
    print(f"Wrote {results_path}")

    # Short summary table: normalized RMSE averaged over all compared variables
    summary = frame[(frame["metric"] == "normalized_rmse") & frame["lead_hours"].notna()
            & frame["variable"].isin(variables)]
    for column in ("region", "variable_group", "event_type"):
        if column in summary:
            summary = summary[summary[column].isna()]
    if not summary.empty:
        table = summary.pivot_table(index="lead_hours", columns="model_id",
                values="value", aggfunc="mean")
        print("\nNormalized RMSE, mean over variables:")
        print(table.round(3).to_string())

    if not args.no_plots:
        plot_variables = args.plot_variables or variables
        for metric, label in PLOT_METRICS.items():
            path = plot_by_lead(frame, metric, label, plot_variables, output_dir)
            if path:
                print(f"Wrote {path}")


if __name__ == "__main__":
    main()

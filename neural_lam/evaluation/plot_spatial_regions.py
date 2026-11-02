"""Plot the spatial masks configured for an offline evaluation."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from neural_lam.evaluation.config import load_evaluation_config
from neural_lam.evaluation.forecast_cache import open_forecast_cache
from neural_lam.evaluation.spatial_regions import (
    build_configured_region_masks,
    build_spatial_region_masks,
)


def plot_configured_regions(cache_path, config_path, output_path):
    """Write a map preview and return the output path."""
    config = load_evaluation_config(config_path)
    if not config.spatial_regions["enabled"]:
        raise ValueError("Enable spatial_regions in the evaluation config")

    group = open_forecast_cache(cache_path)
    if group.attrs.get("dataset") != config.dataset:
        raise ValueError("Evaluation config dataset does not match the cache")
    latitude = np.asarray(group["latitude"][:], dtype=np.float64)
    longitude = np.asarray(group["longitude"][:], dtype=np.float64)
    land_sea_mask = np.asarray(group["land_sea_mask"][:], dtype=np.float32)
    boundary_mask = np.asarray(group["boundary_mask"][:], dtype=bool)
    region_definitions = config.spatial_regions["regions"]
    if region_definitions:
        regions = build_configured_region_masks(
            latitude, longitude, land_sea_mask, region_definitions
        )
    else:
        regions = build_spatial_region_masks(
            latitude,
            longitude,
            land_sea_mask,
            coastal_distance_km=config.spatial_regions["coastal_distance_km"],
        )
    regions = {
        name: mask & ~boundary_mask
        for name, mask in regions.items()
    }

    columns = min(3, len(regions))
    rows = (len(regions) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
        constrained_layout=True,
    )
    land_sea_cmap = ListedColormap(["#5b9bd5", "#9e9e9e"])
    region_cmap = ListedColormap(["#d62728"])
    for axis, (region_name, region_mask) in zip(axes.flat, regions.items()):
        axis.pcolormesh(
            longitude,
            latitude,
            (land_sea_mask >= 0.5).astype(np.int8),
            cmap=land_sea_cmap,
            vmin=0,
            vmax=1,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        selected = np.ma.masked_where(~region_mask, np.ones(region_mask.shape))
        axis.pcolormesh(
            longitude,
            latitude,
            selected,
            cmap=region_cmap,
            vmin=0,
            vmax=1,
            alpha=0.75,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        axis.set_title(f"{region_name} ({int(region_mask.sum()):,} cells)")
        axis.set_extent(
            [longitude.min(), longitude.max(), latitude.min(), latitude.max()],
            crs=ccrs.PlateCarree(),
        )
        axis.gridlines(draw_labels=False, linewidth=0.4, alpha=0.4)

    for axis in axes.flat[len(regions):]:
        axis.set_visible(False)

    figure.suptitle(
        "Region masks: blue = sea, gray = land, red = selected cells",
        fontsize=13,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview configured spatial masks without running inference."
    )
    parser.add_argument("--cache", required=True, help="Completed forecast cache")
    parser.add_argument("--config", required=True, help="Evaluation YAML")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args(argv)
    output_path = plot_configured_regions(
        args.cache,
        args.config,
        args.output,
    )
    print(f"Wrote region mask preview to {output_path}")


if __name__ == "__main__":
    main()

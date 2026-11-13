<p align="middle">
    <img src="figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
The code uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/pytorch-lightning).
Graph Neural Networks are implemented using [PyG](https://pyg.org/) and logging is set up through [Weights & Biases](https://wandb.ai/).

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

For more information see our paper: [*Graph-based Neural Weather Prediction for Limited Area Modeling*](https://arxiv.org/abs/2309.17370).
If you use Neural-LAM in your work, please cite:
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```

We plan to continue updating this repository as we improve existing models and develop new ones.
Collaborations around this implementation are very welcome.
If you are working with Neural-LAM feel free to get in touch and/or submit pull requests to the repository.

# ERA5 data (this branch)
On this branch the code reads ERA5 data (6-hourly, 0.25°, Nordic area) from a zarr store instead of the MEPS `.npy` samples described below.

First install Python 3.9 and run
```
pip install -r "requirements.txt"
```

## Steps to download and prepare ERA5 data

**1. Download from CDS**

Requires a [CDS API key](https://cds.climate.copernicus.eu/how-to-api) configured in `~/.cdsapirc`.
Downloads single-level and pressure-level GRIB files as zip archives into `data/ERA5_nordic/raw/`.
Years from 2022 up to the current year are downloaded automatically; ERA5 has a ~5-day lag so the most recent days may be absent.

```
python downloadERA5nordic.py
```

**2. Extract zip archives**

Extracts the downloaded zips into the directory structure expected by `prepare_era5.py`.

```
python unzipERA5.py
```

Expected structure after extraction:
```
data/
├── ERA5_nordic/
│   └── raw/
│       ├── ERA5.zip
│       ├── ERA5_pressure_2022.zip
│       ├── ERA5_pressure_2023.zip
│       └── ...
└── era5grib/
    ├── single_level/
    │   └── data.grib
    └── pressure/
        ├── 2022/
        │   └── data.grib
        ├── 2023/
        │   └── data.grib
        └── ...
```

**3. Convert to zarr and create static files**

```
python prepare_era5.py
python create_parameter_weights.py
python create_grid_features.py
python create_mesh.py --graph era5_multiscale
python create_mesh.py --graph era5_hierarchical --hierarchical 1  # only needed for Hi-LAM models
```

This creates `data/era5/era5.zarr` (all state variables, time x lat x lon x variable) and the files in `data/era5/static/`.
Variables used, their loss weights and the train/val/test date ranges are set in `neural_lam/constants.py`; re-run `create_parameter_weights.py` after changing them.
Surface geopotential is not part of the CDS download; to include it as a grid feature, pass a GRIB file with ERA5 surface geopotential to `prepare_era5.py --orography <file>`.
Top-of-atmosphere solar radiation is computed from time and position, not read from data.
`--step_length` is given in hours and must be a multiple of 6.

**Starting from an existing zarr store.**
If you already have `era5.zarr` (e.g. copied from another machine), steps 1 and 2 can be skipped.
Place it at `data/era5/era5.zarr` and run step 3 with `--static_only`, which reads the grid from the zarr store instead of the GRIB files:
```
python prepare_era5.py --static_only
```
followed by the `create_*` scripts as above.

## Training on ERA5
Training uses `train_model.py`; `--dataset era5` and `--graph era5_multiscale` are the defaults.
Results are logged to Weights & Biases, so run `wandb login` first, or set `WANDB_MODE=offline` to only log locally (see [Weights & Biases Integration](#weights--biases-integration)).

Check that everything runs on a small subset first (50 samples per split):
```
python train_model.py --subset_ds 1 --epochs 1
```

Train the models:
```
python train_model.py --model graph_lam --graph era5_multiscale --epochs 100 --patience 10
python train_model.py --model hi_lam --graph era5_hierarchical --epochs 100 --patience 10
python train_model.py --model hi_lam_parallel --graph era5_hierarchical --epochs 100 --patience 10
```

**Suggested number of epochs.**
One epoch is roughly 990 optimizer steps (about 3950 training samples at batch size 4).
Training uses AdamW without a learning rate schedule, so rely on early stopping rather than a fixed epoch count, and use the same settings for all three models so they are compared fairly:

1. Single-step training: `--epochs 100 --patience 10`. Training usually stops before the cap once `val_mean_loss` levels off.
2. Fine-tuning with a longer unroll, starting from the best single-step checkpoint:
```
python train_model.py --model graph_lam --graph era5_multiscale --ar_steps 4 --epochs 30 --patience 5 --load saved_models/<run_name>/min_val_loss.ckpt
```
Each step costs about 4 times as much with `--ar_steps 4`, so fewer epochs are needed.
Use the matching `--model` and `--graph` for Hi-LAM models.

If `val_mean_loss` is still decreasing when the epoch cap is reached, continue with `--resume` and a larger `--epochs`.
`--patience` counts validation runs, so with `--val_interval 2` a patience of 10 means 20 epochs.

Useful options (run `python train_model.py --help` for all):

* `--epochs`: Maximum number of epochs (default 200).
* `--patience`: Stop early when `val_mean_loss` has not improved for this many validation runs. Off by default, in which case training always runs for `--epochs`.
* `--ar_steps`: Number of 6 h steps unrolled in the training loss (default 1). Training first with 1 step and then continuing from that checkpoint with `--load <ckpt> --ar_steps 4` (or similar) usually gives more stable long forecasts.
* `--batch_size`, `--hidden_dim`, `--processor_layers`, `--lr`: Usual model and optimization settings.

Each run gets a directory `saved_models/<run_name>/` containing
* `min_val_loss.ckpt`: Model with the lowest `val_mean_loss`. Use this one for testing.
* `last.ckpt`: Model after the latest epoch. Use this one to resume.

**Resuming an interrupted run.**
`--resume` restores weights, optimizer, epoch counter and early stopping state, and keeps writing checkpoints to the same directory:
```
python train_model.py --model graph_lam --graph era5_multiscale --patience 10 --resume saved_models/<run_name>/last.ckpt
```
Pass the same model options as in the original run. The resumed run is logged as a new W&B run named `resume-<run_name>`.
`--load` is different: it only loads the weights (and with `--restore_opt 1` the optimizer) and starts a new run from epoch 0, which is what you want for fine-tuning with other settings such as a larger `--ar_steps`.

## Hyperparameter tuning
`hparam_search.py` tunes the model with [Optuna](https://optuna.org/). It needs one extra package:
```
pip install optuna
```

Each search (study) tunes one model.
Each trial runs `train_model.py` unchanged with a new set of arguments and is scored by its best `val_mean_loss`.
Optuna picks the arguments of the next trial based on the scores so far, and stops trials early (prunes them) when their `val_mean_loss` after an epoch is worse than the median of earlier trials at that epoch.
The searched arguments and their ranges are set in `suggest_config()` in `hparam_search.py`:

* `--hidden_dim` (32, 64, 128, 256), `--processor_layers` (2 to 8), `--hidden_layers` (1, 2) and `--lr` (1e-4 to 3e-3, log scale).
* `--mesh_aggr` (sum, mean), only for `graph_lam`.

Check that the search runs on the small subset first:
```
python hparam_search.py --study test --n_trials 2 --epochs 2 -- --subset_ds 1
```

Run a search for each model, one after another, with the same settings so the models are compared fairly:
```
python hparam_search.py --model graph_lam --n_trials 30 --epochs 30 --patience 0 -- --batch_size 4
python hparam_search.py --model hi_lam --n_trials 30 --epochs 30 --patience 0 -- --batch_size 4
python hparam_search.py --model hi_lam_parallel --n_trials 30 --epochs 30 --patience 0 -- --batch_size 4
```

* `--model`: Model to tune, `graph_lam` (default), `hi_lam` or `hi_lam_parallel`.
* `--graph`: Graph to use. Defaults to `era5_multiscale` for `graph_lam` and `era5_hierarchical` for the Hi-LAM models.
* `--study`: Name of the search, by default the model name. Trial logs are saved in `hparam_search/<study>/` and the results in `hparam_search/optuna.db`. A study belongs to one model and graph, so use a new name to tune e.g. `graph_lam` with the `1level` graph.
* `--n_trials`: Number of trials to run in this call. Running the same command again continues the study with more trials, also after an interruption (Ctrl-C).
* `--epochs`, `--patience`: Maximum epochs and early stopping patience of each trial. Short trials are enough to rank settings; train the best ones longer afterwards.
* `--timeout_h`: Stop a trial after this many hours.
* `--no_prune`: Run every trial to the end.
* Arguments after `--` are passed to every trial. Searched arguments given here are not tuned, e.g. `-- --lr 0.001` keeps the learning rate fixed.

Show the best trials, their run directories and how much each argument mattered:
```
python hparam_search.py --study graph_lam --summary
```

**Notes.**
* Keep `--loss` and the data settings the same within a study, since `val_mean_loss` is only comparable between runs with the same loss (see [Interpreting results](#interpreting-results)).
* Every trial is a normal training run, so it writes a `saved_models/<run_name>/` directory and a W&B run, grouped and tagged by the study name. Run only one search per study at a time on this single-GPU setup.
* The test split is not used during the search. After choosing settings, train them with `train_model.py` for the full number of epochs, fine-tune with a longer `--ar_steps` as above, and only then evaluate on the test split.
* `val_mean_loss` differs somewhat between runs with different `--seed`, so rerun the top 2 to 3 settings with a few seeds before picking one.

## Testing on ERA5
Evaluate the best checkpoint on the test split (2025-09-18 to 2026-09-18), with the same model options as in training:
```
python train_model.py --model graph_lam --graph era5_multiscale --eval test \
    --load saved_models/<run_name>/min_val_loss.ckpt --n_example_pred 3
```
Use `--eval val` instead to evaluate on the validation split.
Each forecast is started at 00 or 12 UTC from two initial states and unrolled for `--eval_steps` steps (default 19, i.e. 114 h).
`--n_example_pred` sets how many example forecasts are plotted.

## Interpreting results
**Where results are stored.**
Scalars and plots are logged to W&B.
Files are also saved in the local run directory `wandb/<run>/files/` (`wandb/offline-run-*/files/` in offline mode).

**Loss values (`train_loss`, `val_mean_loss`, `test_loss_unrollN`, `test_mean_loss`).**
These are weighted sums over all variables of the error in normalized units: each variable's error is divided by the variance of its typical 6 h change, and multiplied by its loss weight in `neural_lam/constants.py`.
This makes the loss unitless and dominated by `t2m` (weight 1.0) and the lower pressure levels.
Use it to compare runs and models, not as a physical error.
Losses are only comparable between runs with the same `--loss`, variables, weights and normalization statistics.

* `test_loss_unrollN` is the loss at lead time N steps (N × 6 h), averaged over all test forecasts. It should grow with lead time; errors accumulate as the model feeds its own predictions back in.
* `test_mean_loss` (and `val_mean_loss`) is the average over all lead times.
* During training, look at `val_mean_loss` rather than `train_loss`: training uses only `--ar_steps` steps, while validation unrolls full forecasts. A falling `train_loss` with a rising `val_mean_loss` means overfitting, which is what `--patience` guards against.

**Errors per variable in physical units.**
* `test_mae.csv` and `test_rmse.csv` contain the mean absolute error and root mean squared error in each variable's own unit (K, m/s, Pa, ...). Rows are lead times (6 h, 12 h, ..., 114 h), columns are the variables in the order of `param_names` in `neural_lam/constants.py`.
* `test_mae.pdf` and `test_rmse.pdf` show the same numbers as a heatmap with variables as rows and lead times as columns. The colour is normalized per row, so it shows how the error of each variable grows with lead time, not which variable is worst. Compare variables using the numbers.
* RMSE is always at least as large as MAE. RMSE much larger than MAE means a few large errors, for example misplaced fronts or precipitation.

**Where the errors are.**
`spatial_loss_maps/loss_t<N>.pdf` (and `test_loss` images in W&B) show the loss on the map at lead times 1, 2, 3, 5, 10, 15 and 19 steps.
The grid cells along the edge of the domain are not predicted: they are set to the true ERA5 state at every step, so their error is zero, and they are drawn faded.
Since information enters through this boundary, look in particular at how the error develops just inside it and over complex terrain.
The underlying values are saved in `mean_spatial_loss.pt`.

**Example forecasts.**
With `--n_example_pred`, W&B shows the prediction next to ERA5 for every variable and lead time (`<var>_example_<i>`), and the raw values are saved as `example_pred_<i>.pt` and `example_target_<i>.pt` (shape lead time × grid cell × variable, physical units, grid cells in row-major order over the 106 × 115 lat-lon grid).
Use these to see whether forecasts become too smooth or drift at long lead times, which the average scores do not show.

**Is the model any good?**
There is no built-in baseline.
A simple reference is persistence, i.e. using the last initial state as the forecast for all lead times; a useful model should beat it clearly at every lead time.
Also check that `val_mean_loss` and `test_mean_loss` are close: a much higher test loss points to overfitting to the validation period through checkpoint selection, or to differences between the two years.

## Config-driven evaluation (evaluation framework)

The evaluation framework in `neural_lam/evaluation/` provides reusable metric,
configuration, and result-writing helpers. The example configuration is
`evaluation_configs/evaluation.yaml`. It specifies the dataset and split,
forecast lead times, requested metrics, batch size, output directory, and
output format.

**Integration status:** `train_model.py` reads the YAML configuration and
passes the evaluation settings to the model. A small end-to-end evaluation
with forecast caching has completed successfully. For example:

```text
python train_model.py --model <model-architecture> --graph <graph-name> --load <checkpoint.ckpt> --eval_config evaluation_configs/evaluation.yaml --evaluation_model_id <model-label>
```

Use the matching architecture and graph settings for each trained checkpoint.
Run the same evaluation configuration for all four project models—1L-LAM,
Graphcast-LAM, HI-LAM, and HI-LAM-Parallel—and retain a distinct model label
and checkpoint identifier for each run. The YAML configuration defines the
shared evaluation settings; it does not replace the architecture settings
required to load a checkpoint.

`lead_hours` is expressed in hours after initialization. Leads must be
multiples of `step_length_hours`, and the greatest requested lead determines
the minimum rollout length. For example, a 114-hour maximum lead at a
6-hour step requires 19 prediction steps. A configuration may request selected
leads while the model rolls out all intermediate steps; metrics are then
reported for the selected leads.

### Structured result files

The output is a tidy metrics table under the configured
`output_dir`, with CSV as the default and Parquet as an option. Each row
identifies a model/checkpoint, lead time, variable, unit, metric, value,
valid grid-cell forecast sample count, and (when area weights are used) the
total metric weight. The count is the number of unmasked grid-cell values
across forecast initializations, not just the number of initialization times.
There is one row per metric, lead time, and variable. This makes results
straightforward to filter or plot without rerunning inference.

The run metadata JSON is saved beside the table and records the configuration,
split, model and checkpoint identifiers, and other settings needed to
reproduce the evaluation. Forecast-time metadata is also written as a separate
CSV. Full predictions are not part of these tabular result files; when caching
is enabled, they are stored separately in the Zarr store described below.

The helper can write Parquet through pandas, but Parquet output also requires
an installed engine such as `pyarrow` or `fastparquet`. CSV output does not
require an extra package.

### Reusable forecast cache (Zarr)

When `save_forecast_cache` is enabled in the evaluation YAML, each evaluation
run writes a separate Zarr store under `prediction_store_dir`. The store
contains physical-unit `prediction` and `target` arrays with dimensions
`(initialization, lead, y, x, variable)`, along with the latest observed
state, initialization and valid times, lead hours, coordinates, land–sea and
boundary masks, units, subset selection, and run metadata. Forecast batches are written as
inference proceeds; an interrupted store remains marked incomplete and is
rejected by the reader unless explicitly opened with
`require_complete=False`.
The current cache stores every rollout lead, even when `lead_hours` selects
only a subset for the configured metric output.

A `.zarr` path is a directory-backed dataset, not a collection of individual
files named after predicted values. Its array directories contain metadata
files and compressed binary chunks. For example, `prediction/0.1.0.0.0` is
the chunk at chunk-index coordinates `(initialization=0, lead=1, y=0, x=0,
variable=0)`; it is not a forecast value. In the current chunk layout, each
prediction chunk contains one initialization, one lead, the full grid, and
all variables. This supports efficient access to selected forecast
cases/leads without parsing or loading the entire dataset.

### Recompute metrics offline

The cached forecasts can be scored again without loading a checkpoint or
running inference. The offline evaluator reads one initialization at a time,
uses the dataset's saved training standard deviations for normalized metrics,
uses the cached latest observed state for persistence, and excludes boundary
cells using the mask in the cache:

```powershell
.\.venv\Scripts\python.exe -m neural_lam.evaluation.offline_metrics `
  --cache evaluation_results\predictions\<cache-name>.zarr `
  --config evaluation_configs\evaluation.yaml
```

It writes a separate `*_offline_metrics.csv` (or Parquet) and metadata JSON
under the configured `output_dir`, leaving inference-time results untouched.
The config's dataset and split must match the cache; requested leads and
variables must be present in it. The cache's physical-unit prediction and
target arrays are scored with MAE/RMSE, normalized MAE/RMSE, persistence
errors, and persistence skill according to the requested metrics.

### Compare forecasts at a specific location

Use the point-comparison tool to extract forecasts from the completed caches
at the grid cell nearest a geographic coordinate. By default, it compares the
four production model IDs on the test split at Grimstad, Norway:

```powershell
.\.venv\Scripts\python.exe -m neural_lam.evaluation.point_comparison `
  --cache-dir evaluation_results\predictions `
  --split test `
  --latitude 58.3405 `
  --longitude 8.5934 `
  --output-dir evaluation_results\point_comparisons\grimstad\test
```

The tool requires exactly one completed cache per requested model, with the
same dataset, grid and leads. Only the variables and initialization times that
every cache has are compared. If `data/ifs/ifs_hres_<split>.zarr` exists, IFS
forecasts are included as model `IFS-HRES` (see
[IFS HRES baseline](#ifs-hres-baseline-ecmwf); pass `--no-ifs` to leave them out).
Create those caches by evaluating each production checkpoint with
`evaluation_configs/evaluation.yaml` (which requests `split: test`), without
`--subset_ds 1` for the full available test split. The test-cache provenance is
checked; validation caches are not used in this comparison.
It selects the nearest grid-cell center using great-circle distance; it does
not interpolate between cells. For the current caches, Grimstad maps to
`58.25°N, 8.50°E`, about 11.45 km from the requested coordinate. The selected
cell is not boundary-forced; its land-sea fraction is about 0.435, so it is
classified as sea by the project's 0.5 threshold.

The output directory contains a per-initialization CSV with forecasts, truth,
errors, persistence forecasts, and timestamps; a summary CSV with MAE, RMSE,
normalized errors, bias, and skill against persistence by model, variable, and
lead; metadata describing the selected cell; pairwise relative MAE-reduction
tables; and comparison plots. By default, point plots are generated for
`t2m`, `d2m`, `tp`, `msl`, and `sst` when present. Each scalar plot shows
normalized MAE and normalized RMSE side by side. Wind components are retained
in the CSV results and used to calculate wind speed and direction, but are not
plotted separately. The derived wind plots show speed and direction at each
available component pair (10 m and the configured pressure levels).

Add `--include-persistence-skill` to include MAE and RMSE skill-versus-
persistence plots as well. Without that option, the figures contain only the
two normalized-error metrics; with it, scalar plots use four panels and wind
plots use four rows for each of speed and direction. The persistence skill
values are still calculated and saved in the result tables either way.
Use a new `--output-dir` when changing plot settings so figures from an earlier
run (such as separate `u`/`v` plots) are not mistaken for newly generated
figures; the script does not delete old output files.

The `grimstad_point_macro_metrics.csv` and
`grimstad_mean_normalized_mae_comparison` plot report the equal-weighted mean
normalized MAE across the included variables, separately at each lead. The
`grimstad_point_pairwise_overall.csv` reports each model pair's relative
reduction in that equal-variable, equal-lead mean error. Positive reduction
means the compared model has lower error than the baseline model.

Normalized MAE is MAE divided by that variable's training-split standard
deviation, so it is unitless and permits scale-aware comparisons across
variables; it is **not** a percentage of accuracy. For a direct pairwise
statement on one variable and lead, relative MAE reduction is
`100 * (1 - compared_model_normalized_MAE / baseline_model_normalized_MAE)`.
The macro result gives each cached variable and lead equal weight; it is a
descriptive aggregate, not a universal measure of weather quality. Also inspect
per-variable scores and persistence skill. In particular, precipitation's
meaning depends on the one-hour `tp` sample semantics and should not be
interpreted as a six-hour accumulation score.

For example, restrict extraction to variables needed for temperature,
precipitation, and wind-vector plots, and choose the plotted scalar variables:

```powershell
.\.venv\Scripts\python.exe -m neural_lam.evaluation.point_comparison `
  --cache-dir evaluation_results\predictions `
  --split test `
  --latitude 58.3405 `
  --longitude 8.5934 `
  --variables t2m d2m tp msl sst u10 v10 `
  --plot-variables t2m d2m tp msl sst `
  --calm-threshold 0.5 `
  --output-dir evaluation_results\point_comparisons\grimstad\test
```

Wind-vector plots are named
`grimstad_wind_vector_<level>_comparison.png/.pdf`; the corresponding
per-case and per-lead tables are `grimstad_wind_vector_cases.csv` and
`grimstad_wind_vector_metrics.csv`. Wind speed is calculated from each paired
`u`/`v` component set using `sqrt(u**2 + v**2)`. Direction uses the
meteorological wind-from convention. Speed normalization uses the
training-split standard deviation of derived speed at that level. Direction
error is the smallest circular angular difference; its MAE/RMSE are normalized
by the maximum circular error of 180 degrees. Direction is scored only when
observed wind speed meets the calm threshold; a calm forecast against a valid
windy observation receives 180 degrees error. If
`--include-persistence-skill` is set, the skill panels compare each model with
persistence computed from the latest observed wind components.

Use `--variables` to choose which values are extracted; both `u10` and `v10`
must be included for the combined wind plot. Use `--plot-variables` to select
the scalar variables for which figures are generated. Scores describe the
nearest ERA5 grid cell, not a point observation at the town itself.

Offline evaluation also adds derived 10 m wind-speed records when both
`u10` and `v10` are selected, plus configured variable-group and lead-group
summaries. Variable-group scores are unweighted means of the member variables'
normalized scores; lead-group scores are unweighted means over the configured
lead times. The per-variable and per-lead records remain available alongside
these summaries.

By default, SST metrics exclude land-dominant cells (`land_sea_mask >= 0.5`).
SST normalization uses the standard deviation of training-split SST over
interior ocean-dominant cells (`land_sea_mask < 0.5`). This differs from the
legacy `parameter_std.pt` SST statistic, which was calculated over all grid
cells, including filled land SST and boundary cells. The model's saved
normalization is not changed; this policy applies only to offline metric
interpretation. The run metadata records the masks and derived statistics.
Set `exclude_land_sst: false` only to reproduce the legacy all-cell SST
metric; that score includes cells where SST is undefined over land.

The current ERA5 audit found `parameter_std.pt` stores an SST standard
deviation of about 5.316 K. Recomputed over training-split interior
ocean-dominant cells, the standard deviation is about 4.643 K. The derived
training standard deviation for 10 m wind speed is about 3.710 m/s. These
statistics are written into the offline run metadata.

### Regional and extreme-weather metrics

The evaluation YAML can enable geographic metrics and threshold-based event
verification:

```yaml
spatial_regions:
  enabled: true
  scope: full_grid
  coastal_distance_km: 25

extreme_events:
  enabled: true
  lower_percentile: 1
  upper_percentile: 99
  precipitation: false
```

Regional masks cover the full ERA5 grid and split cells into coastal land,
inland land, nearshore sea, and offshore sea. Land is classified where the
fractional land-sea mask is at least 0.5. The configured coastal distance is
currently a great-circle distance between a grid-cell center and the nearest
opposite land/sea-class cell center, not a distance to a vector coastline.
This raster-based approximation and the selected 25 km band are recorded in
metadata; inspect region maps and consider sensitivity tests before treating
small coastal/inland differences as definitive. Regional MAE/RMSE and
normalized scores use cosine-of-latitude area weights and exclude
boundary-forced cells. Land-dominant SST cells are excluded as in the global
offline metrics.

You can replace those default bands with named regions for a particular
offline evaluation. A region can use a GeoJSON polygon, a longitude/latitude
bounding box, a land/sea class, and/or a distance band from the land-sea
boundary. For example:

```yaml
spatial_regions:
  enabled: true
  regions:
    norway_coastal_land:
      polygon_file: boundaries/norway.geojson
      land_sea: land
      max_coast_distance_km: 25
    norway_inland:
      polygon_file: boundaries/norway.geojson
      land_sea: land
      min_coast_distance_km: 25
    ocean:
      land_sea: sea
    norwegian_sea_box:
      polygon_file: boundaries/norwegian_sea.geojson
      land_sea: sea
```

GeoJSON paths are relative to the evaluation YAML file and should describe
valid Polygon/MultiPolygon geometries in longitude/latitude coordinates using
the same longitude convention as the cached grid. A bounding-box definition
can replace `polygon_file`, using `[min_longitude, min_latitude,
max_longitude, max_latitude]`. If `regions` is present, its names replace the
default global/coastal region set for that evaluation; if it is omitted, the
default set is used. You may define several masks in one run or use separate
evaluation YAML files to compute different focused subsets from the same
cache. Region definitions are recorded in the output metadata.

The land/sea mask identifies land versus water, not country borders or named
seas. For Norway-only scores, supply an appropriate Norway polygon; for a
specific ocean/sea, supply its polygon or a documented bounding box. Grid
cells are selected by whether their centers fall within the geometry. This
does not apportion cells that straddle a region boundary.

Preview the selected cells (including the boundary-cell exclusion) before
interpreting regional scores:

```powershell
.\.venv\Scripts\python.exe -m neural_lam.evaluation.plot_spatial_regions `
  --cache evaluation_results\predictions\<cache-name>.zarr `
  --config evaluation_configs\evaluation.yaml `
  --output evaluation_results\figures\region_masks.png
```

The preview uses blue for sea, gray for land, and red for cells selected by
each configured region. It does not retrieve a coastline or country boundary
from an external service; those are represented only if the corresponding
polygon file is supplied.

Extreme thresholds are empirical percentiles computed from the entire
training split over all interior grid cells, using physical-unit data. The
default 99th/1st percentiles detect high/low 2 m temperature, low mean sea
level pressure, and high derived 10 m wind speed. The same threshold is used
for every forecast lead and region. Outputs include POD, precision, false
alarm ratio, CSI, frequency bias, and MAE/RMSE over observed extremes; the
metadata stores threshold values, units, percentiles, and sample scope.
Contingency scores use cosine-of-latitude area weights. These are grid-cell
event samples, not counts of distinct meteorological storm systems.

Precipitation events are deliberately disabled. The `tp` accumulation
interval has not been verified; setting `extreme_events.precipitation: true`
is rejected rather than silently applying an ambiguous threshold. Fractions
Skill Score (FSS), map-ready per-grid-cell error outputs, and sensitivity
analysis are not yet implemented. To calculate these additions from the
existing forecasts, rerun only the offline command above; model inference is
not required.

Use the Zarr API to inspect or slice the cache:

```python
import zarr

group = zarr.open_group(
    "evaluation_results/predictions/<cache-name>.zarr", mode="r"
)
print(group["prediction"].shape)
print(group["lead_hours"][:])
one_forecast = group["prediction"][0, 0, :, :, :]
```

CSV and Parquet remain appropriate for the much smaller metric tables. Saving
every grid-cell prediction as CSV would flatten a large multidimensional
dataset, repeat coordinates and identifiers, use substantially more space,
and make selective reads slower. Export a selected variable, lead, or case to
CSV only when a tabular slice is specifically needed. A 50-initialization
smoke cache occupied about 1.93 GiB on disk; full-run storage should be
estimated before evaluating all models.

### Interpreting normalized metrics and persistence skill

`normalized_mae` and `normalized_rmse` divide each variable's physical-unit
MAE or RMSE by that variable's standard deviation computed from the training
data. They are unitless, so they help compare relative errors across
temperature, wind, pressure, and other variables with different units and
scales. Lower values indicate smaller errors relative to the variable's
training-data spread. They are not percentages, and should not be confused
with the training loss, which uses the standard deviation of forecast-step
differences and the configured loss weights. Use the same training
statistics, variables, and spatial mask for every model comparison.

The persistence baseline repeats the latest observed state at every forecast
lead. Persistence skill is computed as
`1 - model_error / persistence_error`, separately for MAE and RMSE:

* Positive skill means the model error is lower than persistence. For example,
  `0.2` means 20% lower error than the baseline.
* Zero means the model and persistence have equal error.
* Negative skill means persistence performs better.
* If persistence error is zero, the skill is undefined and should be reported
  as missing rather than interpreted as a finite score.

Compare model and persistence errors on identical forecast initializations,
leads, variables, and valid grid cells. In particular, exclude the
boundary-forced cells: the model receives reference ERA5 values there during
rollout, so including them would make both the model evaluation and baseline
comparison misleading.

## IFS HRES baseline (ECMWF)

ECMWF's operational forecast (IFS HRES) can be added as an extra model in the
evaluation. Its forecasts are taken from the
[ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
archive (0.25°, the same grid as ERA5, free and without an account),
converted to the forecast cache format and scored against ERA5 like the
trained models. The scripts are in `ifs/` and are run from the repository root.
They need `requests`, `cfgrib` and `eccodes` (all in `requirements.txt`) and
`data/era5` prepared as described above.

**1. Download**

```
python ifs/download_ifs.py
```

By default this downloads the 00, 06, 12 and 18 UTC runs (the same
initialization hours as the model evaluation, `eval_init_hours` in
`neural_lam/constants.py`) for every day of the test
split (`split_times["test"]` in `neural_lam/constants.py`), lead times 6 to
114 h every 6 h. Only the needed fields are fetched, cropped to the ERA5 grid
and saved as one file per run in `data/ifs/nordic/ifs_YYYYMMDDHH.npz`
(about 14 MB each). A run takes about a minute, so the full test year
(1460 runs) takes around a day and 20 GB. The script can be stopped (Ctrl-C) and
restarted at any time: finished runs are skipped, and an unfinished or failed
run continues from the forecast steps it already has (kept in
`data/ifs/grib/ifs_YYYYMMDDHH/` until the run is done). Each run fetches about 350 MB of global GRIB messages (about
500 GB for the full test year), which are deleted after cropping.

Useful options (run with `--help` for all):

* `--source`: Mirror of the archive, `google` (default) or `aws`. Both have the
  same files, but the AWS bucket often throttles requests (`503 SlowDown`).

* `--start`, `--end`: Run dates, `YYYY-MM-DD` (default: the test split).
* `--every_n_days`: Only download every n-th day, e.g. `2` halves the time.
* `--init_hours`: Run hours (default `0 6 12 18`), e.g. `0 12` halves the time.
* `--max_lead`: Longest lead time (default 114, the same as the model caches).

**2. Preprocess**

```
python ifs/preprocess_ifs.py
```

This writes `data/ifs/ifs_hres_test.zarr`, a forecast cache with model id
`IFS-HRES` in the same format as the model caches in
`evaluation_results/predictions`. The targets and persistence states are read
from `data/era5/era5.zarr`, and boundary cells are set to ERA5 like for the
models (they are excluded from all scores). Use `--overwrite` to rebuild it
after downloading more runs, and `--split val` for the validation year.

Differences from the ERA5 variables:

* `sst` is not in the open data, so IFS is not scored on it.
* `z_<level>` is computed from geopotential height as `gh * 9.80665`.
* ERA5 `tp` at the 6-hourly samples is a 1 h accumulation, while the open data
  has 3-hourly steps. IFS `tp` is the mean hourly rate over the last 3 h,
  `(tp(L) - tp(L-3)) / 3`, so precipitation scores are only roughly
  comparable. Pass `--exclude_tp` to leave it out.

**3. Evaluate**

The IFS cache can be used with the existing evaluation tools like any model
cache:

```
# All configured metrics for IFS alone
python -m neural_lam.evaluation.offline_metrics --cache data/ifs/ifs_hres_test.zarr --config evaluation_configs/evaluation.yaml

# Point comparison: IFS-HRES is added automatically when data/ifs/ifs_hres_<split>.zarr exists
python -m neural_lam.evaluation.point_comparison --split test --output-dir evaluation_results/point_comparisons/grimstad/test
```

`point_comparison` takes `--ifs-cache <path>` to use another IFS store and
`--no-ifs` to leave IFS out. When IFS is included, all models are scored on
the initializations and variables that every cache has, so `sst` is left out,
and any runs missing from the IFS download are left out for all models.

To compare IFS and all models over the whole grid on the same forecasts, use:

```
python ifs/evaluate_ifs.py
```

It finds the complete model caches for the config's split in
`evaluation_results/predictions` (or use `--model_caches <path> ...`), keeps
the initializations, variables and lead times that all of them share, and runs
the offline evaluation with `evaluation_configs/evaluation.yaml` for each one.
Results go to `evaluation_results/metrics/ifs_comparison/`: one metrics CSV
per model, `comparison_metrics.csv` with all of them, and plots of normalized
RMSE and MAE per variable against lead time, with IFS as a dashed black line.

When interpreting the results, note that IFS starts from ECMWF's own
operational analysis, not from ERA5, and is scored against ERA5. This
penalizes IFS somewhat at short lead times compared with models that start
from ERA5.


# Modularity
The Neural-LAM code is designed to modularize the different components involved in training and evaluating neural weather prediction models.
Models, graphs and data are stored separately and it should be possible to swap out individual components.
Still, some restrictions are inevitable:

* The graph used has to be compatible with what the model expects. E.g. a hierarchical model requires a hierarchical graph.
* The graph and data are specific to the limited area under consideration. This is of course true for the data, but also the graph should be created with the exact geometry of the area in mind.

<p align="middle">
  <img src="figures/neural_lam_setup.png" width="600"/>
</p>


## A note on the limited area setting
Currently we are using these models on a limited area covering the Nordic region, the so called MEPS area (see [paper](https://arxiv.org/abs/2309.17370)).
There are still some parts of the code that is quite specific for the MEPS area use case.
This is in particular true for the mesh graph creation (`create_mesh.py`) and some of the constants used (`neural_lam/constants.py`).
If there is interest to use Neural-LAM for other areas it is not a substantial undertaking to refactor the code to be fully area-agnostic.
We would be happy to support such enhancements.
See the issues https://github.com/joeloskarsson/neural-lam/issues/2, https://github.com/joeloskarsson/neural-lam/issues/3 and https://github.com/joeloskarsson/neural-lam/issues/4 for some initial ideas on how this could be done.

# Using Neural-LAM
Below follows instructions on how to use Neural-LAM to train and evaluate models.

## Installation
Follow the steps below to create the neccesary python environment.

1. Install GEOS for your system. For example with `sudo apt-get install libgeos-dev`. This is neccesary for the Cartopy requirement.
2. Use python 3.9.
3. Install version 2.0.1 of PyTorch. Follow instructions on the [PyTorch webpage](https://pytorch.org/get-started/previous-versions/) for how to set this up with GPU support on your system.
4. Install required packages specified in `requirements.txt`.
5. Install PyTorch Geometric version 2.2.0. This can be done by running
```
TORCH="2.0.1"
CUDA="cu117"

pip install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1\
    torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
You will have to adjust the `CUDA` variable to match the CUDA version on your system or to run on CPU. See the [installation webpage](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more information.

## Data
Datasets should be stored in a directory called `data`.
See the [repository format section](#format-of-data-directory) for details on the directory structure.
For the ERA5 data used on this branch, see the [ERA5 data section](#era5-data-this-branch).


The full MEPS dataset is available for download [here](https://nextcloud.liu.se/s/meps). See instructions in the file `README.txt` for how to download and extract the full dataset.

A tiny subset of the MEPS data (named `meps_example`) is also available in `example_data.zip`, which can be downloaded from [here](https://liuonline-my.sharepoint.com/:u:/g/personal/joeos82_liu_se/EaUUq6h9og1EsLwJmKAltWwB7zP2gmObe-K8pL6qGYYiGg?e=yQbFuV).
Download the files and unzip in the neural-lam directory.
All graphs used in the paper are also available for download at the same link (but can as easily be re-generated using `create_mesh.py`).
Note that this subset is far too little data to train any useful models, but all scripts can be ran with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be ran without any issues.

## Pre-processing
An overview of how the different scripts and files depend on each other is given in this figure:
<p align="middle">
  <img src="figures/component_dependencies.png"/>
</p>
In order to start training models at least three pre-processing scripts have to be ran:

* `create_mesh.py`
* `create_grid_features.py`
* `create_parameter_weights.py`

### Create graph
Run `create_mesh.py` with suitable options to generate the graph you want to use (see `python create_mesh.py --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python create_mesh.py --graph multiscale`
* **Hi-LAM**: `python create_mesh.py --graph hierarchical --hierarchical 1` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python create_mesh.py --graph 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

### Create remaining static features
To create the remaining static files run the scripts `create_grid_features.py` and `create_parameter_weights.py`.
The main option to set for these is just which dataset to use.

## Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally to a directory like `wandb/dryrun...`.
The W&B project name is set to `neural-lam`, but this can be changed in `neural_lam/constants.py`.
See the [W&B documentation](https://docs.wandb.ai/) for details.

If you would like to login and use W&B, run:
```
wandb login
```
If you would like to turn off W&B and just log things locally, run:
```
wandb off
```

## Train Models
Models can be trained using `train_model.py`.
Run `python train_model.py --help` for a full list of training options.
A few of the key ones are outlined below:

* `--dataset`: Which data to train on
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps`: Number of time steps to unroll for when making predictions and computing the loss

Checkpoints of trained models are stored in the `saved_models` directory.
The implemented models are:

### Graph-LAM
This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step pedictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](https://arxiv.org/abs/2309.17370), only with different graphs.

To train 1L-LAM use
```
python train_model.py --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python train_model.py --model graph_lam --graph multiscale ...
```

### Hi-LAM
A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python train_model.py --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in paralell.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.

To train Hi-LAM-Parallel use
```
python train_model.py --model hi_lam_parallel --graph hierarchical ...
```

Checkpoint files for our models trained on the MEPS data are available upon request.

## Evaluate Models
Evaluation is also done using `train_model.py`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation (not `ar_steps`, evaluation always unrolls full forecasts).
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable. This issue stems from PyTorch Lightning. See for example [this draft PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion and ongoing work to remedy this.

# Repository Structure
Except for training and pre-processing scripts all the source code can be found in the `neural_lam` directory.
Model classes, including abstract base classes, are located in `neural_lam/models`.

## Format of data directory
It is possible to store multiple datasets in the `data` directory.
Each dataset contains a set of files with static features and a set of samples.
The samples are split into different sub-directories for training, validation and testing.
The directory structure is shown with examples below.
Script names within parenthesis denote the script used to generate the file.
```
data
├── dataset1
│   ├── samples                             - Directory with data samples
│   │   ├── train                           - Training data
│   │   │   ├── nwp_2022040100_mbr000.npy  - A time series sample
│   │   │   ├── nwp_2022040100_mbr001.npy
│   │   │   ├── ...
│   │   │   ├── nwp_2022043012_mbr001.npy
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy   - Solar flux forcing
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
│   │   │   ├── ...
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022043012.npy
│   │   │   ├── wtr_2022040100.npy          - Open water features for one sample
│   │   │   ├── wtr_2022040112.npy
│   │   │   ├── ...
│   │   │   └── wtr_202204012.npy
│   │   ├── val                             - Validation data
│   │   └── test                            - Test data
│   └── static                              - Directory with graph information and static features
│       ├── nwp_xy.npy                      - Coordinates of grid nodes (part of dataset)
│       ├── surface_geopotential.npy        - Geopotential at surface of grid nodes (part of dataset)
│       ├── border_mask.npy                 - Mask with True for grid nodes that are part of border (part of dataset)
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)
│       ├── flux_stats.pt                   - Mean and std.-dev. of solar flux forcing (create_parameter_weights.py)
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)
├── dataset2
├── ...
└── datasetN
```

## Format of graph directory
The `graphs` directory contains generated graph structures that can be used by different graph-based models.
The structure is shown with examples below:
```
graphs
├── graph1                                  - Directory with a graph definition
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── g2m_edge_index.pt                   - Edges from grid to mesh (create_mesh.py)
│   ├── m2g_edge_index.pt                   - Edges from mesh to grid (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── g2m_features.pt                     - Static features of grid to mesh edges (create_mesh.py)
│   ├── m2g_features.pt                     - Static features of mesh to grid edges (create_mesh.py)
│   └── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
├── graph2
├── ...
└── graphN
```

### Mesh hierarchy format
To keep track of levels in the mesh graph, a list format is used for the files with mesh graph information.
In particular, the files
```
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
```
all contain lists of length `L`, for a hierarchical mesh graph with `L` layers.
For non-hierarchical graphs `L == 1` and these are all just singly-entry lists.
Each entry in the list contains the corresponding edge set or features of that level.
Note that the first level (index 0 in these lists) corresponds to the lowest level in the hierarchy.

In addition, hierarchical mesh graphs (`L > 1`) feature a few additional files with static data:
```
├── graph1
│   ├── ...
│   ├── mesh_down_edge_index.pt             - Downward edges in mesh graph (create_mesh.py)
│   ├── mesh_up_edge_index.pt               - Upward edges in mesh graph (create_mesh.py)
│   ├── mesh_down_features.pt               - Static features of downward mesh edges (create_mesh.py)
│   ├── mesh_up_features.pt                 - Static features of upward mesh edges (create_mesh.py)
│   ├── ...
```
These files have the same list format as the ones above, but each list has length `L-1` (as these edges describe connections between levels).
Entries 0 in these lists describe edges between the lowest levels 1 and 2.

# Contact
If you are interested in machine learning models for LAM, have questions about our implementation or ideas for extending it, feel free to get in touch.
You can open a github issue on this page, or (if more suitable) send an email to [joel.oskarsson@liu.se](mailto:joel.oskarsson@liu.se).

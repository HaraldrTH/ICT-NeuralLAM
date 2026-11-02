# Implementation checklist — Neural-LAM evaluation framework

Overall goal: Extend your existing evaluation pipeline to compare the four trained models across weather variables, forecast horizons, geographical regions, and extreme-weather events, with clear, informative visualizations.

The plan below is based on the repository files you've provided. I've organized the work into six development phases, in recommended implementation order.

## Status update

The evaluation foundation, Zarr forecast cache, cache validation against the
ERA5 source, and offline core metric recomputation are implemented. Offline
metrics reproduce the smoke-run metric table without loading a model. The
current smoke checkpoint is only a software test and is not one of the
production model comparisons.

The detailed, current status and verification evidence are tracked in
[TODO_EVALUATION_ROADMAP.md](./TODO_EVALUATION_ROADMAP.md). In brief, derived
wind and grouped metrics are now available offline. SST land masking and its
normalization policy have been audited and corrected for offline evaluation.
Full-grid regional metrics and percentile-based wind/temperature/pressure
event metrics are also available offline. Checkpoint loading and single-case
forward passes are verified, but graph identity and aligned all-model caches
remain open. Region-map/sensitivity validation, precipitation events, FSS,
uncertainty estimates, and report visualizations also remain open.

Four newer checkpoints have now been found and each has completed a finite
one-initialization validation forward pass: 1L-LAM, Graphcast-LAM, HI-LAM, and
HI-LAM-Parallel. Full forecast caches and comparable scores for those four
models have not yet been generated. See the roadmap for the checkpoint/graph
details and the remaining evaluation protocol checks.

## Phase 1 — Improve the existing evaluation pipeline

Priority: Essential | Primarily `train_model.py`, `ar_model.py`, `weather_dataset.py`

- [x] Add YAML evaluation configuration for dataset/split, leads, metrics,
  variables, result output, optional forecast-cache settings, and regional/
  extreme-event options.
- [x] Save forecast metadata and identify cached arrays with initialization
  times, valid times/leads, model/checkpoint, variable names, and units.
- [x] Write tidy CSV/Parquet metric results and optionally cache predictions,
  targets, masks, coordinates, latest observations, and provenance in Zarr.
- [x] Stream metric accumulation and write forecast batches incrementally
  rather than retaining the full prediction set in RAM.
- [x] Preserve existing evaluation metrics and Weights & Biases logging.
- [x] Use metadata-enabled validation and test datasets for evaluation.
- [x] Inventory and load the four newer candidate checkpoints. Their saved
  arguments identify ERA5, 6-hour cadence, expected architectures, and graphs;
  each completed one finite validation forward pass. The older flat checkpoint
  files are a separate legacy set and are not compatible as-is.
- [ ] Verify checkpoint graph identity. The missing `1level` and
  `era5_hierarchical` graph folders were regenerated from current ERA5 static
  coordinates, but no graph fingerprint was saved in the checkpoints.
- [ ] Generate full aligned forecast caches and compare metrics for all four
  models before making performance claims.

Outcome: A reusable evaluation pipeline that can calculate additional metrics without retraining models or unnecessarily rerunning inference.

## Phase 2 — Implement the core evaluation metrics

Priority: Essential | `neural_lam/evaluation/`

- [x] Calculate MAE and RMSE per variable and forecast lead from inference and
  cached predictions.
- [x] Calculate normalized MAE/RMSE using the dataset's saved training
  standard deviations. Review the statistics policy before interpreting
  cross-model scores.
- [x] Compare against persistence using cached latest observations, with
  matching valid cases and the boundary mask.
- [x] Add configurable variable-group summaries as equal-weight averages of
  normalized per-variable metrics, while retaining per-variable results.
- [x] Add derived 10 m wind speed from physical-unit `u10`/`v10`, normalized
  by its own training-split standard deviation.
- [x] Add configurable lead-time groups (short, medium, extended, long)
  without discarding per-lead results. Group values are equal-weight means of
  the configured lead-specific metric values.
- [x] Audit the training normalization policy. Offline SST metrics exclude
  land-dominant cells and use an interior ocean-dominant training standard
  deviation; the checkpoint's saved training normalization remains unchanged.
- [x] Preserve per-lead results so new groupings can be computed without
  repeating inference.

Outcome: Meaningful, unit-consistent comparisons between models, variables, and forecast horizons.

## Phase 3 — Implement geographical evaluation

Priority: High | `evaluation/spatial_regions.py`

- [x] Create reusable full-grid land/sea and coastal-band masks from ERA5
  coordinates and the land–sea mask.
- [x] Define coastal land, inland land, nearshore sea, and offshore sea, with
  a configurable 25 km raster distance to the nearest opposite-class grid
  cell center. This approximates, but is not a vector-coastline distance.
- [x] Allow per-evaluation named masks from GeoJSON polygons or geographic
  bounding boxes, combined with land/sea and coast-distance filters.
- [x] Add an offline preview command to inspect configured masks before
  interpreting regional scores.
- [ ] Inspect preview maps and test coast-distance sensitivity for the
  regions used in final comparisons.
- [ ] Provide suitable Norway/named-sea boundary polygons before reporting
  scores for those exact geographic areas.
- [ ] Add Norway-only regions, northern/southern Norway, or terrain regions
  when a suitable boundary/elevation source is available.
- [x] Exclude boundary-forced cells from evaluation, since they receive reference
  ERA5 values. The global core metrics already do this; regional and spatial
  metrics must apply the same mask.
- [x] Use cosine-of-latitude grid-area weights in regional averages.
- [x] Calculate regional physical and normalized MAE/RMSE from cached
  forecasts, excluding boundary cells and land SST.
- [ ] Save per-grid-cell error statistics for generating geographical maps.

Outcome: Determine where models perform well or poorly, especially along Norway's coastline and over surrounding waters.

## Phase 4 — Implement extreme-weather evaluation

Priority: High | New `evaluation/extreme_events.py`

- [x] Configure reproducible training-data percentiles (default 99th/1st)
  over all interior grid cells and training timestamps.
- [x] Detect high 10 m wind speed, high/low 2 m temperature, and low mean
  sea-level pressure.
- [ ] Defer precipitation events until the `tp` accumulation interval is
  verified; config rejects enabling them in the meantime.
- [x] Calculate area-weighted POD, precision, false-alarm ratio, CSI,
  frequency bias, and error during observed extremes by lead and region.
- Support lead-time-specific event evaluation, distinguishing short- and long-range extreme-weather forecasts.
- [ ] Implement Fractions Skill Score (FSS) to evaluate spatial displacement
  of severe precipitation and wind events.
- Save selected extreme-weather forecasts for detailed map-based case studies.

Outcome: Measure whether the models successfully predict important or dangerous weather, rather than only minimizing average error.

## Phase 5 — Build the visualization system

Priority: High | New `evaluation_scripts/` directory

| Visualization                          | What it should show                                                        |
| -------------------------------------- | -------------------------------------------------------------------------- |
| 1. Lead-time error curves              | MAE/RMSE for all four models at each forecast horizon                      |
| 2. Normalized performance heatmaps     | Cross-variable model comparisons without unit-scale bias                   |
| 3. Regional comparison charts          | Coastal vs. inland vs. offshore forecasting performance                    |
| 4. Geographical error maps             | Where individual models make the largest forecast errors                   |
| 5. Model-difference maps               | Locations where one model outperforms another                              |
| 6. Extreme-event detection plots       | Recall, CSI and false alarms versus lead time and threshold                |
| 7. Extreme vs. ordinary weather charts | How forecast errors change during observed extreme conditions              |
| 8. Selected forecast maps              | Ground truth alongside predictions from the four models                    |
| 9. Forecast skill vs. compute cost     | Whether accuracy improvements justify increased computational requirements |

Implementation requirements:

- Make all figures automatically generated from stored evaluation results.
- Use consistent model colors, units, labels, and scales across figures.
- Support filtering by model, variable, region, and lead-time group.
- Support PNG, SVG, and PDF export for the report.
- Clearly identify the number of samples or events included in each evaluation.
- Keep raw physical-unit and normalized comparisons clearly distinguished.

Outcome: Publication-ready figures that communicate the evaluation results far more clearly than the existing large tables.

## Phase 6 — Validation and reproducibility

Priority: Essential before finalizing results

- [x] Verify the cached ERA5 grid dimensions: 106 × 115 = 12,190 cells.
- [x] Reproduce the smoke-run metrics from the cache without model inference.
- [x] Verify the smoke cache's timestamps and lead-time alignment.
- [x] Verify cached variable names/units against project constants. The
  physical interpretation and accumulation interval of variables such as
  `tp` still needs a scientific audit before event thresholds are selected.
- [x] Exclude artificial SST values over land from offline SST evaluation.
- [x] Verify boundary masking in core and regional/event metrics.
- [ ] Use identical test cases and masks across all four production models;
  this requires the compatible production checkpoints and caches.
- Calculate uncertainty estimates for important model comparisons, where feasible.
- [x] Save evaluation configurations and checkpoint identifiers alongside
  results and forecast caches.
- [x] Document reference-boundary forcing and distinguish the current
  experiment from fully operational forecasting.

Outcome: Reproducible and scientifically defensible model comparisons.

## Suggested repository structure

The existing training and model code should remain largely unchanged. Most new functionality belongs in a separate evaluation package.

```
project/
│
├── train_model.py                 # Loads evaluation configuration
│
├── neural_lam/
│   ├── models/
│   │   └── ar_model.py           # Evaluation and cache-writing hooks
│   ├── weather_dataset.py        # Forecast timestamps in evaluation batches
│   ├── constants.py
│   └── evaluation/
│       ├── accumulators.py
│       ├── config.py
│       ├── forecast_cache.py
│       ├── metrics.py            # Helpers; derived/group metrics not integrated
│       ├── offline_metrics.py    # Recompute core metrics without inference
│       ├── persistence.py
│       └── results_io.py
│
├── evaluation_scripts/           # PLANNED
│   ├── compare_models.py
│   ├── plot_lead_times.py
│   ├── plot_heatmaps.py
│   ├── plot_regions.py
│   ├── plot_spatial_maps.py
│   ├── plot_extremes.py
│   └── generate_report.py
│
├── evaluation_configs/
│   ├── evaluation.yaml
│   ├── regions.yaml
│   └── extreme_events.yaml
│
└── evaluation_results/           # Generated outputs; not source modules
    ├── metrics/
    ├── spatial/
    ├── predictions/
    ├── case_studies/
    └── figures/
```

## Recommended development order

### Current next actions

1. Run a small validation-cache evaluation for all four candidate checkpoints
   on identical cases, then verify timestamp/cache alignment and offline
   metric outputs.
2. Verify/document the recreated graph assets and inspect region masks; test
   coastal-band sensitivity or provide versioned Norway/named-sea polygons.
3. Freeze the final evaluation protocol and estimate disk use before full
   inference on the chosen split.
4. Verify `tp` accumulation semantics, then consider precipitation events;
   also implement FSS and event-count reliability checks.
5. Generate comparison figures and uncertainty estimates after all models
   have aligned caches.

The one-epoch smoke checkpoint is suitable for testing the software only; do
not use its scores to rank the production models.

### Overall development milestones

Milestone 1 — Reliable evaluation foundation

Phases 1, 2 and the essential correctness checks from Phase 6.

Deliverable: Reproducible MAE/RMSE, normalized metrics, persistence
comparison, and lead-time curves.

Milestone 2 — Full-grid geographical analysis

Phase 3 and region-specific visualization from Phase 5. The current regions
cover the ERA5 grid and are not Norway-only.

Deliverable: Coastal, inland and offshore performance comparisons and spatial error maps.

Milestone 3 — Extreme-weather assessment

Phase 4 and extreme-event visualization from Phase 5.

Deliverable: Strong-wind, temperature, pressure, and (after accumulation
verification) precipitation event evaluations.

Milestone 4 — Final comparisons and report figures

Remaining visualization, uncertainty estimation, and documentation.

Deliverable: A coherent set of figures and tables comparing all four trained models.

The evaluation foundation, offline core metrics, derived wind speed, variable
groups, lead groups, SST normalization policy, full-grid and user-configurable
regional metrics, and initial extreme-event metrics are implemented. The four
newer candidate checkpoints have all passed a one-case validation forward
pass. Next, build small aligned caches for those candidates, then proceed to
full inference only after checkpoint graphs, evaluation cases, and storage
requirements are understood. This avoids repeating expensive inference and
lets each stage build on validated caches.

The result should be a modular system where you can investigate new combinations of variables, forecast horizons, and regions primarily by changing evaluation configurations — without modifying or retraining the four neural weather models.
# Evaluation roadmap: reusable forecast cache and model comparison

## Goal

Evaluate the four trained model variants—1L-LAM, Graphcast-LAM, HI-LAM, and
HI-LAM-Parallel—on the same forecast cases, then compare them by variable,
lead time, region, and extreme-weather conditions. Preserve enough forecast
data that new metrics and plots can usually be calculated without rerunning
the models.

This roadmap is a proposed, prioritized successor to the broader checklist in
[TODO.md](./TODO.md). It does not replace or edit that file.

## Current implementation snapshot

### Already implemented

- YAML evaluation configuration is loaded by `train_model.py`.
- Test/validation datasets can return initialization and valid times.
- Evaluation computes per-lead MAE/RMSE, normalized MAE/RMSE, and persistence
  skill using the interior (non-boundary) mask.
- Metric tables, run metadata, and forecast-time metadata can be written.
- Optional Zarr caching writes physical-unit predictions, targets, latest
  observed states, timestamps, coordinates, masks, and model/run metadata
  incrementally during evaluation.
- Cache stores remain marked incomplete while writing and are marked complete
  only after successful evaluation; the reader rejects incomplete stores by
  default.
- A Zarr 3.1.6 end-to-end smoke evaluation completed with 50 initializations,
  19 leads, a 106 × 115 grid, and 27 variables. Reopening the cache showed
  prediction and target arrays with shape `(50, 19, 106, 115, 27)`.
- A synthetic cache lifecycle test verified initialization, append, finish,
  and reopen. The implementation supports the repository's Zarr 2 pin and
  Zarr 3 installations.
- Existing W&B logging and test plots remain available.

### Still missing or incomplete

- The smoke cache has not yet been exhaustively checked for timestamp
  alignment, all coordinates/metadata, and value-by-value agreement with the
  tensors produced during inference. Schema, timestamps, source coordinates,
  masks, variable metadata, target values, and initial states have now been
  checked; the original prediction tensors were not retained separately for
  a direct element-by-element comparison.
- The offline evaluator now recomputes supported metrics from Zarr without
  loading a model. It reads one initialization at a time, uses training
  statistics, excludes boundary-forced cells, and reproduces the original
  non-SST smoke metrics.
- Offline output now includes derived physical-unit 10 m wind speed,
  configurable variable-group summaries, and configurable lead-group
  summaries. Group aggregation is documented in output metadata.
- Offline SST evaluation excludes land-dominant cells (`land_sea_mask >= 0.5`)
  and uses a training-split standard deviation over interior ocean-dominant
  cells. This corrects the legacy all-grid normalization for interpreting
  ocean SST, without changing the model's training/input normalization.
- Three older flat/root checkpoint files have datasets
  `regular_1level`, `regular`, and `hierarchical`, all use 3-hour steps, and
  have 54 grid embedder inputs. They are not compatible with the current
  6-hour ERA5 evaluation setup as-is.
- Four newer candidate checkpoints are present and have been mapped to
  1L-LAM (`graph_lam`, graph `1level`), Graphcast-LAM (`graph_lam`, graph
  `era5_multiscale`), HI-LAM (`hi_lam`, graph `era5_hierarchical`), and
  HI-LAM-Parallel (`hi_lam_parallel`, same hierarchical graph). All record
  dataset `era5`, 6-hour steps, four autoregressive training steps, and
  compatible 27-variable/grid input dimensions. Each checkpoint loaded and
  produced finite predictions with shape `(1, 19, 12190, 27)` for the same
  validation initialization.
- The absent `1level` and `era5_hierarchical` graph directories were rebuilt
  from the current ERA5 static coordinates with the repository's
  `create_mesh.py` settings. The model loads and forward passes succeed, but
  checkpoints do not store graph hashes, so bit-for-bit identity with the
  original training graph cannot be proven from checkpoint metadata alone.
- The new candidate checkpoints do not yet have full validation/test forecast
  caches or four-model metric comparisons. The existing full cache is only
  the GraphLAM smoke run.
- Full-grid coastal-land, inland-land, nearshore-sea, and offshore-sea masks
  and area-weighted regional MAE/RMSE/normalized metrics are implemented.
  Coast distance is measured between grid-cell centers and is an approximation
  to vector-coastline distance; the 25 km mask should be map-checked and
  sensitivity-tested before scientific interpretation.
- Offline evaluation now accepts per-run named regions defined by GeoJSON
  polygons or geographic bounding boxes, optionally intersected with land/sea
  classes and coast-distance bands. Custom region definitions replace the
  default bands for that evaluation and are saved in output metadata. The
  dataset does not supply country or named-sea boundaries; a suitable polygon
  must be provided for Norway-only/named-sea analysis.
- A no-inference preview command plots selected cells against cached land/sea
  classification and excludes boundary cells consistently with regional
  scoring. Use it to inspect each region definition before interpreting scores.
- Training-percentile event thresholds and area-weighted event verification
  are implemented for 10 m wind speed, 2 m temperature, and mean sea-level
  pressure. Precipitation remains disabled until the `tp` accumulation period
  is verified. Fractions Skill Score, per-cell error maps, and selected
  case-study exports are not implemented.
- Paired uncertainty estimates and a complete comparison/reporting workflow
  are not implemented.

The smoke test demonstrates software execution on a small subset; it does not
validate scientific results or establish that the transferred production
checkpoints are compatible with the evaluation dataset.

---

## Milestone 0 — Freeze a fair comparison protocol

**Priority: Essential before production inference**

- [x] Inventory the four newer candidate checkpoints. Keep the older flat
  `1L-LAM.ckpt`, `GC-LAM.ckpt`, and `Hi-LAM.ckpt` files separate; those legacy
  checkpoints use non-ERA5 datasets and 3-hour steps.
- [x] Confirm the newer candidates record ERA5, 6-hour cadence, expected
  architectures/graphs, and compatible model input dimensions; all four load
  and complete one finite validation forward pass.
- [x] Assign stable labels: `1L-LAM`, `Graphcast-LAM`, `HI-LAM`, and
  `HI-LAM-Parallel`. Record architecture and graph separately; the two
  GraphLAM variants share an architecture class but use different graphs.
- [ ] Select one evaluation split, common initialization-time schedule,
  maximum lead time, and forecast step for all models.
- [ ] Verify the dataset produces identical initialization and valid times for
  every model. Decide how to handle any case a model cannot forecast.
- [ ] Use validation data for development and reserve the test split for final
  comparisons.
- [ ] Record and verify the graph files used for training. The missing graph
  directories were recreated deterministically from the current ERA5 static
  coordinates, but no checkpoint graph fingerprint is available for direct
  equality verification.
- [ ] Record that the current experiments use reference ERA5 values on the
  forced boundary. Exclude these cells from skill calculations and describe
  results as boundary-forced LAM forecasts, not fully operational forecasts.

**Exit criteria:** A documented, reproducible protocol and four loadable
candidate checkpoints exist. Full cache alignment and graph identity checks
remain prerequisites for interpreting final rankings.

## Milestone 1 — Save a reusable Zarr forecast cache

**Priority: Essential; implemented, with verification still in progress**

### Define the cache schema

- [x] Add a configurable Zarr output path and cache settings to the evaluation
  configuration.
- [x] Store prediction data with named dimensions such as
  `(initialization, lead, y, x, variable)`.
- [x] Store physical-unit forecast values, variable names/units, initialization
  times, valid times, lead hours, latitude/longitude, model label, checkpoint
  identifier, dataset/split, graph, step length, and config/provenance.
- [x] Store the boundary mask and land–sea mask with the cache, or link to a
  versioned static-data source and record its identity.
- [x] Store the latest observed input state needed to reproduce persistence
  forecasts. Store ground truth either once in a shared, aligned group or
  separately if the cache format makes shared alignment unreliable.
- [x] Record that cached predictions include the model's reference-boundary
  forcing and preserve the mask used to exclude those cells.
- [x] Use float32 and appropriate chunking/compression. Keep dimensions and
  chunk sizes configurable only if actual memory/storage needs justify it.

### Write safely during inference

- [x] Write each inference batch to Zarr as it is produced; do not retain all
  full predictions in RAM.
- [x] Ensure lead coordinates are valid times relative to the stored
  initialization time and that they align exactly with target states.
- [x] Make cache writes detect duplicate or mismatched initialization times.
- [x] Handle interrupted/incomplete runs explicitly; do not mark a partial
  cache complete.
- [x] Support one model group per model/checkpoint, or a separate store per
  model with a common case index. Avoid concurrent writes to the same Zarr
  store unless the chosen implementation explicitly supports them.
- [x] Add a cache reader that exposes predictions, truth, initial states, masks,
  coordinates, and provenance without requiring model inference.

### Verify the cache

- [x] Run a small cache smoke test with one model and a few forecast
  initializations.
- [x] Reopen the store in a fresh process and verify shapes, dimension names,
  coordinates, units, initialization/valid-time alignment, masks, and model
  metadata. The smoke cache matched the ERA5 coordinates, static boundary and
  land–sea masks, canonical variable names/units, and its test-split
  initialization schedule.
- [x] Compare cached targets and latest observed states with the source
  dataset within float32 round-trip tolerance (`rtol=1e-6`, `atol=1e-5`).
- [ ] Compare cached predictions element-by-element with the tensors from the
  same inference batch. Those original tensors were not retained separately;
  reproducing all saved aggregate metrics provides a strong indirect check,
  but is not a direct value comparison.
- [ ] Confirm the store can be extended with another model without overwriting
  the first model or misaligning cases.
- [ ] Estimate full-run disk usage before launching all four evaluations.
  The current 50-initialization smoke cache occupies about 1.93 GiB. The
  configured full test split has 200 initializations, so a rough linear
  estimate is about 7.7 GiB per model and 31 GiB for four models; actual size
  depends on compression and forecast values. Check free space before a
  production run.

**Implementation status:** Cache writing and a small end-to-end smoke run are
complete. The exit criteria remain open until the unchecked data-integrity,
cross-model alignment, and production storage checks are performed.

## Milestone 2 — Build offline core metrics from the cache

**Priority: Essential; core evaluator implemented and checked**

- [x] Implement an offline evaluator that reads cached data and does not load a
  neural model.
- [x] Compute MAE/RMSE per model, variable, and lead.
- [x] Compute normalized MAE/RMSE using one documented set of training-split
  statistics shared across model comparisons. Keep physical-unit metrics
  alongside normalized metrics.
- [x] Audit how stored training statistics were calculated. They include all
  grid cells and boundary cells; `prepare_era5.py` fills missing SST with a
  constant before statistics are calculated.
- [x] Correct offline SST metric interpretation by excluding land-dominant
  cells and using ocean-dominant interior training SST spread. The model's
  saved training/input normalization remains unchanged.
- [x] Compute persistence forecasts from the cached latest observed states,
  with MAE/RMSE and skill scores on exactly the same cases and mask.
- [x] Add 10 m wind speed from `u10`/`v10`, calculated from physical-unit
  components.
- [x] Add clearly defined, configurable variable groups; preserve each
  variable's own result and report group aggregation rules.
- [x] Add configurable lead-time groups while retaining per-lead metrics.
- [x] Include valid sample counts, event/case counts where relevant, model and
  checkpoint identifiers, and configuration provenance in outputs.
- [x] Preserve metric values as missing when a denominator or sample count is
  zero; do not replace undefined scores with zero.

**Validation result:** Core-only offline recomputation from the 50-case smoke
cache produced 1,134 per-variable/per-lead metric records and matched the
inference-time CSV within
`rtol=2e-6`, `atol=1e-8` (maximum absolute difference about `4.8e-10`).

**Remaining Milestone 2 work:** validate common initialization coverage,
normalization compatibility, and metric fairness across compatible production
checkpoints.

## Milestone 3 — Implement geographic evaluation

**Priority: High**

- [x] Construct full-grid geographic masks using cached coordinates and the
  land–sea mask. The current mask is raster-based and does not use a vector
  coastline source.
- [x] Define coastal land, inland land, nearshore sea, and offshore sea using
  a configurable 25 km great-circle distance to the nearest opposite-class
  grid-cell center. This is explicitly an approximation to shoreline distance.
- [x] Allow named per-run regions from GeoJSON polygons or lon/lat bounding
  boxes, optionally combined with land/sea and coast-distance filters.
- [x] Add a map preview for configured masks, including the boundary-cell
  exclusion used by scoring.
- [ ] Add or supply a versioned Norway polygon and other desired geographic
  boundary data before running those country/sea-specific definitions.
- [x] The four land/sea subregions are disjoint; `all_grid` intentionally
  overlaps them as the full-domain reference.
- [ ] Add optional northern/southern Norway and terrain/elevation regions when
  reliable ancillary data is available.
- [x] Exclude boundary-forced cells in regional metrics.
- [x] Use cosine-of-latitude area weights for regional and event averages.
- [x] Compute per-variable, region, and lead MAE/RMSE and normalized metrics
  from cached physical-unit fields.
- [ ] Run the mask-preview command on the intended regions and inspect the
  output; run coastal-distance sensitivity checks before drawing conclusions
  about small coastal/inland differences.
- [ ] Save per-grid-cell error sums/counts or means for map creation without
  retaining all cases.
- [ ] Test region masks against plotted maps and verify that no cell is
  accidentally counted in the wrong land/sea category.

**Exit criteria:** Regional scores use the same cases, boundary mask, and
documented area-weighting policy. Map-ready per-cell error output remains open.

## Milestone 4 — Implement extreme-weather evaluation

**Priority: High**

- [x] Configure reproducible 99th/1st percentile thresholds from the training
  split over all interior grid cells; save exact thresholds, units, sample
  scope, and percentile settings in run metadata.
- [x] Detect high 10 m wind speed, high/low 2 m temperature, and low mean
  sea-level pressure.
- [ ] Verify `tp` accumulation semantics before implementing precipitation
  events. They are currently disabled, and the config rejects enabling them.
- [x] Calculate area-weighted POD, precision, false-alarm ratio, CSI, and
  frequency bias per event type, lead, and region.
- [x] Report MAE/RMSE restricted to observed event grid-cell samples.
- [ ] Flag metrics with too few events for reliable interpretation.
- [ ] Implement Fractions Skill Score for selected spatial events, with
  configurable neighborhood sizes and documented edge handling.
- [ ] Save selected case studies with forecast, truth, initialization/valid
  times, event definition, and model/checkpoint metadata.
- [ ] Report event counts and flag metrics with too few events for reliable
  interpretation.

**Exit criteria:** Core event metrics are reproducible from the cache and have
unit-, threshold-, lead-, region-, and sample-count information. Precipitation
and spatial-displacement claims remain blocked until their definitions/FSS
are validated.

## Milestone 5 — Validate fairness and quantify uncertainty

**Priority: Essential before final claims**

- [ ] Verify the ERA5 grid is 106 × 115 = 12,190 cells and coordinate ordering
  matches the cached data.
- [ ] Verify lead-time alignment and variable units against the source data.
- [x] Exclude land-dominant SST cells from offline SST metrics using the
  configured land-sea fraction policy.
- [ ] Reproduce the current MAE/RMSE results from cached predictions before
  trusting new scores.
- [ ] Use identical test initialization times, valid times, masks, region
  definitions, and event thresholds for all four models.
- [ ] Compare models with paired resampling over forecast initializations or
  suitable time blocks. Do not treat neighboring grid cells as independent
  samples.
- [ ] Report confidence intervals or uncertainty estimates for the model
  differences that support the main conclusions.
- [ ] Document missing cases, checkpoint/config provenance, and known
  boundary-forcing limitations.

**Exit criteria:** A reproducible comparison protocol supports the reported
model rankings and uncertainty.

## Milestone 6 — Generate comparisons and report figures offline

**Priority: High, after metric schemas stabilize**

- [ ] Generate lead-time curves for physical and normalized errors across all
  four models.
- [ ] Generate per-variable and normalized performance heatmaps.
- [ ] Generate variable-group and lead-group summaries without discarding the
  per-variable/per-lead results.
- [ ] Generate coastal/inland/nearshore/offshore comparisons.
- [ ] Generate per-model spatial error maps and paired model-difference maps
  with consistent scales.
- [ ] Generate event-detection, extreme-versus-ordinary, and FSS plots.
- [ ] Generate selected forecast case-study panels with truth and all four
  models for the same initialization and lead.
- [ ] If compute-cost comparisons are desired, record inference time and
  resource settings consistently for each model.
- [ ] Use consistent model colors, units, labels, scales, and sample counts.
- [ ] Support PNG, SVG, and PDF output and filters for model, variable, region,
  event threshold, and lead group.
- [ ] Confirm plots can be regenerated solely from stored caches/results.

**Exit criteria:** Report figures are reproducible from saved data without
rerunning inference.

## Recommended immediate sequence

1. Run a small validation cache for each of the four newer candidates using
   identical initialization times, leads, variables, and masks; compare their
   offline metric output and cache metadata.
2. Verify or document the recreated graph assets and inspect the configurable
   region masks. Acquire versioned country/sea polygons for any Norway-specific
   regions.
3. Before full inference, freeze the common case schedule and evaluation
   policy, estimate disk use, then cache each selected model.
4. Finish cross-model metric/fairness validation. Where possible, retain one
   inference batch during a future run to enable direct element-by-element
   validation of cached predictions.
5. Test coastal-band sensitivity; use a vector coastline if a precise
   shoreline distance is required. Verify the `tp` accumulation interval
   before enabling precipitation
   events; implement FSS and event-count reliability checks.
6. Build report plots and paired uncertainty estimates from the validated
   caches and result tables.

## Interpretation cautions

- Normalization makes different variable scales easier to compare, but one
  combined score still needs an explicit rule for weighting variables and
  groups. Do not let a large group of related variables dominate by count
  without intending that.
- Report physical-unit errors as well as normalized errors; they answer
  different questions.
- Skill relative to persistence is a useful baseline comparison, not a
  substitute for absolute error or event-specific verification.
- Results describe the current boundary-forced experiment. They should not be
  presented as performance without reference boundary information.
- A cached forecast is reusable only for analyses supported by the saved
  variables, cases, leads, units, masks, and precision. Preserve a little more
  data than the first set of planned plots needs, while estimating storage
  requirements in advance.

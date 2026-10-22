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

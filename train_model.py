import random
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities import seed
from argparse import ArgumentParser
import time
import matplotlib
# Figures are only logged to W&B. A GUI backend (Tk) crashes data loader workers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel

from neural_lam.weather_dataset import WeatherDataset
from neural_lam import constants, utils

MODELS = {"graph_lam": GraphLAM,
          "hi_lam": HiLAM,
          "hi_lam_parallel": HiLAMParallel}

def main():
    parser = ArgumentParser(description='Train or evaluate NeurWP models for LAM')

    # General options
    parser.add_argument('--dataset', type=str, default="meps_example",
        help='Dataset, corresponding to name in data directory (default: meps_example)')
    parser.add_argument('--model', type=str, default="graph_lam",
        help='Model architecture to train/evaluate (default: graph_lam)')
    parser.add_argument('--subset_ds', type=int, default=0,
        help='Use only a small subset of the dataset, for debugging (default: 0=false)')
    parser.add_argument('--seed', type=int, default=42,
        help='random seed (default: 42)')
    parser.add_argument('--n_workers', type=int, default=4,
        help='Number of workers in data loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
        help='upper epoch limit (default: 200)')
    parser.add_argument('--batch_size', type=int, default=4,
        help='batch size (default: 4)')
    parser.add_argument('--load', type=str,
        help='Path to load model parameters from (default: None)')
    parser.add_argument('--restore_opt', type=int, default=0,
        help='If optimizer state shoudl be restored with model (default: 0 (false))')
    parser.add_argument('--precision', type=str, default=32,
        help='Numerical precision to use for model (32/16/bf16) (default: 32)')

    # Model architecture
    parser.add_argument('--graph', type=str, default="multiscale",
        help='Graph to load and use in graph-based model (default: multiscale)')
    parser.add_argument('--hidden_dim', type=int, default=64,
        help='Dimensionality of all hidden representations (default: 64)')
    parser.add_argument('--hidden_layers', type=int, default=1,
        help='Number of hidden layers in all MLPs (default: 1)')
    parser.add_argument('--processor_layers', type=int, default=4,
        help='Number of GNN layers in processor GNN (default: 4)')
    parser.add_argument('--mesh_aggr', type=str, default="sum",
        help='Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)')

    # Training options
    parser.add_argument('--ar_steps', type=int, default=1,
        help='Number of steps to unroll prediction for in loss (1-19) (default: 1)')
    parser.add_argument('--control_only', type=int, default=0,
        help='Train only on control member of ensemble data (default: 0 (False))')
    parser.add_argument('--loss', type=str, default="mse",
        help='Loss function to use (default: mse)')
    parser.add_argument('--step_length', type=int, default=6,
        help='Step length in hours to consider single time step, '
            'multiple of 6 (default: 6)')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate (default: 0.001)')
    parser.add_argument('--val_interval', type=int, default=1,
        help='Number of epochs training between each validation run (default: 1)')
    parser.add_argument('--patience', type=int, default=30,
        help='Stop training when validation loss over the first ar_steps steps '
            'has not improved for this many '
            'validation runs, 0 to always train for all epochs (default: 30)')

    # Evaluation options
    parser.add_argument('--eval', type=str,
        help='Eval model on given data split (val/test) (default: None (train model))')
    parser.add_argument('--n_example_pred', type=int, default=1,
        help='Number of example predictions to plot during evaluation (default: 1)')
    args = parser.parse_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length % constants.raw_step_hours == 0, (
            f"Step length has to be a multiple of {constants.raw_step_hours} h")
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    subsample_step = args.step_length // constants.raw_step_hours
    max_pred_length = (constants.sample_length_raw // subsample_step) - 2  # 19

    train_loader = None
    val_loader = None

    if not args.eval:
        train_loader = torch.utils.data.DataLoader(
            WeatherDataset(
                args.dataset,
                pred_length=args.ar_steps,
                split="train",
                subsample_step=subsample_step,
                subset=bool(args.subset_ds),
                control_only=args.control_only,
            ),
            args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            WeatherDataset(
                args.dataset,
                pred_length=max_pred_length,
                split="val",
                subsample_step=subsample_step,
                subset=bool(args.subset_ds),
                control_only=args.control_only,
            ),
            args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )

    # Instatiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision("high") # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args,
                weights_only=False)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load,
                weights_only=False)["optimizer_states"][0]
    else:
        model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"\
            f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}", filename="min_val_loss",
            monitor="val_ar_loss", mode="min", save_last=True)
    callbacks = [checkpoint_callback]
    if args.patience > 0:
        # Stop when validation loss no longer improves
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val_ar_loss",
            mode="min", patience=args.patience))
    logger = pl.loggers.WandbLogger(project=constants.wandb_project, name=run_name,
            config=args)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         deterministic=True,
                         accelerator=device_name,
                         devices=1,
                         strategy="auto",
                         logger=logger,
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         check_val_every_n_epoch=args.val_interval,
                         precision=args.precision)

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger) # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else: # Test
            eval_loader = torch.utils.data.DataLoader(WeatherDataset(args.dataset,
                pred_length=max_pred_length, split="test",
                subsample_step=subsample_step, subset=bool(args.subset_ds)),
            args.batch_size, shuffle=False, num_workers=args.n_workers)

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

if __name__ == "__main__":
    main()

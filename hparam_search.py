"""
Hyperparameter tuning of train_model.py with Optuna.

Each trial runs train_model.py unchanged as a subprocess. The trial is scored by
the best val_mean_loss, read from the ModelCheckpoint state in the trial's
last.ckpt (rewritten whenever val_mean_loss improves). During training the
best score so far is reported to Optuna after every epoch, with epochs counted
from the progress bar in the trial log, so that Optuna can stop (prune) trials
that are clearly worse than earlier ones.

The study is stored in hparam_search/optuna.db, so an interrupted study is
resumed by running the same command again.

Each study tunes one model, given by --model. Arguments after "--" are passed
on to every trial, e.g.
    python hparam_search.py --model graph_lam --n_trials 30 --epochs 30 \
        -- --batch_size 4 --n_workers 8
"""
import os
import sys
import re
import glob
import time
import subprocess
from argparse import ArgumentParser, REMAINDER

import optuna
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "saved_models")
SEARCH_DIR = os.path.join(ROOT, "hparam_search")
POLL_SECONDS = 15
EPOCH_PATTERN = re.compile(rb"Epoch (\d+):")

# Default graph of each model. Hierarchical models need a hierarchical graph.
DEFAULT_GRAPHS = {
    "graph_lam": "era5_multiscale",
    "hi_lam": "era5_hierarchical",
    "hi_lam_parallel": "era5_hierarchical",
}


def suggest_config(trial, model, graph, fixed_flags):
    """
    Sample one configuration of train_model.py arguments.
    Arguments given as fixed arguments to the search are not sampled.
    """
    config = {"model": model, "graph": graph}

    def suggest(name, suggest_fn, *suggest_args, **suggest_kwargs):
        if f"--{name}" not in fixed_flags:
            config[name] = suggest_fn(name, *suggest_args, **suggest_kwargs)

    suggest("hidden_dim", trial.suggest_categorical, [32, 64, 128, 256])
    suggest("processor_layers", trial.suggest_int, 2, 8, step=2)
    suggest("hidden_layers", trial.suggest_int, 1, 2)
    suggest("lr", trial.suggest_float, 1e-4, 3e-3, log=True)
    if model == "graph_lam":
        # mesh_aggr is only used by graph_lam
        suggest("mesh_aggr", trial.suggest_categorical, ["sum", "mean"])
    return config


def to_cli(config):
    cli = []
    for key, val in config.items():
        if isinstance(val, float):
            val = f"{val:.3g}"
        cli += [f"--{key}", str(val)]
    return cli


def read_best_score(run_dir):
    """
    Read best val_mean_loss from last.ckpt in run directory,
    or None if not available yet
    """
    ckpt_path = os.path.join(run_dir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None # Checkpoint is being written, try again later
    for key, state in ckpt["callbacks"].items():
        if key.startswith("ModelCheckpoint") and \
                state.get("best_model_score") is not None:
            return float(state["best_model_score"])
    return None


class EpochCounter:
    """
    Count epochs started by a training run from the progress bar in its log
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.offset = 0
        self.epochs_started = 0

    def update(self):
        with open(self.log_path, "rb") as f:
            f.seek(self.offset)
            new_output = f.read()
        # Keep a few bytes in case a match is cut at the end
        self.offset += max(len(new_output) - 16, 0)
        epochs = [int(e) for e in EPOCH_PATTERN.findall(new_output)]
        if epochs:
            self.epochs_started = max(self.epochs_started, max(epochs) + 1)
        return self.epochs_started


def find_new_run_dir(existing_dirs):
    new_dirs = set(glob.glob(os.path.join(MODELS_DIR, "*"))) - existing_dirs
    return max(new_dirs, key=os.path.getmtime) if new_dirs else None


def make_objective(args, fixed_flags, study_dir):
    def objective(trial):
        config = suggest_config(trial, args.model, args.graph, fixed_flags)
        log_path = os.path.join(study_dir, f"trial_{trial.number:03d}.log")
        # Fixed args go last so they take precedence
        cmd = [sys.executable, os.path.join(ROOT, "train_model.py")] \
            + to_cli(config) \
            + ["--epochs", str(args.epochs), "--patience", str(args.patience)] \
            + args.train_args

        env = os.environ.copy()
        # Group the trials of a study in wandb, without touching train_model.py
        env["WANDB_RUN_GROUP"] = args.study
        env["WANDB_TAGS"] = f"hparam_search,{args.study}"

        print(f"\n=== Trial {trial.number} === {' '.join(to_cli(config))}")
        print(f"Log: {log_path}", flush=True)

        existing_dirs = set(glob.glob(os.path.join(MODELS_DIR, "*")))
        start = time.time()
        run_dir = None
        last_reported = 0
        best_score = None
        pruned = False
        epoch_counter = EpochCounter(log_path)
        with open(log_path, "w") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    time.sleep(POLL_SECONDS)
                    if args.timeout_h and \
                            time.time() - start > args.timeout_h*3600:
                        proc.terminate()
                        proc.wait()
                        trial.set_user_attr("timeout", True)
                        break
                    if run_dir is None:
                        run_dir = find_new_run_dir(existing_dirs)
                        if run_dir is None:
                            continue
                    # Validation of an epoch is done when the next one starts
                    epochs_done = epoch_counter.update() - 1
                    if epochs_done <= last_reported:
                        continue
                    best_score = read_best_score(run_dir) or best_score
                    if best_score is None:
                        continue

                    last_reported = epochs_done
                    trial.report(best_score, step=epochs_done)
                    if not args.no_prune and trial.should_prune():
                        pruned = True
                        proc.terminate()
                        proc.wait()
                        break
            except BaseException:
                # Do not leave training running if the search is stopped
                proc.terminate()
                proc.wait()
                raise

        minutes = (time.time() - start)/60
        run_dir = run_dir or find_new_run_dir(existing_dirs)
        best_score = (read_best_score(run_dir) if run_dir else None) \
            or best_score
        epochs = epoch_counter.update()
        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("minutes", round(minutes, 1))
        trial.set_user_attr("returncode", proc.returncode)
        trial.set_user_attr("cli", " ".join(to_cli(config)))
        trial.set_user_attr("epochs", epochs)

        if pruned:
            print(f"Trial {trial.number}: pruned after {epochs} epochs, "
                  f"val_mean_loss={best_score:.4f}")
            raise optuna.TrialPruned()
        if best_score is None:
            raise RuntimeError(f"Trial {trial.number} produced no validation "
                f"score (return code {proc.returncode}), see {log_path}")

        print(f"Trial {trial.number}: val_mean_loss={best_score:.4f}, "
              f"{epochs} epochs, {minutes:.0f} min")
        return best_score
    return objective


def print_leaderboard(study, top_k):
    trials = study.get_trials(deepcopy=False)
    counts = {s.name: sum(t.state == s for t in trials)
              for s in optuna.trial.TrialState}
    print(f"\nStudy {study.study_name}: " + ", ".join(
        f"{n} {s.lower()}" for s, n in counts.items() if n))
    done = sorted([t for t in trials if t.state ==
        optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    if not done:
        return

    print(f"{'trial':>5}  {'val_mean_loss':>13}  {'epochs':>6}  {'min':>5}  args")
    for t in done[:top_k]:
        print(f"{t.number:>5}  {t.value:>13.4f}  {t.user_attrs.get('epochs', '-'):>6}"
              f"  {t.user_attrs.get('minutes', 0):>5.0f}  {t.user_attrs.get('cli')}")
    print(f"\nBest run directory: {done[0].user_attrs.get('run_dir')}")

    if len(done) >= 2:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("Parameter importance: " + ", ".join(
                f"{k} {v:.2f}" for k, v in importance.items()))
        except Exception:
            pass # Not available for all studies, e.g. with too few trials


def main():
    parser = ArgumentParser(description="Hyperparameter tuning of "
        "train_model.py with Optuna. Arguments after '--' are passed to every "
        "trial.")
    parser.add_argument("--model", type=str, default="graph_lam",
        choices=list(DEFAULT_GRAPHS),
        help="Model to tune (default: graph_lam)")
    parser.add_argument("--graph", type=str,
        help="Graph to use, era5_multiscale for graph_lam and "
        "era5_hierarchical for the Hi-LAM models if not given")
    parser.add_argument("--study", type=str,
        help="Name of study, logs go in hparam_search/<study> "
        "(default: name of model)")
    parser.add_argument("--n_trials", type=int, default=20,
        help="Number of trials to run in this call (default: 20)")
    parser.add_argument("--epochs", type=int, default=30,
        help="Max epochs per trial (default: 30)")
    parser.add_argument("--patience", type=int, default=5,
        help="Early stopping patience per trial, 0 to disable (default: 5)")
    parser.add_argument("--timeout_h", type=float, default=0,
        help="Wall-clock limit per trial in hours, 0 for none (default: 0)")
    parser.add_argument("--no_prune", action="store_true",
        help="Do not stop unpromising trials early")
    parser.add_argument("--seed", type=int, default=0,
        help="Seed of the Optuna sampler (default: 0)")
    parser.add_argument("--summary", action="store_true",
        help="Only print the leaderboard of the study")
    parser.add_argument("--top_k", type=int, default=10,
        help="Number of trials in leaderboard (default: 10)")
    parser.add_argument("train_args", nargs=REMAINDER,
        help="Fixed arguments for train_model.py, given after '--'")
    args = parser.parse_args()

    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    for flag in ("--model", "--graph", "--epochs", "--patience", "--eval",
            "--load", "--resume"):
        assert flag not in args.train_args, f"{flag} can not be given after " \
            "'--', give it as an argument to hparam_search.py instead"
    args.graph = args.graph or DEFAULT_GRAPHS[args.model]
    args.study = args.study or args.model
    fixed_flags = {flag: val for flag, val in
        zip(args.train_args, args.train_args[1:] + [None])
        if flag.startswith("--")}

    study_dir = os.path.join(SEARCH_DIR, args.study)
    os.makedirs(study_dir, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{os.path.join(SEARCH_DIR, 'optuna.db')}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8),
        # Do not prune during the first trials or the first epochs of a trial
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
            n_warmup_steps=5),
    )

    # A study only covers one model and graph, so that its trials are comparable
    for key in ("model", "graph"):
        study_value = study.user_attrs.get(key)
        if study_value is None:
            study.set_user_attr(key, getattr(args, key))
        elif not args.summary:
            assert study_value == getattr(args, key), f"Study {args.study} " \
                f"tunes {key} {study_value}, use another --study name"

    if args.summary:
        print_leaderboard(study, args.top_k)
        return

    # Trials left running by an interrupted search are marked as failed
    for t in study.get_trials(deepcopy=False,
            states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    try:
        study.optimize(make_objective(args, fixed_flags, study_dir),
            n_trials=args.n_trials, catch=(RuntimeError,))
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to continue the study.")
    print_leaderboard(study, args.top_k)


if __name__ == "__main__":
    main()

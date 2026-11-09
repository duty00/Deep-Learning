import argparse
from copy import deepcopy
from pathlib import Path
import sys

from .config import load_config, pending_decisions, validate_config
from .tasks import TASKS
from .tracking import create_run


def apply_overrides(config, args):
    """Command-line overrides are written into the run's config.json snapshot, so runs stay traceable."""
    config = deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
        config["experiment_name"] = f"{config['experiment_name']}-s{args.seed}"
    if args.checkpoint is not None:
        config["model"]["checkpoint"] = args.checkpoint
    if args.split is not None:
        config["evaluation"]["split"] = args.split
    if args.micro_batch_size is not None:  # hardware-only change: effective batch_size stays the same
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.debug:
        config["debug"] = True
        config["experiment_name"] = f"{config['experiment_name']}-debug"
    validate_config(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="DL team research project")
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", required=True, help="Config path, relative to project root or absolute")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Run from repo root or set this explicitly")
    parser.add_argument("--seed", type=int, help="Override the training seed (e.g. 42/43/44)")
    parser.add_argument("--checkpoint", help="evaluate: checkpoint path (overrides model.checkpoint)")
    parser.add_argument("--split", choices=("validation", "test"), help="evaluate: split to score")
    parser.add_argument("--micro-batch-size", type=int, help="Images per forward pass (VRAM); batch_size unchanged")
    parser.add_argument("--debug", action="store_true", help="Tiny run (2 epochs x 8 batches) for sanity checks")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        config = apply_overrides(load_config(root / args.config), args)
        pending = pending_decisions(config)
        if args.command == "check":
            print("Configuration structure: OK")
            for item in pending:
                print(f"PENDING: {item}")
            print("Task implementation: " + ("registered" if config["task"] in TASKS else "not registered yet"))
            return 0
        if pending:
            raise ValueError("Complete project decisions first: " + ", ".join(pending))
        if config["task"] not in TASKS:
            raise ValueError("Implement a Task and register it in src/dl_project/tasks/__init__.py first.")
        if args.command == "evaluate":
            checkpoint = config["model"].get("checkpoint")
            if not checkpoint or not (root / checkpoint).is_file():
                raise ValueError("Set model.checkpoint (or --checkpoint) to an existing trained checkpoint.")
        for key in ("root", "split_manifest"):
            path = config["dataset"].get(key)
            if path and not (root / path).exists():
                raise ValueError(f"dataset.{key} not found: {root / path} (see scripts/download_data.md).")
        task = TASKS[config["task"]]()
        run_dir = create_run(config, root, args.command)
        print(f"Run directory: {run_dir}")
        operation = task.train if args.command == "train" else task.evaluate
        operation(config, root, run_dir)
        return 0
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

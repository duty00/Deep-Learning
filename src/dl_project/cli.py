import argparse
from pathlib import Path
import sys

from .config import load_config, pending_decisions
from .tasks import TASKS
from .tracking import create_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="DL team research project")
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", required=True, help="Config path, relative to project root or absolute")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Run from repo root or set this explicitly")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        config = load_config(root / args.config)
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
                raise ValueError("Set model.checkpoint to an existing trained checkpoint before evaluation.")
        task = TASKS[config["task"]]()
        run_dir = create_run(config, root, args.command)
        print(f"Run directory: {run_dir}")
        operation = task.train if args.command == "train" else task.evaluate
        operation(config, root, run_dir)
        return 0
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

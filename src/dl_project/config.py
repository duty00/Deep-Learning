"""JSON configuration and basic protocol checks; no ML dependencies required."""
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    for key in ("experiment_name", "task", "seed", "dataset", "model", "training", "evaluation", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    for key in ("experiment_name", "task", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    name = config["experiment_name"]
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError("experiment_name may contain only letters, digits, underscores and hyphens.")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer.")
    for key in ("dataset", "model", "training", "evaluation"):
        if not isinstance(config[key], dict):
            raise ValueError(f"{key} must be an object.")
    train = config["training"]
    for key in ("epochs", "batch_size"):
        if type(train.get(key)) is not int or train[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    # Optional gradient accumulation: batch_size is the effective batch (kept identical on every GPU),
    # micro_batch_size is what fits in VRAM; accumulation steps = batch_size // micro_batch_size.
    if "micro_batch_size" in train:
        micro = train["micro_batch_size"]
        if type(micro) is not int or micro <= 0 or train["batch_size"] % micro:
            raise ValueError("training.micro_batch_size must be a positive integer dividing training.batch_size.")
    lr = train.get("learning_rate")
    if type(lr) not in (int, float) or not 0 < lr < float("inf"):
        raise ValueError("training.learning_rate must be a finite positive number.")
    if train.get("selection_split") != "validation":
        raise ValueError("Select hyperparameters/checkpoints using validation, never test.")
    if config["evaluation"].get("split") not in ("validation", "test"):
        raise ValueError("evaluation.split must be validation or test.")


def pending_decisions(config):
    fields = {
        "task": config["task"],
        "dataset.name": config["dataset"].get("name"),
        "dataset.group_key (use 'none' with justification if independent)": config["dataset"].get("group_key"),
        "model.name": config["model"].get("name"),
        "evaluation.primary_metric": config["evaluation"].get("primary_metric"),
    }
    return [key for key, value in fields.items() if not value or value == "TBD"]

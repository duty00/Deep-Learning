"""Create an auditable run directory; results must be written by the task."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

TEST_LOG = Path("experiments/test_log.csv")
TEST_FIELDS = ("utc_time", "experiment", "seed", "checkpoint", "git_commit", "run_dir")


def logged_test_checkpoints(root):
    path = root / TEST_LOG
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {(root / row["checkpoint"]).resolve() for row in csv.DictReader(handle)}


@contextmanager
def final_test_guard(root, checkpoint, allow_repeat=False):
    """Hold the ledger lock across checking, scoring and logging to prevent concurrent repeats."""
    lock = (root / TEST_LOG).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"Final test ledger is locked ({lock}); check for a running evaluation.") from exc
    try:
        if checkpoint.resolve() in logged_test_checkpoints(root):
            if not allow_repeat:
                raise ValueError("This checkpoint is already in experiments/test_log.csv. "
                                 "Use --allow-repeat-test only if the repeat will be disclosed in the report.")
            print("WARNING: Disclose this repeated test evaluation in the report.", file=sys.stderr)
        yield
    finally:
        lock.rmdir()


def record_test(root, config, checkpoint, run_dir):
    path = root / TEST_LOG
    new = not path.exists() or path.stat().st_size == 0
    checkpoint = checkpoint.resolve()
    if checkpoint.is_relative_to(root):
        checkpoint = checkpoint.relative_to(root)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(TEST_FIELDS)
        writer.writerow([datetime.now(timezone.utc).isoformat(),
                         metrics.get("trained_experiment", config["experiment_name"]),
                         metrics.get("trained_seed", config["seed"]), checkpoint.as_posix(),
                         metadata["git_commit"], run_dir.relative_to(root).as_posix()])


def git_revision(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run(config, root, mode):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = root / config["output_dir"] / config["experiment_name"] / mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "git_commit": git_revision(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware_note": "Record GPU, VRAM and measurement settings in the run log.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata.get("Name"))
    (run_dir / "environment.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    return run_dir

"""Create an auditable run directory; results must be written by the task."""
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import platform
import subprocess
from uuid import uuid4


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

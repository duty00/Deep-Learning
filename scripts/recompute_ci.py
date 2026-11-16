"""Write lesion-level interval sidecars from existing HAM10000 validation predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl_project.data.ham10000 import CLASSES, RARE_CLASSES  # noqa: E402
from dl_project.metrics.classification import grouped_score_intervals  # noqa: E402


def recompute_ci(run_dir, lesion_by_image, n_boot):
    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    class_index = {name: i for i, name in enumerate(CLASSES)}
    intervals = grouped_score_intervals([class_index[r["y_true"]] for r in rows],
                                        [class_index[r["y_pred"]] for r in rows], list(CLASSES), RARE_CLASSES,
                                        [lesion_by_image[r["image_id"]] for r in rows], n_boot=n_boot)
    path = run_dir / "metrics_ci_lesion.json"
    path.write_text(json.dumps({"ci_resampling": "lesion", "n_boot": n_boot, **intervals}, indent=2) + "\n",
                    encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Validation evaluate directory (default: scan outputs/)")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    with args.metadata.open(newline="", encoding="utf-8-sig") as handle:
        lesion_by_image = {row["image_id"]: row["lesion_id"] for row in csv.DictReader(handle)}
    candidates = ([args.run_dir / "metrics.json"] if args.run_dir
                  else sorted(ROOT.glob("outputs/*/evaluate/*/metrics.json")))
    count = 0
    for metrics_path in candidates:
        if json.loads(metrics_path.read_text(encoding="utf-8"))["split"] != "validation":
            continue
        print(recompute_ci(metrics_path.parent, lesion_by_image, args.n_boot))
        count += 1
    if not count:
        parser.error("No saved validation evaluations found.")


if __name__ == "__main__":
    main()

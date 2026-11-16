"""Collect evaluate runs into mean ± std tables (CSV + LaTeX) for the report.

Scans outputs/*/evaluate/*/metrics.json (debug runs excluded), groups by the trained experiment
(seed suffix removed) and split, and writes reports/tables/results_<split>.csv/.tex plus a long-form
experiments/results.csv (one row per run and metric, with git commit and run path).
Usage: python scripts/aggregate_results.py [--split test]
"""
import argparse
import csv
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl_project.reporting import select_evaluation_runs  # noqa: E402

METRICS = ("macro_f1", "balanced_accuracy", "rare_recall", "mel_auroc", "ece", "accuracy")
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def load_commit_map(root):
    with (root / "experiments/COMMIT_MAP.csv").open(newline="", encoding="utf-8") as handle:
        return {row["recorded_commit"]: row["current_commit"] for row in csv.DictReader(handle)}


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "", None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}", mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    args = parser.parse_args()
    runs = select_evaluation_runs(ROOT)
    commit_map = load_commit_map(ROOT)
    with (ROOT / "experiments/results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment_id", "git_commit", "seed", "split", "metric", "value", "run_path",
                         "verified_by", "git_commit_current"])
        for run in runs:
            for metric in METRICS:
                writer.writerow([run["experiment"], run["commit"], run["seed"], run["split"], metric,
                                 run["metrics"].get(metric), run["run_dir"], "",
                                 commit_map.get(run["commit"], run["commit"])])
    selected = [run for run in runs if run["split"] == args.split]
    groups = {}
    for run in selected:
        groups.setdefault(run["experiment"], []).append(run)
    out = ROOT / "reports/tables"
    out.mkdir(parents=True, exist_ok=True)
    header = ["experiment", "n_seeds"] + list(METRICS) + [f"recall_{c}" for c in CLASSES]
    rows = []
    for experiment in sorted(groups):
        group = groups[experiment]
        row = {"experiment": experiment, "n_seeds": len({r["seed"] for r in group})}
        for metric in METRICS:
            row[metric] = fmt([r["metrics"].get(metric) for r in group])[0]
        for c in CLASSES:
            row[f"recall_{c}"] = fmt([r["metrics"]["per_class"][c]["recall"] for r in group])[0]
        rows.append(row)
    with (out / f"results_{args.split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    latex_cols = ["experiment", "n_seeds", "macro_f1", "balanced_accuracy", "rare_recall", "mel_auroc", "ece"]
    lines = [r"\begin{tabular}{l" + "c" * (len(latex_cols) - 1) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in latex_cols) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(row[c]).replace("_", r"\_").replace("±", r"$\pm$") for c in latex_cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"results_{args.split}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(runs)} evaluate runs; {len(rows)} experiments on {args.split} -> {out}")


if __name__ == "__main__":
    main()

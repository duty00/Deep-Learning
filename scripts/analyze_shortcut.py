"""Metadata-shortcut analysis: accuracy on metadata-typical vs metadata-atypical images (no GPU needed).

For every image, r = P_train(true class | site group x age group) / P_train(true class), with class
frequencies counted on the TRAIN split only. r < 1 means the patient's metadata makes the true class less
likely than average ("metadata-atypical"); r >= 1.5 means the metadata makes it more likely ("typical").
Groups were fixed before the test split was scored:
  site: head/neck (face, scalp, neck, ear) | torso (back, trunk, abdomen, chest) | limb/other
  age:  <40 | 40-59 | 60+ | unknown

Writes
  reports/tables/shortcut_subgroups_<split>.csv  seed-averaged accuracy per experiment and subset
  reports/tables/shortcut_pairs_<split>.csv      difference A - B with a lesion-level paired bootstrap CI
Usage: python scripts/analyze_shortcut.py [--split validation] [--n-boot 2000]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl_project.reporting import select_evaluation_runs  # noqa: E402

HEAD = {"face", "scalp", "neck", "ear"}
TORSO = {"back", "trunk", "abdomen", "chest"}
SUBSETS = [  # name, predicate on (r, y_true, age_group)
    ("all", lambda r, y, a: True),
    ("typical (r>=1.5)", lambda r, y, a: r >= 1.5),
    ("atypical (r<0.8)", lambda r, y, a: r < 0.8),
    ("atypical (r<0.67)", lambda r, y, a: r < 0.67),
    ("atypical (r<0.5)", lambda r, y, a: r < 0.5),
    ("mel, age<60", lambda r, y, a: y == "mel" and a in ("<40", "40-59")),
]
PAIRS = [  # (A, B, question): positive difference means A is better than B
    ("e02_concat_ce", "e01_image_only_ce", "concat metadata vs image only"),
    ("e06_film_ce", "e02_concat_ce", "FiLM+concat vs concat (CE)"),
    ("e07_film_la", "e05_concat_la", "FiLM+concat vs concat (LA)"),
    ("e14_filmonly_la", "e05_concat_la", "FiLM-only vs concat (LA)"),
    ("e15_concat_la_metadrop30", "e05_concat_la", "metadata dropout 0.3 vs none (LA)"),
    ("e11_effb0_film_la", "e13_effb0_concat_la", "EffNet: FiLM+concat vs concat (LA)"),
]


def site_group(location):
    location = (location or "").strip().lower()
    return "head" if location in HEAD else "torso" if location in TORSO else "limb/other"


def age_group(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if age != age:
        return "unknown"
    return "<40" if age < 40 else "40-59" if age < 60 else "60+"


def load_metadata():
    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))["splits"]
    with (ROOT / "data/raw/ham10000/HAM10000_metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = manifest.get(row["image_id"])
        row["group"] = (site_group(row["localization"]), age_group(row["age"]))
    return {row["image_id"]: row for row in rows}


def ratio_table(meta):
    train = [row for row in meta.values() if row["split"] == "train"]
    prior = Counter(row["dx"] for row in train)
    joint = Counter((row["group"], row["dx"]) for row in train)
    group_n = Counter(row["group"] for row in train)

    def ratio(row):
        g = row["group"]
        if not group_n[g]:
            return 1.0
        return (joint[(g, row["dx"])] / group_n[g]) / (prior[row["dx"]] / len(train))
    return ratio


def load_correct(split, meta):
    """{experiment: {image_id: mean correctness over seeds}}."""
    per_seed = defaultdict(list)
    for run in select_evaluation_runs(ROOT, split):
        with (ROOT / run["run_dir"] / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            per_seed[run["experiment"]].append({r["image_id"]: float(r["y_true"] == r["y_pred"])
                                                for r in csv.DictReader(handle)})
    out = {}
    for exp, seeds in per_seed.items():
        ids = set.intersection(*(set(s) for s in seeds))
        out[exp] = {i: float(np.mean([s[i] for s in seeds])) for i in ids}
    return out, {exp: len(seeds) for exp, seeds in per_seed.items()}


def paired_ci(a, b, ids, meta, n_boot, rng):
    lesions = sorted({meta[i]["lesion_id"] for i in ids})
    index = {lesion: j for j, lesion in enumerate(lesions)}
    diff_sum = np.zeros(len(lesions))
    count = np.zeros(len(lesions))
    for i in ids:
        j = index[meta[i]["lesion_id"]]
        diff_sum[j] += a[i] - b[i]
        count[j] += 1
    draws = rng.integers(0, len(lesions), size=(n_boot, len(lesions)))
    boot = diff_sum[draws].sum(1) / count[draws].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return diff_sum.sum() / count.sum(), low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    meta = load_metadata()
    ratio = ratio_table(meta)
    correct, n_seeds = load_correct(args.split, meta)
    if not correct:
        raise SystemExit(f"No evaluate runs found for split={args.split!r} under outputs/.")
    ids = sorted(set.intersection(*(set(c) for c in correct.values())))
    subsets = {name: [i for i in ids if pred(ratio(meta[i]), meta[i]["dx"], meta[i]["group"][1])]
               for name, pred in SUBSETS}
    tables = ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)

    out = tables / f"shortcut_subgroups_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "seeds"] + [f"{name} (n={len(s)})" for name, s in subsets.items()])
        for exp in sorted(correct):
            writer.writerow([exp, n_seeds[exp]] + [f"{np.mean([correct[exp][i] for i in s]):.4f}" if s else ""
                                                   for s in subsets.values()])
    print(f"-> {out.relative_to(ROOT)}")

    rng = np.random.default_rng(0)
    out = tables / f"shortcut_pairs_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A", "B", "question", "subset", "n_images", "diff_A_minus_B", "ci95_low", "ci95_high"])
        for a, b, question in PAIRS:
            if a not in correct or b not in correct:
                print(f"skip {a} vs {b}: missing runs", file=sys.stderr)
                continue
            for name, subset in subsets.items():
                if subset:
                    d, low, high = paired_ci(correct[a], correct[b], subset, meta, args.n_boot, rng)
                    writer.writerow([a, b, question, name, len(subset), f"{d:.4f}", f"{low:.4f}", f"{high:.4f}"])
    print(f"-> {out.relative_to(ROOT)} ({args.n_boot} lesion-level resamples)")


if __name__ == "__main__":
    main()

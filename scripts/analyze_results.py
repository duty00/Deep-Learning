"""Paired comparisons and report figures from saved predictions (no GPU needed).

Reads outputs/<exp>-s<seed>/evaluate/*/predictions.csv for the chosen split and writes:
  reports/tables/paired_bootstrap_<split>.csv   difference A - B with a lesion-level paired bootstrap CI
  reports/figures/recall_heatmap_<split>.png    per-class recall, every experiment (mean over seeds)
  reports/figures/confusion_<A>_vs_<B>_<split>.png   row-normalised confusion matrices (pooled over seeds)
  reports/figures/reliability_<split>.png       reliability diagram for a few experiments

Bootstrap: resample *lesions* (not images) with replacement, because images of one lesion are correlated;
the same resample is applied to every seed and to both experiments, then the metric is averaged over seeds.
Usage: python scripts/analyze_results.py [--split validation] [--n-boot 2000]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl_project.metrics.classification import expected_calibration_error  # noqa: E402
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RARE = [CLASSES.index(c) for c in ("akiec", "bcc", "df", "mel", "vasc")]
K = len(CLASSES)
PAIRS = [  # (A, B, question): positive difference means A is better than B
    ("e02_concat_ce", "e01_image_only_ce", "metadata: concat vs none"),
    ("e06_film_ce", "e02_concat_ce", "RQ1: FiLM vs concat (CE)"),
    ("e07_film_la", "e05_concat_la", "RQ1: FiLM vs concat (LA)"),
    ("e03_concat_wce", "e02_concat_ce", "RQ2: weighted CE vs CE"),
    ("e04_concat_focal", "e02_concat_ce", "RQ2: focal vs CE"),
    ("e05_concat_la", "e02_concat_ce", "RQ2: LA vs CE"),
    ("e05_concat_la", "e03_concat_wce", "RQ2: LA vs weighted CE"),
    ("e05_concat_la", "e04_concat_focal", "RQ2: LA vs focal"),
    ("e07_film_la", "e02_concat_ce", "proposed vs baseline"),
]
METRICS = ["macro_f1", "balanced_accuracy", "rare_recall", "recall_mel", "recall_nv"]
RELIABILITY = ["e02_concat_ce", "e04_concat_focal", "e05_concat_la"]
LABELS = {"e01_image_only_ce": "E1 image only · CE", "e02_concat_ce": "E2 concat · CE",
          "e03_concat_wce": "E3 concat · wCE", "e04_concat_focal": "E4 concat · focal",
          "e05_concat_la": "E5 concat · LA", "e06_film_ce": "E6 FiLM · CE", "e07_film_la": "E7 FiLM · LA"}
# Reference palette (dataviz skill): categorical slots 1-3, sequential blue, text and surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(split):
    """{experiment: [(seed, image_ids, y_true, probs)]}, latest evaluate run per seed, debug excluded."""
    runs = defaultdict(list)
    for exp_dir in sorted((ROOT / "outputs").glob("*-s*")):
        if exp_dir.name.endswith("-debug"):
            continue
        exp, seed = exp_dir.name.rsplit("-s", 1)
        candidates = []
        for pred in sorted(exp_dir.glob("evaluate/*/predictions.csv")):
            meta = (pred.parent / "metrics.json")
            if meta.exists() and f'"split": "{split}"' in meta.read_text(encoding="utf-8"):
                candidates.append(pred)
        if not candidates:
            continue
        with candidates[-1].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        ids = [r["image_id"] for r in rows]
        y = np.array([CLASSES.index(r["y_true"]) for r in rows])
        p = np.array([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
        runs[exp].append((int(seed), ids, y, p))
    return runs


def metrics_from_confusion(cm):
    """cm: (..., K, K) weighted confusion matrices (rows = true class). Returns dict of arrays."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    actual, predicted = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(actual > 0, tp / actual, 0.0)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"macro_f1": f1.mean(-1), "balanced_accuracy": recall.mean(-1),
            "rare_recall": recall[..., RARE].mean(-1), "recall_mel": recall[..., CLASSES.index("mel")],
            "recall_nv": recall[..., CLASSES.index("nv")], "recall": recall}


def weighted_confusion(y, pred, weights):
    """weights: (B, N) bootstrap multiplicities -> (B, K, K)."""
    onehot = np.zeros((len(y), K * K))
    onehot[np.arange(len(y)), y * K + pred] = 1.0
    return (weights @ onehot).reshape(-1, K, K)


def paired_bootstrap(runs, split, n_boot, seed=0):
    lesion_of = {}
    with (ROOT / "data/raw/ham10000/HAM10000_metadata.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lesion_of[row["image_id"]] = row["lesion_id"]
    ref_ids = next(iter(runs.values()))[0][1]
    for exp, items in runs.items():
        for _, ids, _, _ in items:
            if ids != ref_ids:
                raise ValueError(f"{exp}: image order differs from the other runs; cannot pair.")
    lesions = sorted({lesion_of[i] for i in ref_ids})
    index = {lesion: j for j, lesion in enumerate(lesions)}
    image_lesion = np.array([index[lesion_of[i]] for i in ref_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(lesions), size=(n_boot, len(lesions)))
    lesion_counts = np.stack([np.bincount(d, minlength=len(lesions)) for d in draws])
    weights = lesion_counts[:, image_lesion].astype(float)  # (B, N) image multiplicities
    ones = np.ones((1, len(ref_ids)))

    def seed_mean(exp, w):
        per_seed = [metrics_from_confusion(weighted_confusion(y, p.argmax(1), w)) for _, _, y, p in runs[exp]]
        return {m: np.mean([s[m] for s in per_seed], axis=0) for m in METRICS}

    cache = {}
    rows = []
    for a, b, question in PAIRS:
        if a not in runs or b not in runs:
            continue
        for exp in (a, b):
            if exp not in cache:
                cache[exp] = (seed_mean(exp, ones), seed_mean(exp, weights))
        for metric in METRICS:
            point = float(cache[a][0][metric][0] - cache[b][0][metric][0])
            diffs = cache[a][1][metric] - cache[b][1][metric]
            low, high = np.quantile(diffs, [0.025, 0.975])
            p_two = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
            rows.append({"question": question, "A": a, "B": b, "metric": metric, "diff_A_minus_B": f"{point:+.4f}",
                         "ci95_low": f"{low:+.4f}", "ci95_high": f"{high:+.4f}", "p_bootstrap": f"{p_two:.3f}",
                         "significant": "yes" if low > 0 or high < 0 else "no"})
    out = ROOT / f"reports/tables/paired_bootstrap_{split}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(lesions)} lesions, {len(ref_ids)} images, {n_boot} lesion-level resamples -> {out}")
    for r in rows:
        if r["metric"] in ("macro_f1", "rare_recall", "recall_mel"):
            print(f"  {r['question']:<28} {r['metric']:<12} {r['diff_A_minus_B']} "
                  f"[{r['ci95_low']}, {r['ci95_high']}] p={r['p_bootstrap']} {r['significant']}")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)


def recall_heatmap(runs, split, plt, cmap):
    exps = [e for e in LABELS if e in runs]
    recall = np.array([np.mean([metrics_from_confusion(weighted_confusion(y, p.argmax(1), np.ones((1, len(y)))))
                                ["recall"][0] for _, _, y, p in runs[e]], axis=0) for e in exps])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(exps) + 1.2), dpi=200)
    # Recalls all sit in ~0.5-1.0; a 0-1 scale would paint every cell the same dark blue.
    ax.imshow(recall, cmap=cmap, vmin=0.5, vmax=1, aspect="auto")
    for i in range(len(exps)):
        for j in range(K):
            ax.text(j, i, f"{recall[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if recall[i, j] > 0.8 else TEXT)
    ax.set_xticks(range(K), CLASSES)
    ax.set_yticks(range(len(exps)), [LABELS[e] for e in exps])
    ax.tick_params(length=0, colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Per-class recall ({split}, mean of 3 seeds)", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    path = ROOT / f"reports/figures/recall_heatmap_{split}.png"
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def confusion_pair(runs, split, a, b, plt, cmap):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=200)
    for ax, exp in zip(axes, (a, b)):
        cm = sum(weighted_confusion(y, p.argmax(1), np.ones((1, len(y))))[0] for _, _, y, p in runs[exp])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if norm[i, j] > 0.6 else (TEXT if norm[i, j] >= 0.005 else GRID))
        ax.set_xticks(range(K), CLASSES, rotation=45)
        ax.set_yticks(range(K), CLASSES)
        ax.set_xlabel("predicted", fontsize=9, color=TEXT2)
        ax.tick_params(length=0, colors=TEXT, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(LABELS[exp], fontsize=10, color=TEXT, loc="left")
    axes[0].set_ylabel("true class", fontsize=9, color=TEXT2)
    fig.suptitle(f"Row-normalised confusion matrix ({split}, pooled over 3 seeds)", fontsize=10, color=TEXT,
                 x=0.01, ha="left")
    fig.tight_layout()
    path = ROOT / f"reports/figures/confusion_{a.split('_')[0]}_vs_{b.split('_')[0]}_{split}.png"
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def reliability(runs, split, plt, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=200)
    ax.plot([0, 1], [0, 1], color=TEXT2, lw=1, ls=(0, (3, 3)), label="perfect calibration")
    edges = np.linspace(0, 1, n_bins + 1)
    markers = ["o", "s", "^"]
    for colour, marker, exp in zip(SERIES, markers, [e for e in RELIABILITY if e in runs]):
        conf = np.concatenate([p.max(1) for _, _, _, p in runs[exp]])
        correct = np.concatenate([(p.argmax(1) == y) for _, _, y, p in runs[exp]])
        bins = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for k in range(n_bins):
            mask = bins == k
            if mask.sum() >= 15:  # skip near-empty bins
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        # Same ECE as the results table (project implementation, 15 bins, mean over seeds).
        ece = np.mean([expected_calibration_error(p, y) for _, _, y, p in runs[exp]])
        ax.plot(xs, ys, color=colour, lw=2, marker=marker, ms=6, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=f"{LABELS[exp]}  (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (max softmax probability)", fontsize=9, color=TEXT2)
    ax.set_ylabel("accuracy", fontsize=9, color=TEXT2)
    ax.grid(color=GRID, lw=0.6)
    style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    ax.set_title(f"Reliability diagram ({split}, pooled over 3 seeds)", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    path = ROOT / f"reports/figures/reliability_{split}.png"
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    runs = load_runs(args.split)
    if not runs:
        raise SystemExit(f"No {args.split} predictions under outputs/.")
    print("experiments:", {e: sorted(s for s, *_ in v) for e, v in runs.items()})
    paired_bootstrap(runs, args.split, args.n_boot)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": "#fcfcfb"})
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    paths = [recall_heatmap(runs, args.split, plt, cmap), reliability(runs, args.split, plt)]
    if "e02_concat_ce" in runs and "e05_concat_la" in runs:
        paths.append(confusion_pair(runs, args.split, "e02_concat_ce", "e05_concat_la", plt, cmap))
    for path in paths:
        print("figure:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

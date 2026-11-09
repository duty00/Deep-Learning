"""Classification metrics (PLAN_BTL.md section 5.9). NumPy (+ scikit-learn for AUROC) only.

Definitions (all computed on the same predictions for every method):
* macro_f1 (PRIMARY): unweighted mean of per-class F1; every class counts equally, so the dominant
  ``nv`` class cannot hide failures on rare classes. Higher is better. A class with no prediction and no
  sample would get F1 = 0 (does not occur on HAM10000 splits).
* balanced_accuracy: mean per-class recall.
* rare_recall: mean recall of the pre-registered rare classes (akiec, bcc, df, mel, vasc).
* mel_auroc: one-vs-rest AUROC for melanoma from softmax probabilities (threshold-free).
* ece: expected calibration error of the top-1 confidence, 15 equal-width bins. Lower is better.
Limitations: none of these weighs errors by clinical cost (e.g. missing a melanoma vs. a false alarm);
with only ~20 test images of df/vasc, their per-class scores have high variance (see ``bootstrap_ci``).
"""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_scores(matrix):
    tp = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, actual.astype(int)


def expected_calibration_error(probs, y_true, n_bins=15):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_report(y_true, probs, class_names, rare_classes=(), positive_class="mel"):
    """All metrics as a JSON-serialisable dict. ``probs``: (N, K) softmax probabilities."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    k = len(class_names)
    if probs.ndim != 2 or probs.shape[1] != k or len(y_true) != len(probs):
        raise ValueError("probs must be (N, num_classes) aligned with y_true.")
    y_pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, k)
    precision, recall, f1, support = per_class_scores(matrix)
    rare_idx = [class_names.index(name) for name in rare_classes]
    report = {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "rare_recall": float(recall[rare_idx].mean()) if rare_idx else None,
        "ece": expected_calibration_error(probs, y_true),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
        "confusion_matrix": matrix.tolist(),
    }
    report[f"{positive_class}_auroc"] = _binary_auroc(y_true == class_names.index(positive_class),
                                                      probs[:, class_names.index(positive_class)]) \
        if positive_class in class_names else None
    return report


def _binary_auroc(is_positive, score):
    is_positive = np.asarray(is_positive, dtype=bool)
    if is_positive.all() or not is_positive.any():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(is_positive, score))


def bootstrap_ci(y_true, probs, metric="macro_f1", class_names=None, n_boot=1000, seed=0, alpha=0.05,
                 rare_classes=()):
    """Percentile bootstrap CI of a scalar metric over test images (resampling with replacement)."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    class_names = list(class_names or range(probs.shape[1]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        value = classification_report(y_true[idx], probs[idx], class_names, rare_classes,
                                      positive_class=None)[metric]
        values.append(value)
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)

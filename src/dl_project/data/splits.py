"""Group-aware train/validation/test split for HAM10000 (PLAN_BTL.md section 5.4).

All images of one lesion (``lesion_id``) go to exactly one split, so near-identical
views of the same lesion can never appear in both training and evaluation data.
Splitting uses scikit-learn ``StratifiedGroupKFold`` (groups = ``lesion_id``,
stratified by ``dx``): the data is cut into ``n_folds`` group-disjoint folds of
~1/n_folds each, then whole folds are assigned to train/validation/test according
to the ratios (default 20 folds -> 14/3/3 folds = 70/15/15).

The resulting manifest (``data/splits/manifest.json``) is committed to Git and is the
single source of truth for every experiment; training seeds (42/43/44) vary, the
split does not. Only the split creation needs scikit-learn (imported lazily);
reading/verifying a manifest uses the standard library only.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import random

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_N_FOLDS = 20
DEFAULT_SPLIT_SEED = 42
REQUIRED_COLUMNS = ("image_id", "lesion_id", "dx")
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- I/O
def read_metadata(path, required=REQUIRED_COLUMNS):
    """Read HAM10000 metadata (.csv, or the Dataverse tab-separated .tab export).

    Returns a list of dicts with stripped string values, sorted by ``image_id`` so the
    split never depends on the row order of the downloaded file.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = [name.strip().strip('"') for name in (reader.fieldnames or [])]
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}; found {columns}")
        reader.fieldnames = columns
        records = [{key: (value or "").strip().strip('"') for key, value in row.items() if key is not None}
                   for row in reader]
    if not records:
        raise ValueError(f"{path}: no data rows.")
    for line, record in enumerate(records, start=2):
        empty = [name for name in required if not record.get(name)]
        if empty:
            raise ValueError(f"{path}: row {line} has empty value(s) for {empty}.")
    duplicates = [image for image, count in Counter(r["image_id"] for r in records).items() if count > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicated image_id values, e.g. {sorted(duplicates)[:5]}.")
    return sorted(records, key=lambda record: record["image_id"])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- splitting
def fold_allocation(ratios=None, n_folds=DEFAULT_N_FOLDS):
    """Number of folds per split; every ratio must be a whole number of folds."""
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must have exactly the keys {SPLIT_NAMES}.")
    if type(n_folds) is not int or n_folds < len(SPLIT_NAMES):
        raise ValueError("n_folds must be an integer >= 3.")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.")
    allocation = {}
    for name in SPLIT_NAMES:
        exact = ratios[name] * n_folds
        folds = round(exact)
        if abs(exact - folds) > 1e-6 or folds < 1:
            raise ValueError(f"ratio {name}={ratios[name]} is not a positive multiple of 1/{n_folds}.")
        allocation[name] = folds
    return allocation


def make_group_splits(records, seed=DEFAULT_SPLIT_SEED, ratios=None, n_folds=DEFAULT_N_FOLDS,
                      group_key="lesion_id", stratify_key="dx"):
    """Return ``{image_id: split}`` with every group confined to a single split."""
    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('Creating splits needs scikit-learn: pip install -e ".[data]"') from exc

    allocation = fold_allocation(ratios, n_folds)
    records = sorted(records, key=lambda record: record["image_id"])
    labels = [record[stratify_key] for record in records]
    groups = [record[group_key] for record in records]
    if len(set(groups)) < n_folds:
        raise ValueError(f"Only {len(set(groups))} groups; need at least n_folds={n_folds}.")

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_index = [None] * len(records)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros((len(records), 1)), labels, groups)):
        for index in held_out:
            fold_of_index[index] = fold

    # Randomly decide which folds form test/validation/train (seeded, independent of fold order).
    fold_ids = list(range(n_folds))
    random.Random(seed).shuffle(fold_ids)
    split_of_fold, start = {}, 0
    for name in ("test", "validation", "train"):
        for fold in fold_ids[start:start + allocation[name]]:
            split_of_fold[fold] = name
        start += allocation[name]

    assignments = {record["image_id"]: split_of_fold[fold_of_index[i]] for i, record in enumerate(records)}
    verify_no_group_leakage(records, assignments, group_key)
    return assignments


# ------------------------------------------------------------------ validation
def verify_no_group_leakage(records, assignments, group_key="lesion_id"):
    """Raise ValueError if any image is unassigned/unknown or a group spans >1 split."""
    image_ids = {record["image_id"] for record in records}
    unassigned = sorted(image_ids - set(assignments))
    unknown = sorted(set(assignments) - image_ids)
    invalid = sorted({split for split in assignments.values() if split not in SPLIT_NAMES})
    problems = []
    if unassigned:
        problems.append(f"{len(unassigned)} image(s) without split, e.g. {unassigned[:5]}")
    if unknown:
        problems.append(f"{len(unknown)} manifest image(s) not in metadata, e.g. {unknown[:5]}")
    if invalid:
        problems.append(f"invalid split name(s) {invalid}")
    splits_of_group = defaultdict(set)
    for record in records:
        if record["image_id"] in assignments:
            splits_of_group[record[group_key]].add(assignments[record["image_id"]])
    leaking = sorted(group for group, splits in splits_of_group.items() if len(splits) > 1)
    if leaking:
        problems.append(f"{len(leaking)} {group_key}(s) in more than one split, e.g. {leaking[:5]}")
    if problems:
        raise ValueError("Split check failed: " + "; ".join(problems))


def split_summary(records, assignments, group_key="lesion_id", stratify_key="dx"):
    """Per-split counts of images, groups and classes (for the manifest and the report)."""
    summary = {}
    total = len(records)
    for name in SPLIT_NAMES:
        subset = [record for record in records if assignments.get(record["image_id"]) == name]
        summary[name] = {
            "images": len(subset),
            "groups": len({record[group_key] for record in subset}),
            "image_fraction": round(len(subset) / total, 4) if total else 0.0,
            "class_counts": dict(sorted(Counter(record[stratify_key] for record in subset).items())),
        }
    return summary


def groups_with_multiple_labels(records, group_key="lesion_id", stratify_key="dx"):
    labels = defaultdict(set)
    for record in records:
        labels[record[group_key]].add(record[stratify_key])
    return sorted(group for group, values in labels.items() if len(values) > 1)


# -------------------------------------------------------------------- manifest
def assignments_checksum(assignments):
    canonical = json.dumps(dict(sorted(assignments.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(records, assignments, *, dataset, source_file, source_sha256, seed,
                   ratios=None, n_folds=DEFAULT_N_FOLDS, group_key="lesion_id", stratify_key="dx",
                   sklearn_version=None):
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "source": {"metadata_file": str(source_file).replace("\\", "/"), "sha256": source_sha256,
                   "rows": len(records)},
        "method": {
            "splitter": "sklearn.model_selection.StratifiedGroupKFold",
            "sklearn_version": sklearn_version,
            "group_key": group_key,
            "stratify_key": stratify_key,
            "n_folds": n_folds,
            "fold_allocation": fold_allocation(ratios, n_folds),
            "ratios": ratios,
            "seed": seed,
            "note": "Split seed is fixed for all experiments; training seeds vary independently.",
        },
        "summary": split_summary(records, assignments, group_key, stratify_key),
        "groups_with_multiple_labels": groups_with_multiple_labels(records, group_key, stratify_key),
        "assignments_sha256": assignments_checksum(assignments),
        "splits": dict(sorted(assignments.items())),
    }


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_manifest(path):
    """Load a manifest and check its version, split names and assignment checksum."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: unsupported manifest_version {manifest.get('manifest_version')}.")
    assignments = manifest.get("splits")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError(f"{path}: 'splits' must be a non-empty object.")
    invalid = sorted({split for split in assignments.values() if split not in SPLIT_NAMES})
    if invalid:
        raise ValueError(f"{path}: invalid split name(s) {invalid}.")
    if assignments_checksum(assignments) != manifest.get("assignments_sha256"):
        raise ValueError(f"{path}: assignments_sha256 mismatch; manifest was edited or corrupted.")
    return manifest


def split_image_ids(manifest, split):
    """Sorted image ids of one split from a loaded manifest."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}.")
    return sorted(image for image, name in manifest["splits"].items() if name == split)

"""Create or verify the HAM10000 lesion-grouped split manifest (PLAN_BTL.md section 5.4).

Run from the repository root after ``pip install -e ".[data]"``:

    python scripts/make_splits.py                # create data/splits/manifest.json
    python scripts/make_splits.py --verify       # re-check an existing manifest

The manifest is committed to Git; regenerate it only deliberately (``--force``) and
never after looking at test results.
"""
import argparse
from pathlib import Path
import sys

from dl_project.data.splits import (
    DEFAULT_N_FOLDS, DEFAULT_SPLIT_SEED, SPLIT_NAMES, build_manifest, file_sha256, load_manifest,
    make_group_splits, read_metadata, save_manifest, verify_no_group_leakage,
)

DEFAULT_METADATA = "data/raw/ham10000/HAM10000_metadata.csv"
DEFAULT_OUTPUT = "data/splits/manifest.json"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="HAM10000 metadata .csv/.tab")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Manifest path")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Relative paths start here")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Split seed (fixed for all runs)")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--verify", action="store_true", help="Verify an existing manifest instead of writing")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    return parser.parse_args(argv)


def print_summary(manifest):
    summary = manifest["summary"]
    classes = sorted({name for split in summary.values() for name in split["class_counts"]})
    header = f"{'split':<11}{'images':>8}{'lesions':>9}{'frac':>7}  " + " ".join(f"{c:>6}" for c in classes)
    print(header)
    for split in SPLIT_NAMES:
        row = summary[split]
        counts = " ".join(f"{row['class_counts'].get(c, 0):>6}" for c in classes)
        print(f"{split:<11}{row['images']:>8}{row['groups']:>9}{row['image_fraction']:>7.3f}  {counts}")
    multi = manifest.get("groups_with_multiple_labels") or []
    if multi:
        print(f"WARNING: {len(multi)} lesion_id(s) have more than one dx label, e.g. {multi[:5]}")


def relative(path, root):
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.project_root.resolve()
    metadata_path = root / args.metadata
    output_path = root / args.output
    ratios = dict(zip(SPLIT_NAMES, args.ratios))
    try:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}. See scripts/download_data.md.")
        records = read_metadata(metadata_path)
        source_sha = file_sha256(metadata_path)

        if args.verify:
            manifest = load_manifest(output_path)
            if manifest["source"]["sha256"] != source_sha:
                raise ValueError("Metadata file differs from the one used to build the manifest (sha256).")
            method = manifest["method"]
            verify_no_group_leakage(records, manifest["splits"], method["group_key"])
            # The committed manifest is the single source of truth; regeneration is only a reproducibility
            # check. StratifiedGroupKFold's assignment can change across scikit-learn versions, so a
            # mismatch is an error only when the installed version equals the one that built the manifest.
            try:
                import sklearn
            except ImportError:
                sklearn = None
            reproducible = "skipped (scikit-learn not installed)"
            if sklearn is not None:
                regenerated = make_group_splits(records, seed=method["seed"], ratios=method["ratios"],
                                                n_folds=method["n_folds"], group_key=method["group_key"],
                                                stratify_key=method["stratify_key"])
                if regenerated == manifest["splits"]:
                    reproducible = "reproducible"
                elif sklearn.__version__ == method.get("sklearn_version"):
                    raise ValueError("Regenerated split differs from manifest with the SAME scikit-learn version "
                                     f"({sklearn.__version__}): metadata or split code changed.")
                else:
                    reproducible = (f"not regenerated identically (manifest: scikit-learn "
                                    f"{method.get('sklearn_version')}, installed {sklearn.__version__}); "
                                    "using the committed manifest as-is")
                    print(f"WARNING: split {reproducible}.", file=sys.stderr)
            print(f"OK: {output_path} verified (checksum, source sha256, no lesion leakage; {reproducible}).")
            print_summary(manifest)
            return 0

        if output_path.exists() and not args.force:
            raise FileExistsError(f"{output_path} exists. Use --verify, or --force to deliberately regenerate.")
        import sklearn
        assignments = make_group_splits(records, seed=args.seed, ratios=ratios, n_folds=args.n_folds)
        manifest = build_manifest(records, assignments, dataset="ham10000",
                                  source_file=relative(metadata_path, root), source_sha256=source_sha,
                                  seed=args.seed, ratios=ratios, n_folds=args.n_folds,
                                  sklearn_version=sklearn.__version__)
        save_manifest(manifest, output_path)
        print(f"Wrote {output_path}")
        print_summary(manifest)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

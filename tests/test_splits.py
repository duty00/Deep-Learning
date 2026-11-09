"""Tests for the lesion-grouped HAM10000 split (PLAN_BTL.md sections 0.5 and 5.4).

Synthetic metadata mimics HAM10000: 7 imbalanced classes, 1-4 images per lesion.
Tests needing scikit-learn are skipped when it is not installed (pip install -e ".[data]").
If the real metadata and manifest exist locally, they are checked for leakage too.
"""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from dl_project.data.splits import (
    SPLIT_NAMES, assignments_checksum, build_manifest, file_sha256, fold_allocation, load_manifest,
    read_metadata, save_manifest, split_image_ids, split_summary, verify_no_group_leakage,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
COLUMNS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
# Roughly HAM10000-like lesion counts per class (strong imbalance, rare df/vasc).
LESIONS_PER_CLASS = {"nv": 900, "mel": 160, "bkl": 150, "bcc": 70, "akiec": 45, "vasc": 25, "df": 22}


def synthetic_records(seed=0):
    """One dx per lesion, 1-4 images per lesion; ids shuffled so sorting by id mixes classes."""
    rng = random.Random(seed)
    lesions = [(dx, index) for dx, count in LESIONS_PER_CLASS.items() for index in range(count)]
    lesion_numbers = rng.sample(range(10**6), len(lesions))
    records = []
    for (dx, _), number in zip(lesions, lesion_numbers):
        for _ in range(rng.choice([1, 1, 1, 2, 2, 3, 4])):
            records.append({"lesion_id": f"HAM_{number:07d}", "image_id": None, "dx": dx,
                            "dx_type": "histo", "age": str(rng.randrange(5, 90)),
                            "sex": rng.choice(["male", "female"]), "localization": "back"})
    for record, number in zip(records, rng.sample(range(10**7), len(records))):
        record["image_id"] = f"ISIC_{number:07d}"
    return records


def write_csv(path, records, delimiter=","):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)


def load_make_splits_script():
    spec = importlib.util.spec_from_file_location("make_splits_script", ROOT / "scripts" / "make_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetadataAndValidationTests(unittest.TestCase):
    """Standard-library parts: always run (also in CI without scikit-learn)."""

    def test_read_metadata_csv_and_tab_sorted_by_image_id(self):
        records = synthetic_records()[:50]
        with TemporaryDirectory() as temp:
            for name, delimiter in (("meta.csv", ","), ("meta.tab", "\t")):
                path = Path(temp) / name
                write_csv(path, list(reversed(records)), delimiter)
                loaded = read_metadata(path)
                self.assertEqual([r["image_id"] for r in loaded], sorted(r["image_id"] for r in records))
                self.assertEqual(set(loaded[0]), set(COLUMNS))

    def test_read_metadata_rejects_bad_files(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "meta.csv"
            path.write_text("image_id,dx\nISIC_1,nv\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lesion_id"):
                read_metadata(path)
            path.write_text("lesion_id,image_id,dx\nHAM_1,ISIC_1,nv\nHAM_2,ISIC_1,mel\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated image_id"):
                read_metadata(path)
            path.write_text("lesion_id,image_id,dx\n,ISIC_1,nv\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                read_metadata(path)

    def test_fold_allocation(self):
        self.assertEqual(fold_allocation(), {"train": 14, "validation": 3, "test": 3})
        with self.assertRaises(ValueError):
            fold_allocation({"train": 0.7, "validation": 0.2, "test": 0.15})  # sum != 1
        with self.assertRaises(ValueError):
            fold_allocation({"train": 0.72, "validation": 0.14, "test": 0.14})  # not k/20
        with self.assertRaises(ValueError):
            fold_allocation({"train": 0.7, "val": 0.15, "test": 0.15})

    def test_leakage_check_detects_shared_lesion(self):
        records = [{"image_id": "a", "lesion_id": "L1", "dx": "nv"},
                   {"image_id": "b", "lesion_id": "L1", "dx": "nv"},
                   {"image_id": "c", "lesion_id": "L2", "dx": "mel"}]
        verify_no_group_leakage(records, {"a": "train", "b": "train", "c": "test"})
        with self.assertRaisesRegex(ValueError, "L1"):
            verify_no_group_leakage(records, {"a": "train", "b": "test", "c": "test"})

    def test_leakage_check_detects_missing_unknown_and_invalid(self):
        records = [{"image_id": "a", "lesion_id": "L1", "dx": "nv"},
                   {"image_id": "b", "lesion_id": "L2", "dx": "nv"}]
        with self.assertRaisesRegex(ValueError, "without split"):
            verify_no_group_leakage(records, {"a": "train"})
        with self.assertRaisesRegex(ValueError, "not in metadata"):
            verify_no_group_leakage(records, {"a": "train", "b": "test", "z": "test"})
        with self.assertRaisesRegex(ValueError, "invalid split"):
            verify_no_group_leakage(records, {"a": "train", "b": "val"})

    def test_manifest_roundtrip_and_tamper_detection(self):
        records = [{"image_id": "a", "lesion_id": "L1", "dx": "nv"},
                   {"image_id": "b", "lesion_id": "L2", "dx": "mel"},
                   {"image_id": "c", "lesion_id": "L3", "dx": "nv"}]
        assignments = {"a": "train", "b": "validation", "c": "test"}
        manifest = build_manifest(records, assignments, dataset="toy", source_file="meta.csv",
                                  source_sha256="0" * 64, seed=42)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "splits" / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path)
            self.assertEqual(loaded["splits"], assignments)
            self.assertEqual(split_image_ids(loaded, "train"), ["a"])
            self.assertEqual(loaded["summary"]["test"]["class_counts"], {"nv": 1})
            data = json.loads(path.read_text(encoding="utf-8"))
            data["splits"]["c"] = "train"  # silent edit must be detected
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum|sha256"):
                load_manifest(path)

    def test_checksum_independent_of_insertion_order(self):
        self.assertEqual(assignments_checksum({"a": "train", "b": "test"}),
                         assignments_checksum({"b": "test", "a": "train"}))


@unittest.skipUnless(HAS_SKLEARN, 'scikit-learn not installed (pip install -e ".[data]")')
class GroupSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dl_project.data.splits import make_group_splits
        cls.make_group_splits = staticmethod(make_group_splits)
        cls.records = synthetic_records()
        cls.assignments = make_group_splits(cls.records, seed=42)

    def test_no_lesion_in_more_than_one_split_and_all_assigned(self):
        verify_no_group_leakage(self.records, self.assignments)
        self.assertEqual(set(self.assignments), {r["image_id"] for r in self.records})

    def test_ratios_close_to_70_15_15_and_every_class_in_every_split(self):
        summary = split_summary(self.records, self.assignments)
        for name, target in (("train", 0.70), ("validation", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(summary[name]["image_fraction"], target, delta=0.03, msg=name)
            self.assertEqual(set(summary[name]["class_counts"]), set(LESIONS_PER_CLASS), msg=name)
        # Class proportions in val/test follow train (stratification), including rare classes.
        for dx in LESIONS_PER_CLASS:
            share = {n: summary[n]["class_counts"][dx] / summary[n]["images"] for n in SPLIT_NAMES}
            overall = sum(1 for r in self.records if r["dx"] == dx) / len(self.records)
            for name in SPLIT_NAMES:
                self.assertAlmostEqual(share[name], overall, delta=max(0.02, overall * 0.5), msg=(dx, name))

    def test_deterministic_and_independent_of_row_order(self):
        shuffled = list(self.records)
        random.Random(1).shuffle(shuffled)
        self.assertEqual(self.make_group_splits(shuffled, seed=42), self.assignments)

    def test_different_seed_gives_different_split(self):
        self.assertNotEqual(self.make_group_splits(self.records, seed=7), self.assignments)

    def test_make_splits_script_end_to_end(self):
        script = load_make_splits_script()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw").mkdir()
            write_csv(root / "raw" / "meta.csv", self.records)
            args = ["--project-root", temp, "--metadata", "raw/meta.csv", "--output", "splits/manifest.json"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(args), 0)
                manifest = load_manifest(root / "splits" / "manifest.json")
                self.assertEqual(manifest["splits"], self.assignments)
                self.assertEqual(manifest["source"]["metadata_file"], "raw/meta.csv")
                self.assertEqual(manifest["source"]["sha256"], file_sha256(root / "raw" / "meta.csv"))
                self.assertEqual(script.main(args + ["--verify"]), 0)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(script.main(args), 2)  # refuses to overwrite without --force
                    write_csv(root / "raw" / "meta.csv", self.records[:-1])  # metadata changed
                    self.assertEqual(script.main(args + ["--verify"]), 2)


@unittest.skipUnless((ROOT / "data/splits/manifest.json").is_file()
                     and (ROOT / "data/raw/ham10000/HAM10000_metadata.csv").is_file(),
                     "real HAM10000 metadata/manifest not present locally")
class RealManifestTests(unittest.TestCase):
    def test_committed_manifest_has_no_lesion_leakage(self):
        manifest = load_manifest(ROOT / "data/splits/manifest.json")
        records = read_metadata(ROOT / "data/raw/ham10000/HAM10000_metadata.csv")
        verify_no_group_leakage(records, manifest["splits"], manifest["method"]["group_key"])
        self.assertEqual(manifest["source"]["sha256"],
                         file_sha256(ROOT / "data/raw/ham10000/HAM10000_metadata.csv"))


if __name__ == "__main__":
    unittest.main()

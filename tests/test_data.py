"""Tests for HAM10000 metadata encoding, train-only fitting and the Dataset (PLAN_BTL.md 5.3-5.5)."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dl_project.data.ham10000 import (
    CLASSES, LOCALIZATIONS, MetadataEncoder, RARE_CLASSES, class_counts, class_prior, parse_age,
    subsample_records,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("PIL") is not None
REAL_ROOT = ROOT / "data/raw/ham10000"
REAL_MANIFEST = ROOT / "data/splits/manifest.json"


def record(image_id, dx="nv", age="50.0", sex="male", loc="back", lesion=None):
    return {"image_id": image_id, "lesion_id": lesion or f"L_{image_id}", "dx": dx,
            "age": age, "sex": sex, "localization": loc}


class MetadataTests(unittest.TestCase):
    def test_parse_age_keeps_zero_and_flags_missing(self):
        self.assertEqual(parse_age("0.0"), 0.0)
        self.assertIsNone(parse_age(""))
        self.assertIsNone(parse_age("nan"))

    def test_encoder_statistics_come_from_training_records_only(self):
        train = [record("a", age="20"), record("b", age="40")]
        test = [record("c", age="90")]
        encoder = MetadataEncoder().fit(train)
        self.assertAlmostEqual(encoder.age_mean, 30.0)
        self.assertAlmostEqual(encoder.age_std, 10.0)
        numeric, _ = encoder.encode(test[0])
        self.assertAlmostEqual(numeric[0], 6.0)  # (90 - 30) / 10, test age not used in fitting

    def test_missing_and_unknown_values_are_encoded_safely(self):
        encoder = MetadataEncoder().fit([record("a", age="20"), record("b", age="40")])
        numeric, loc = encoder.encode(record("x", age="", sex="", loc="elbow"))
        self.assertEqual(len(numeric), MetadataEncoder.NUM_NUMERIC)
        self.assertEqual(numeric[:2], [0.0, 1.0])
        self.assertEqual(numeric[2:], [0.0, 0.0, 1.0])  # sex -> unknown
        self.assertEqual(loc, 0)  # unseen localization -> unknown
        _, loc = encoder.encode(record("y", loc="lower extremity"))
        self.assertEqual(LOCALIZATIONS[loc], "lower extremity")

    def test_encoder_round_trip(self):
        encoder = MetadataEncoder().fit([record("a", age="20"), record("b", age="44")])
        with TemporaryDirectory() as temp:
            path = Path(temp) / "encoder.json"
            encoder.save(path)
            loaded = MetadataEncoder.load(path)
        self.assertEqual(loaded.encode(record("z")), encoder.encode(record("z")))

    def test_unfitted_encoder_refuses_to_encode(self):
        with self.assertRaises(RuntimeError):
            MetadataEncoder().encode(record("a"))

    def test_class_counts_and_prior(self):
        train = [record("a", "nv"), record("b", "nv"), record("c", "mel"), record("d", "df")]
        counts = class_counts(train)
        self.assertEqual(sum(counts), 4)
        self.assertEqual(counts[CLASSES.index("nv")], 2)
        self.assertAlmostEqual(sum(class_prior(train)), 1.0)
        self.assertTrue(set(RARE_CLASSES) < set(CLASSES))
        with self.assertRaises(ValueError):
            class_counts([record("e", "bogus")])

    def test_subsample_keeps_lesions_whole_and_every_class(self):
        records = [record(f"{dx}{i}{k}", dx, lesion=f"{dx}{i}") for dx in ("nv", "df") for i in range(20)
                   for k in range(2)]
        half = subsample_records(records, 0.5, seed=1)
        lesions = {r["lesion_id"] for r in half}
        self.assertEqual(len(half), 2 * len(lesions))  # both images of each kept lesion
        self.assertEqual({r["dx"] for r in half}, {"nv", "df"})
        self.assertEqual(half, subsample_records(records, 0.5, seed=1))


@unittest.skipUnless(HAS_TORCH, "needs torch + pillow")
class DatasetTests(unittest.TestCase):
    def test_item_shapes_with_synthetic_images(self):
        from PIL import Image
        from dl_project.data.ham10000 import HAM10000Dataset, build_transforms, index_images

        with TemporaryDirectory() as temp:
            folder = Path(temp) / "HAM10000_images_part_1"
            folder.mkdir()
            records = [record(f"ISIC_{i}", dx) for i, dx in enumerate(CLASSES)]
            for item in records:
                Image.new("RGB", (300, 225), (120, 80, 60)).save(folder / f"{item['image_id']}.jpg")
            encoder = MetadataEncoder().fit(records)
            for train in (True, False):
                dataset = HAM10000Dataset(records, index_images(temp), encoder,
                                          build_transforms(train, 64, 72, domain_aug=train))
                item = dataset[3]
                self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
                self.assertEqual(tuple(item["meta_num"].shape), (MetadataEncoder.NUM_NUMERIC,))
                self.assertEqual(int(item["label"]), 3)
                self.assertEqual(item["image_id"], "ISIC_3")

    def test_missing_image_file_is_reported(self):
        from dl_project.data.ham10000 import HAM10000Dataset

        encoder = MetadataEncoder().fit([record("a")])
        with self.assertRaises(FileNotFoundError):
            HAM10000Dataset([record("a")], {}, encoder)


@unittest.skipUnless(HAS_TORCH and REAL_MANIFEST.is_file() and REAL_ROOT.is_dir(), "real HAM10000 not present")
class RealDataTests(unittest.TestCase):
    def test_real_datasets_follow_manifest_and_load(self):
        from dl_project.data.ham10000 import build_datasets

        datasets, encoder, train_records = build_datasets(REAL_ROOT, REAL_MANIFEST, image_size=64)
        sizes = {name: len(ds) for name, ds in datasets.items()}
        self.assertEqual(sum(sizes.values()), 10015)
        train_lesions = {r["lesion_id"] for r in train_records}
        test_lesions = {r["lesion_id"] for r in datasets["test"].records}
        self.assertFalse(train_lesions & test_lesions)
        self.assertTrue(20 < encoder.age_mean < 80)
        item = datasets["validation"][0]
        self.assertEqual(tuple(item["image"].shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()

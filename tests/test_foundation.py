import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dl_project.cli import main
from dl_project.config import load_config, pending_decisions, validate_config
from dl_project.tracking import create_run

ROOT = Path(__file__).resolve().parents[1]


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/baseline.json")

    def test_chosen_topic_configs_have_no_pending_decisions(self):
        for name in ("baseline", "proposed"):
            config = load_config(ROOT / f"configs/{name}.json")
            self.assertEqual(pending_decisions(config), [])
            self.assertEqual(config["dataset"]["group_key"], "lesion_id")

    def test_placeholder_values_are_reported_as_pending(self):
        self.config["task"] = "TBD"
        self.config["model"]["name"] = "TBD"
        pending = pending_decisions(self.config)
        self.assertIn("task", pending)
        self.assertIn("model.name", pending)

    def test_test_split_cannot_select_checkpoints(self):
        self.config["training"]["selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_config(self.config)

    def test_invalid_numeric_values_and_run_names_rejected(self):
        for field, value in (("epochs", 0), ("batch_size", True), ("learning_rate", -1)):
            config = deepcopy(self.config)
            config["training"][field] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        self.config["experiment_name"] = "../escape"
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_micro_batch_must_divide_effective_batch(self):
        self.assertEqual(self.config["training"]["batch_size"] % self.config["training"]["micro_batch_size"], 0)
        for value in (0, 5, True, 64):
            config = deepcopy(self.config)
            config["training"]["micro_batch_size"] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_unconfigured_training_does_not_create_results(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["train", "--project-root", temp, "--config", "config.json"])
            self.assertEqual(code, 2)
            self.assertFalse((root / "outputs").exists())

    def test_run_snapshot_preserves_config_without_fabricating_metrics(self):
        with TemporaryDirectory() as temp:
            run = create_run(self.config, Path(temp), "train")
            self.assertEqual(load_config(run / "config.json"), self.config)
            self.assertTrue((run / "metadata.json").is_file())
            self.assertTrue((run / "environment.txt").is_file())
            self.assertFalse((run / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

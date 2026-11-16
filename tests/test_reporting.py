"""Regression checks for evaluate-run selection and the generated validation table."""
import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dl_project.reporting import select_evaluation_runs
from scripts import aggregate_results, analyze_results


class ReportingTests(unittest.TestCase):
    def test_duplicate_seed_counts_once_and_created_utc_wins(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            exp_dir = root / "outputs/example-s42"
            checkpoint = exp_dir / "train/run/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            wrong_checkpoint = root / "best.pt"
            wrong_checkpoint.touch()
            candidates = (
                ("zzz_older", "2026-11-11T10:00:00Z", 0.3, checkpoint),
                ("aaa_newer", "2026-11-11T11:00:00+00:00", 0.7, checkpoint),
                ("zzz_invalid", "2026-11-11T12:00:00+00:00", 0.9, wrong_checkpoint),
            )
            for name, created, score, model in candidates:
                run = exp_dir / "evaluate" / name
                run.mkdir(parents=True)
                (run / "config.json").write_text("{}", encoding="utf-8")
                (run / "metadata.json").write_text(json.dumps({"created_utc": created, "git_commit": "old"}),
                                                   encoding="utf-8")
                metrics = {"split": "validation", "trained_experiment": "example-s42", "trained_seed": 42,
                           "checkpoint": model.relative_to(root).as_posix(), "macro_f1": score,
                           "per_class": {c: {"recall": score} for c in aggregate_results.CLASSES}}
                (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                header = ",".join(["image_id", "y_true"] + [f"p_{c}" for c in analyze_results.CLASSES])
                (run / "predictions.csv").write_text(f"{header}\n{name},akiec,1,0,0,0,0,0,0\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as warnings:
                selected, = select_evaluation_runs(root, "validation")
            self.assertTrue(selected["run_dir"].endswith("/aaa_newer"))
            for name in ("zzz_older", "zzz_invalid"):
                self.assertIn(name, warnings.getvalue())

            (root / "experiments").mkdir()
            (root / "experiments/COMMIT_MAP.csv").write_text("recorded_commit,current_commit,note\nold,new,\n",
                                                            encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(aggregate_results, "ROOT", root), patch("sys.argv", ["agg", "--split", "validation"]):
                    aggregate_results.main()
                with patch.object(analyze_results, "ROOT", root):
                    (_, image_ids, _, _), = analyze_results.load_runs("validation")["example"]
            self.assertEqual(image_ids, ["aaa_newer"])
            with (root / "reports/tables/results_validation.csv").open(newline="", encoding="utf-8") as handle:
                row, = csv.DictReader(handle)
            self.assertEqual(row["n_seeds"], "1")
            self.assertEqual(row["macro_f1"], "0.700 ± 0.000")
            with (root / "experiments/results.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual({(r["git_commit"], r["git_commit_current"]) for r in csv.DictReader(handle)},
                                 {("old", "new")})

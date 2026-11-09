"""Tests for metrics (NumPy only), losses and the training loop on a tiny synthetic problem."""
import importlib.util
import unittest

import numpy as np

from dl_project.metrics.classification import (bootstrap_ci, classification_report, confusion_matrix,
                                               expected_calibration_error)

HAS_TORCH = importlib.util.find_spec("torch") is not None
NAMES = ["a", "b", "c"]


def one_hot_probs(pred, k=3, confidence=0.9):
    probs = np.full((len(pred), k), (1 - confidence) / (k - 1))
    probs[np.arange(len(pred)), pred] = confidence
    return probs


class MetricTests(unittest.TestCase):
    def test_hand_computed_macro_f1_and_recalls(self):
        y_true = [0, 0, 0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 0, 1, 1, 0, 2, 2]
        report = classification_report(y_true, one_hot_probs(y_pred), NAMES, rare_classes=["b", "c"],
                                       positive_class="c")
        # class a: P=3/4 R=3/4 F1=.75 ; b: P=1/2 R=1/2 F1=.5 ; c: P=1 R=1 F1=1
        self.assertAlmostEqual(report["macro_f1"], (0.75 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(report["balanced_accuracy"], (0.75 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(report["rare_recall"], 0.75)
        self.assertAlmostEqual(report["accuracy"], 6 / 8)
        self.assertEqual(report["per_class"]["b"]["support"], 2)
        self.assertAlmostEqual(report["c_auroc"], 1.0)

    def test_accuracy_hides_minority_failure_but_macro_f1_does_not(self):
        y_true = [0] * 90 + [1] * 10
        report = classification_report(y_true, one_hot_probs([0] * 100, k=2), ["maj", "min"], ["min"], "min")
        self.assertAlmostEqual(report["accuracy"], 0.9)
        self.assertLess(report["macro_f1"], 0.5)
        self.assertEqual(report["rare_recall"], 0.0)

    def test_confusion_matrix_and_ece(self):
        self.assertEqual(confusion_matrix([0, 1, 1], [0, 1, 0], 2).tolist(), [[1, 0], [1, 1]])
        perfect = one_hot_probs([0, 1, 2], confidence=1.0)
        self.assertAlmostEqual(expected_calibration_error(perfect, [0, 1, 2]), 0.0)
        overconfident = one_hot_probs([0, 0, 0, 0], confidence=1.0)
        self.assertAlmostEqual(expected_calibration_error(overconfident, [0, 1, 1, 1]), 0.75)

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, 200)
        y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 3, 200))
        probs = one_hot_probs(y_pred)
        point = classification_report(y_true, probs, NAMES)["macro_f1"]
        low, high = bootstrap_ci(y_true, probs, "macro_f1", NAMES, n_boot=200)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)


@unittest.skipUnless(HAS_TORCH, "needs torch")
class LossTests(unittest.TestCase):
    def test_losses(self):
        import torch
        from dl_project.engine.losses import build_loss

        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        target = torch.tensor([0, 1])
        counts = [80, 15, 5]
        ce = build_loss("ce", counts)(logits, target)
        self.assertAlmostEqual(ce.item(), torch.nn.functional.cross_entropy(logits, target).item(), places=6)
        focal = build_loss("focal", counts)(logits, target)
        self.assertLess(focal.item(), ce.item())  # down-weights easy examples
        weighted = build_loss("weighted_ce", counts)
        self.assertAlmostEqual(weighted.weight.mean().item(), 1.0, places=5)
        self.assertGreater(weighted.weight[2].item(), weighted.weight[0].item())
        la = build_loss("logit_adjusted", counts, tau=1.0)
        prior = torch.tensor(counts, dtype=torch.float) / sum(counts)
        expected = torch.nn.functional.cross_entropy(logits + prior.log(), target)
        self.assertAlmostEqual(la(logits, target).item(), expected.item(), places=5)
        for bad in (("logit_adjusted", None), ("bogus", 1.0)):
            with self.assertRaises(ValueError):
                build_loss(bad[0], counts, tau=bad[1])

    def test_logit_adjustment_penalises_majority_predictions_more(self):
        import torch
        from dl_project.engine.losses import build_loss

        # Equal logits: plain CE is the same for every target; LA-CE makes the rare target costlier,
        # pushing the model to raise rare-class logits during training.
        logits = torch.zeros(2, 3)
        la = build_loss("logit_adjusted", [90, 9, 1], tau=1.0)
        self.assertGreater(la(logits[:1], torch.tensor([2])).item(), la(logits[:1], torch.tensor([0])).item())


@unittest.skipUnless(HAS_TORCH, "needs torch")
class TrainerTests(unittest.TestCase):
    def test_trainer_learns_tiny_problem_with_accumulation(self):
        import tempfile
        from pathlib import Path
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from dl_project.engine.trainer import Trainer

        class Toy(Dataset):
            def __init__(self, n, seed):
                g = torch.Generator().manual_seed(seed)
                self.y = torch.randint(0, 3, (n,), generator=g)
                self.x = torch.randn(n, 3, 8, 8, generator=g) * 0.3 + self.y[:, None, None, None].float()

            def __len__(self):
                return len(self.y)

            def __getitem__(self, i):
                return {"image": self.x[i], "meta_num": torch.zeros(5), "meta_loc": torch.tensor(0),
                        "label": self.y[i], "image_id": f"id{i}"}

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)

            def forward(self, image, meta_num, meta_loc):
                return self.fc(image.mean(dim=(2, 3)))

        torch.manual_seed(0)
        model = Net()
        with tempfile.TemporaryDirectory() as temp:
            trainer = Trainer(model, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.05),
                              torch.device("cpu"), temp, NAMES, ["c"], batch_size=16, micro_batch_size=4,
                              epochs=6, warmup_epochs=1, patience=10, log=lambda m: None)
            summary = trainer.fit(DataLoader(Toy(96, 0), batch_size=4, shuffle=True), DataLoader(Toy(48, 1), 16))
            self.assertEqual(summary["accumulation_steps"], 4)
            self.assertEqual(summary["steps_per_epoch"], 6)
            self.assertGreater(summary["best_val_macro_f1"], 0.8)
            for name in ("best.pt", "last.pt", "history.csv", "val_metrics_best.json"):
                self.assertTrue((Path(temp) / name).is_file(), name)

    def test_micro_batch_must_divide_batch(self):
        import torch
        from torch import nn
        from dl_project.engine.trainer import Trainer

        with self.assertRaises(ValueError):
            Trainer(nn.Linear(1, 1), None, None, torch.device("cpu"), ".", NAMES, [], batch_size=32,
                    micro_batch_size=5, epochs=1)


if __name__ == "__main__":
    unittest.main()

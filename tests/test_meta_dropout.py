"""Metadata dropout: whole metadata vector dropped per sample during training only."""
import unittest

import torch

from dl_project.models.classifier import build_model


def _config(meta_dropout, fusion="concat"):
    return {"model": {"name": "convnext_tiny", "pretrained": None},
            "components": {"meta_fusion": fusion, "film_stages": [], "meta_dropout": meta_dropout}}


class MetaDropoutTest(unittest.TestCase):
    def test_default_off_and_eval_deterministic(self):
        self.assertEqual(build_model(_config(0.0)).meta_dropout, 0.0)
        model = build_model(_config(0.3)).eval()
        x, num, loc = torch.randn(4, 3, 32, 32), torch.randn(4, 5), torch.tensor([1, 2, 3, 4])
        self.assertTrue(torch.allclose(model(x, num, loc), model(x, num, loc)))

    def test_train_drops_whole_vectors(self):
        torch.manual_seed(0)
        model = build_model(_config(0.5)).train()
        seen = []
        model.head.register_forward_hook(lambda mod, inp, out: seen.append(inp[0][:, -64:]))
        model(torch.randn(64, 3, 32, 32), torch.randn(64, 5), torch.randint(0, 15, (64,)))
        dropped = (seen[-1].abs().sum(1) == 0).float().mean().item()
        self.assertTrue(0.25 < dropped < 0.75)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            build_model(_config(1.0))
        with self.assertRaises(ValueError):
            build_model(_config(0.3, fusion="none"))


if __name__ == "__main__":
    unittest.main()

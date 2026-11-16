"""FiLM-only fusion: metadata reaches the logits only through FiLM (no metadata in the head)."""
import unittest

import torch

from dl_project.models.classifier import SkinLesionClassifier


class FilmOnlyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SkinLesionClassifier("convnext_tiny", pretrained=False, meta_fusion="film_only",
                                          film_stages=(2, 3)).eval()
        self.x = torch.randn(2, 3, 64, 64)

    def test_head_ignores_metadata_vector(self):
        self.assertEqual(self.model.head[1].in_features, self.model.backbone.num_features)
        self.assertEqual(self.model.film_stages, (2, 3))

    def test_identity_at_init_then_metadata_dependent(self):
        a = self.model(self.x, torch.randn(2, 5), torch.tensor([1, 3]))
        b = self.model(self.x, torch.randn(2, 5), torch.tensor([4, 7]))
        self.assertTrue(torch.allclose(a, b))  # zero-initialised FiLM = identity
        for film in self.model.film.values():
            torch.nn.init.normal_(film.proj.weight, std=0.1)
        a = self.model(self.x, torch.randn(2, 5), torch.tensor([1, 3]))
        b = self.model(self.x, torch.randn(2, 5), torch.tensor([4, 7]))
        self.assertFalse(torch.allclose(a, b))

    def test_requires_stages(self):
        with self.assertRaises(ValueError):
            SkinLesionClassifier("convnext_tiny", meta_fusion="film_only", film_stages=())


if __name__ == "__main__":
    unittest.main()

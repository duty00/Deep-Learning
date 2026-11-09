"""Model tests: output shapes, FiLM identity at initialisation, parameter groups (PLAN_BTL.md 5.6)."""
import importlib.util
import unittest

HAS_TIMM = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timm") is not None


@unittest.skipUnless(HAS_TIMM, "needs torch + timm")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.manual_seed(0)
        cls.torch = torch
        cls.image = torch.randn(2, 3, 64, 64)
        cls.meta_num = torch.randn(2, 5)
        cls.meta_loc = torch.tensor([0, 3])

    def make(self, fusion, stages=(), backbone="convnext_tiny"):
        from dl_project.models.classifier import SkinLesionClassifier

        self.torch.manual_seed(0)
        return SkinLesionClassifier(backbone, pretrained=False, meta_fusion=fusion, film_stages=stages).eval()

    def test_output_shapes_for_all_fusions(self):
        for fusion, stages in (("none", ()), ("concat", ()), ("film", (2, 3))):
            with self.subTest(fusion=fusion):
                model = self.make(fusion, stages)
                logits = model(self.image, self.meta_num, self.meta_loc)
                self.assertEqual(tuple(logits.shape), (2, 7))

    def test_efficientnet_backbone_stages(self):
        model = self.make("film", (2, 3), backbone="efficientnet_b0")
        self.assertEqual(len(model.backbone.stage_channels), 4)
        self.assertEqual(tuple(model(self.image, self.meta_num, self.meta_loc).shape), (2, 7))

    def test_film_layer_is_identity_at_init(self):
        from dl_project.models.film import FiLM

        film = FiLM(cond_dim=8, channels=4)
        features = self.torch.randn(3, 4, 5, 5)
        self.assertTrue(self.torch.equal(film(features, self.torch.randn(3, 8)), features))

    def test_film_model_equals_concat_model_at_init(self):
        concat = self.make("concat")
        film = self.make("film", (2, 3))
        missing, unexpected = film.load_state_dict(concat.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("film.") for key in missing))
        self.assertEqual(unexpected, [])
        with self.torch.no_grad():
            expected = concat(self.image, self.meta_num, self.meta_loc)
            actual = film(self.image, self.meta_num, self.meta_loc)
        self.assertTrue(self.torch.allclose(actual, expected, atol=1e-6))

    def test_film_changes_output_once_trained(self):
        # Note: a channel-UNIFORM gamma/beta would be cancelled by the LayerNorm that follows each ConvNeXt
        # stage, so only channel-specific modulation (what training learns) can change the output.
        film = self.make("film", (3,))
        with self.torch.no_grad():
            before = film(self.image, self.meta_num, self.meta_loc)
            proj = film.film["3"].proj
            proj.weight.normal_(0, 0.1)
            proj.bias.normal_(0, 0.5)
            after = film(self.image, self.meta_num, self.meta_loc)
            other = film(self.image, self.meta_num.flip(0), self.meta_loc.flip(0))
        self.assertFalse(self.torch.allclose(before, after))
        self.assertFalse(self.torch.allclose(after, other))  # output depends on each sample's metadata

    def test_invalid_configurations_rejected(self):
        from dl_project.models.classifier import SkinLesionClassifier

        for kwargs in ({"meta_fusion": "film"}, {"meta_fusion": "concat", "film_stages": [2]},
                       {"meta_fusion": "film", "film_stages": [5]}, {"meta_fusion": "attention"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SkinLesionClassifier("convnext_tiny", **kwargs)
        with self.assertRaises(ValueError):
            self.make("concat")(self.image)  # metadata required

    def test_parameter_groups_cover_every_parameter_once(self):
        model = self.make("film", (2, 3))
        groups = model.parameter_groups(backbone_lr=1e-4, head_lr=1e-3, weight_decay=0.05)
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        lrs = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(lrs["backbone/decay"], 1e-4)
        self.assertEqual(lrs["head/decay"], 1e-3)
        self.assertTrue(all(g["weight_decay"] == 0.0 for g in groups if g["name"].endswith("no_decay")))

    def test_build_model_from_project_config(self):
        import json
        from pathlib import Path
        from dl_project.models.classifier import build_model

        config = json.loads((Path(__file__).resolve().parents[1] / "configs/proposed.json").read_text("utf-8"))
        config["model"]["pretrained"] = None  # no download in tests
        model = build_model(config)
        self.assertEqual(model.meta_fusion, config["components"]["meta_fusion"])
        self.assertEqual(model.film_stages, tuple(config["components"]["film_stages"]))


if __name__ == "__main__":
    unittest.main()

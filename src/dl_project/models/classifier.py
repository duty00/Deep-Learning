"""Skin lesion classifiers B0 / B1 / FiLM (PLAN_BTL.md section 5.6).

* ``meta_fusion="none"``   (B0): backbone -> pooled features -> Linear(7).
* ``meta_fusion="concat"`` (B1, main baseline): [pooled ; MetaEncoder(meta)] -> Linear(7).
* ``meta_fusion="film"``   (proposed): same head as B1, plus FiLM on the outputs of ``film_stages``.

FiLM is therefore an addition on top of B1, so E2 vs E6 isolates the effect of mid-network modulation.
"""
import torch
from torch import nn

from .backbones import StagedBackbone
from .film import FiLM, MetaEncoder

FUSIONS = ("none", "concat", "film")


class SkinLesionClassifier(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=False, num_classes=7, meta_fusion="concat",
                 film_stages=(), num_numeric=5, num_locations=15, meta_dim=64, dropout=0.2,
                 drop_path_rate=0.1):
        super().__init__()
        if meta_fusion not in FUSIONS:
            raise ValueError(f"meta_fusion must be one of {FUSIONS}.")
        film_stages = tuple(int(stage) for stage in (film_stages or ()))
        if meta_fusion == "film" and not film_stages:
            raise ValueError("meta_fusion='film' requires at least one entry in film_stages.")
        if meta_fusion != "film" and film_stages:
            raise ValueError("film_stages must be empty unless meta_fusion='film'.")
        if any(stage not in (1, 2, 3, 4) for stage in film_stages):
            raise ValueError("film_stages entries must be in 1..4.")

        self.meta_fusion = meta_fusion
        self.backbone = StagedBackbone(backbone, pretrained=pretrained, drop_path_rate=drop_path_rate)
        self.meta_encoder = None
        self.film = nn.ModuleDict()
        head_in = self.backbone.num_features
        if meta_fusion != "none":
            self.meta_encoder = MetaEncoder(num_numeric, num_locations, out_dim=meta_dim)
            head_in += meta_dim
        for stage in film_stages:
            self.film[str(stage)] = FiLM(meta_dim, self.backbone.stage_channels[stage - 1])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, num_classes))

    @property
    def film_stages(self):
        return tuple(sorted(int(key) for key in self.film.keys()))

    def forward(self, image, meta_num=None, meta_loc=None):
        if self.meta_encoder is None:
            return self.head(self.backbone(image))
        if meta_num is None or meta_loc is None:
            raise ValueError(f"meta_fusion={self.meta_fusion!r} needs meta_num and meta_loc.")
        h = self.meta_encoder(meta_num, meta_loc)
        hook = None
        if len(self.film):
            def hook(stage, features):
                key = str(stage)
                return self.film[key](features, h) if key in self.film else features
        features = self.backbone(image, stage_hook=hook)
        return self.head(torch.cat([features, h], dim=1))

    def parameter_groups(self, backbone_lr, head_lr, weight_decay):
        """Pretrained backbone gets ``backbone_lr``; new modules (meta encoder, FiLM, head) get ``head_lr``.

        Biases and normalisation parameters are excluded from weight decay.
        """
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("backbone.")
            no_decay = param.ndim <= 1 or name.endswith(".bias")
            key = ("backbone" if is_backbone else "head", "no_decay" if no_decay else "decay")
            groups.setdefault(key, []).append(param)
        result = []
        for (part, decay), params in sorted(groups.items()):
            result.append({"params": params, "lr": backbone_lr if part == "backbone" else head_lr,
                           "weight_decay": 0.0 if decay == "no_decay" else weight_decay,
                           "name": f"{part}/{decay}"})
        return result


def build_model(config, num_numeric=5, num_locations=15):
    """Build the classifier from a project config (``model`` + ``components`` sections)."""
    model_cfg = config["model"]
    components = config.get("components", {})
    return SkinLesionClassifier(
        backbone=model_cfg["name"],
        pretrained=model_cfg.get("pretrained") in ("imagenet", True),
        meta_fusion=components.get("meta_fusion", "concat"),
        film_stages=components.get("film_stages", []),
        num_numeric=num_numeric,
        num_locations=num_locations,
        meta_dim=components.get("meta_dim", 64),
        dropout=components.get("head_dropout", 0.2),
        drop_path_rate=components.get("drop_path_rate", 0.1),
    )

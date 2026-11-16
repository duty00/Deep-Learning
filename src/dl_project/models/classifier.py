"""Skin lesion classifiers B0 / B1 / FiLM.

* ``meta_fusion="none"``   (B0): backbone -> pooled features -> Linear(7).
* ``meta_fusion="concat"`` (B1, main baseline): [pooled ; MetaEncoder(meta)] -> Linear(7).
* ``meta_fusion="film"``   (proposed): same head as B1, plus FiLM on the outputs of ``film_stages``.
* ``meta_fusion="film_only"``: FiLM on ``film_stages`` but the head sees image features only, so metadata
  cannot act as a direct prior on the logits (tests the late-fusion shortcut hypothesis).

FiLM is therefore an addition on top of B1, so E2 vs E6 isolates the effect of mid-network modulation.
"""
import torch
from torch import nn

from .backbones import StagedBackbone
from .film import FiLM, MetaEncoder

FUSIONS = ("none", "concat", "film", "film_only")


class SkinLesionClassifier(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=False, num_classes=7, meta_fusion="concat",
                 film_stages=(), num_numeric=5, num_locations=15, meta_dim=64, dropout=0.2,
                 drop_path_rate=0.1, pretrained_tag=None):
        super().__init__()
        if meta_fusion not in FUSIONS:
            raise ValueError(f"meta_fusion must be one of {FUSIONS}.")
        film_stages = tuple(int(stage) for stage in (film_stages or ()))
        if meta_fusion in ("film", "film_only") and not film_stages:
            raise ValueError("meta_fusion='film' requires at least one entry in film_stages.")
        if meta_fusion not in ("film", "film_only") and film_stages:
            raise ValueError("film_stages must be empty unless meta_fusion='film'.")
        if any(stage not in (1, 2, 3, 4) for stage in film_stages):
            raise ValueError("film_stages entries must be in 1..4.")

        self.meta_fusion = meta_fusion
        self.meta_dropout = 0.0  # set by build_model from components.meta_dropout
        self.backbone = StagedBackbone(backbone, pretrained=pretrained, drop_path_rate=drop_path_rate,
                                       pretrained_tag=pretrained_tag)
        self.meta_encoder = None
        self.film = nn.ModuleDict()
        head_in = self.backbone.num_features
        if meta_fusion != "none":
            self.meta_encoder = MetaEncoder(num_numeric, num_locations, out_dim=meta_dim)
            if meta_fusion != "film_only":  # film_only: metadata reaches the logits only through FiLM
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
        if self.training and self.meta_dropout > 0:  # drop the whole metadata vector per sample
            keep = (torch.rand(h.shape[0], 1, device=h.device) >= self.meta_dropout).to(h.dtype)
            h = h * keep
        hook = None
        if len(self.film):
            def hook(stage, features):
                key = str(stage)
                return self.film[key](features, h) if key in self.film else features
        features = self.backbone(image, stage_hook=hook)
        return self.head(features if self.meta_fusion == "film_only" else torch.cat([features, h], dim=1))

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


def _build_model_base(config, num_numeric=5, num_locations=15):
    """Build the classifier from a project config (``model`` + ``components`` sections)."""
    model_cfg = config["model"]
    components = config.get("components", {})
    return SkinLesionClassifier(
        backbone=model_cfg["name"],
        pretrained=model_cfg.get("pretrained") in ("imagenet", True),
        pretrained_tag=model_cfg.get("pretrained_tag"),
        meta_fusion=components.get("meta_fusion", "concat"),
        film_stages=components.get("film_stages", []),
        num_numeric=num_numeric,
        num_locations=num_locations,
        meta_dim=components.get("meta_dim", 64),
        dropout=components.get("head_dropout", 0.2),
        drop_path_rate=components.get("drop_path_rate", 0.1),
    )


def build_model(config, num_numeric=5, num_locations=15):
    """Build the classifier; ``components.meta_dropout`` (default 0) drops metadata during training."""
    model = _build_model_base(config, num_numeric, num_locations)
    p = float(config.get("components", {}).get("meta_dropout", 0.0) or 0.0)
    if not 0.0 <= p < 1.0:
        raise ValueError("meta_dropout must be in [0, 1).")
    if p > 0 and model.meta_encoder is None:
        raise ValueError("meta_dropout needs a metadata fusion.")
    model.meta_dropout = p
    return model

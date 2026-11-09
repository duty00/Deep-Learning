"""timm backbones exposed as 4 sequential stages so FiLM can be inserted between them.

Stages are numbered 1..4 (output strides 4, 8, 16, 32) for every supported backbone, so
``film_stages: [2, 3]`` means "modulate the outputs of stage 2 and stage 3" for both ConvNeXt and
EfficientNet. Pretrained weights and model code come from timm (Apache-2.0, credited in docs/ATTRIBUTION.md).
"""
import timm
import torch
from torch import nn

SUPPORTED = ("convnext_tiny", "efficientnet_b0")


class StagedBackbone(nn.Module):
    """stem -> stage1..stage4 -> finalize -> pooled feature vector (B, num_features)."""

    def __init__(self, name, pretrained=False, drop_path_rate=0.1):
        super().__init__()
        if name not in SUPPORTED:
            raise ValueError(f"Unsupported backbone {name!r}; choose from {SUPPORTED}.")
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if name.startswith("convnext"):
            kwargs["drop_path_rate"] = drop_path_rate
        self.name = name
        self.model = timm.create_model(name, **kwargs)
        self.num_features = self.model.num_features
        if name.startswith("convnext"):
            self.stem = self.model.stem
            self.stages = nn.ModuleList(self.model.stages)
        else:  # efficientnet: 7 block groups -> 4 stages with strides 4, 8, 16, 32
            m = self.model
            self.stem = nn.Sequential(m.conv_stem, m.bn1)
            b = m.blocks
            self.stages = nn.ModuleList([nn.Sequential(b[0], b[1]), b[2], nn.Sequential(b[3], b[4]),
                                         nn.Sequential(b[5], b[6])])
        self.stage_channels = self._probe_channels()

    @torch.no_grad()
    def _probe_channels(self):
        was_training = self.training
        self.eval()
        x = self.stem(torch.zeros(1, 3, 64, 64))
        channels = []
        for stage in self.stages:
            x = stage(x)
            channels.append(x.shape[1])
        self.train(was_training)
        return channels

    def finalize(self, x):
        """Everything after the last stage up to the pooled (pre-logits) feature vector."""
        m = self.model
        if self.name.startswith("convnext"):
            x = m.norm_pre(x)
        else:
            x = m.bn2(m.conv_head(x))
        return m.forward_head(x, pre_logits=True)

    def forward(self, x, stage_hook=None):
        """``stage_hook(stage_number, features) -> features`` is applied after each stage (1-based)."""
        x = self.stem(x)
        for number, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if stage_hook is not None:
                x = stage_hook(number, x)
        return self.finalize(x)

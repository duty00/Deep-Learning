"""Losses compared in RQ2 (PLAN_BTL.md section 5.7). All class statistics come from TRAIN counts.

* ``ce``             plain cross-entropy.
* ``weighted_ce``    CE with inverse-frequency class weights, normalised to mean 1.
* ``focal``          focal loss (Lin et al., ICCV 2017), gamma = 2.
* ``logit_adjusted`` logit-adjusted CE (Menon et al., ICLR 2021): during training the loss is computed on
                     ``logits + tau * log(prior)``; at prediction time the raw logits are used, which shifts
                     decisions towards rare classes. The model itself is unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

LOSSES = ("ce", "weighted_ce", "focal", "logit_adjusted")


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("Every class needs at least one training sample for weighted_ce.")
        weights = counts.sum() / (len(counts) * counts)
        self.register_buffer("weight", weights / weights.mean())

    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target, weight=self.weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits.float(), dim=1)
        log_pt = log_prob.gather(1, target[:, None]).squeeze(1)
        return (-(1 - log_pt.exp()) ** self.gamma * log_pt).mean()


class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float64)
        if (counts <= 0).any():
            raise ValueError("Every class needs at least one training sample for logit_adjusted.")
        prior = counts / counts.sum()
        self.tau = float(tau)
        self.register_buffer("adjustment", (self.tau * prior.log()).float())

    def forward(self, logits, target):
        return F.cross_entropy(logits.float() + self.adjustment, target)


class CrossEntropy(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target)


def build_loss(name, class_counts, tau=1.0, focal_gamma=2.0):
    if name == "ce":
        return CrossEntropy()
    if name == "weighted_ce":
        return WeightedCrossEntropy(class_counts)
    if name == "focal":
        return FocalLoss(focal_gamma)
    if name == "logit_adjusted":
        if tau is None or not math.isfinite(float(tau)) or float(tau) <= 0:
            raise ValueError("logit_adjusted needs a positive tau.")
        return LogitAdjustedCrossEntropy(class_counts, tau)
    raise ValueError(f"Unknown loss {name!r}; choose from {LOSSES}.")

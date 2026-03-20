from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0)

        embeddings = F.normalize(embeddings, dim=-1)
        logits = torch.matmul(embeddings, embeddings.transpose(0, 1)) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.transpose(0, 1)).to(logits.dtype)
        identity = torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype)
        positive_mask = positive_mask - identity
        logits_mask = 1.0 - identity

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_count = positive_mask.sum(dim=1)
        valid_rows = positive_count > 0
        if not valid_rows.any():
            return embeddings.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
        return -mean_log_prob_pos[valid_rows].mean()


class QualityAssessmentLoss(nn.Module):
    def __init__(
        self,
        *,
        regression_weight: float,
        classification_weight: float,
        contrastive_weight: float,
        temperature: float,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.regression_loss = nn.SmoothL1Loss()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_score: torch.Tensor,
        target_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        regression = self.regression_loss(outputs["score"], target_score)
        classification = self.classification_loss(outputs["logits"], target_label)
        contrastive = self.contrastive_loss(outputs["embedding"], target_label)
        total = (
            self.regression_weight * regression
            + self.classification_weight * classification
            + self.contrastive_weight * contrastive
        )

        return {
            "loss": total,
            "regression_loss": regression.detach(),
            "classification_loss": classification.detach(),
            "contrastive_loss": contrastive.detach(),
        }

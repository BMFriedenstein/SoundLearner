from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OscillatorLoss(nn.Module):
    def __init__(self, activity_weight: float = 1.0, parameter_weight: float = 10.0) -> None:
      super().__init__()
      self.activity_weight = activity_weight
      self.parameter_weight = parameter_weight

    def forward(self, predictions: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
      activity_target = target[:, :, 0]
      parameter_target = target[:, :, 1:]

      activity_loss = F.binary_cross_entropy_with_logits(predictions["activity_logits"], activity_target)

      active_count = mask.sum().clamp_min(1.0)
      parameter_loss_per_value = F.smooth_l1_loss(predictions["parameters"], parameter_target, reduction="none")
      parameter_loss = (parameter_loss_per_value * mask.unsqueeze(-1)).sum() / (active_count * parameter_target.shape[-1])

      total = self.activity_weight * activity_loss + self.parameter_weight * parameter_loss
      metrics = {
          "loss": float(total.detach().cpu()),
          "activity_loss": float(activity_loss.detach().cpu()),
          "parameter_loss": float(parameter_loss.detach().cpu()),
      }
      return total, metrics


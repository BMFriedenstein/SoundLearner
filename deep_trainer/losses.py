from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .differentiable_audio import (
    DifferentiableOscillatorBank,
    MultiResolutionFeatureExtractor,
    RenderLossConfig,
    denormalize_log_frequency,
    frequency_crowding_loss,
    normalize_log_frequency,
)

class OscillatorLoss(nn.Module):
    def __init__(
        self,
        activity_weight: float = 1.0,
        parameter_weight: float = 10.0,
        activity_positive_weight: float = 0.0,
        f0_weight: float = 1.0,
        crowding_weight: float = 0.0,
        render_weight: float = 0.0,
        render_rms_weight: float = 0.0,
        f0_min_frequency: float = 40.0,
        f0_max_frequency: float = 2000.0,
        render_config: RenderLossConfig | None = None,
    ) -> None:
      super().__init__()
      self.activity_weight = activity_weight
      self.parameter_weight = parameter_weight
      self.activity_positive_weight = activity_positive_weight
      self.f0_weight = f0_weight
      self.crowding_weight = crowding_weight
      self.render_weight = render_weight
      self.render_rms_weight = render_rms_weight
      self.f0_min_frequency = f0_min_frequency
      self.f0_max_frequency = f0_max_frequency
      self.render_synth = None
      self.render_features = None
      self.register_buffer("channel_weights", torch.tensor([2.0, 0.75, 1.25], dtype=torch.float32), persistent=False)
      if render_weight > 0.0:
        if render_config is None:
          raise ValueError("render_config is required when render_weight > 0")
        self.render_synth = DifferentiableOscillatorBank(render_config.sample_rate, render_config.render_seconds)
        self.render_features = MultiResolutionFeatureExtractor(render_config)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
        source_features: torch.Tensor | None = None,
        note_frequency: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        source_rms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
      activity_target = target[:, :, 0]
      parameter_target = target[:, :, 1:]

      positive_count = activity_target.sum().clamp_min(1.0)
      negative_count = (1.0 - activity_target).sum().clamp_min(1.0)
      if self.activity_positive_weight > 0.0:
        positive_weight = torch.tensor(self.activity_positive_weight, device=activity_target.device, dtype=activity_target.dtype)
      else:
        positive_weight = (negative_count / positive_count).clamp(1.0, 32.0)
      activity_loss = F.binary_cross_entropy_with_logits(predictions["activity_logits"], activity_target, pos_weight=positive_weight)

      active_count = mask.sum().clamp_min(1.0)
      parameter_loss_per_value = F.smooth_l1_loss(predictions["parameters"], parameter_target, reduction="none")
      parameter_loss = (parameter_loss_per_value * mask.unsqueeze(-1)).sum() / (active_count * parameter_target.shape[-1])

      render_feature_loss = torch.zeros((), device=predictions["parameters"].device)
      render_rms_loss = torch.zeros((), device=predictions["parameters"].device)
      f0_loss = torch.zeros((), device=predictions["parameters"].device)
      f0_cents_mae = torch.zeros((), device=predictions["parameters"].device)
      crowding_loss = torch.zeros((), device=predictions["parameters"].device)
      render_note_frequency = note_frequency
      if note_frequency is not None and "f0_normalized" in predictions:
        target_f0 = normalize_log_frequency(note_frequency, self.f0_min_frequency, self.f0_max_frequency)
        f0_loss = F.smooth_l1_loss(predictions["f0_normalized"], target_f0)
        render_note_frequency = denormalize_log_frequency(predictions["f0_normalized"], self.f0_min_frequency, self.f0_max_frequency)
        f0_cents_mae = torch.mean(torch.abs(1200.0 * torch.log2(render_note_frequency.clamp_min(1e-6) / note_frequency.clamp_min(1e-6))))

      if self.crowding_weight > 0.0:
        crowding_loss = frequency_crowding_loss(predictions["parameters"], predictions["activity_logits"])

      if self.render_weight > 0.0:
        if source_features is None or note_frequency is None or velocity is None:
          raise ValueError("source_features, note_frequency, and velocity are required when render loss is enabled")
        assert self.render_synth is not None
        assert self.render_features is not None
        if render_note_frequency is None:
          render_note_frequency = note_frequency
        rendered_audio = self.render_synth(predictions["parameters"], predictions["activity_logits"], render_note_frequency, velocity)
        rendered_feature_pyramid = self.render_features(rendered_audio)
        source_feature_pyramid = [source_features]
        for level in range(1, len(rendered_feature_pyramid)):
          target_shape = rendered_feature_pyramid[level].shape[-2:]
          pooled = F.adaptive_avg_pool2d(source_features, target_shape)
          source_feature_pyramid.append(pooled)

        feature_losses = []
        for rendered_level, source_level in zip(rendered_feature_pyramid, source_feature_pyramid):
          frequency_count = rendered_level.shape[2]
          frequency_weights = torch.linspace(1.5, 0.75, frequency_count, device=rendered_level.device, dtype=rendered_level.dtype)
          weight_map = self.channel_weights.view(1, 3, 1, 1) * frequency_weights.view(1, 1, frequency_count, 1)
          diff = F.smooth_l1_loss(rendered_level, source_level, reduction="none")
          feature_losses.append((diff * weight_map).mean())
        render_feature_loss = torch.stack(feature_losses).mean()

        if source_rms is not None and self.render_rms_weight > 0.0:
          rendered_rms = torch.sqrt(torch.mean(torch.square(rendered_audio), dim=1) + 1e-8)
          render_rms_loss = F.smooth_l1_loss(torch.log(rendered_rms.clamp_min(1e-5)), torch.log(source_rms.clamp_min(1e-5)))

      total = (
          self.activity_weight * activity_loss
          + self.parameter_weight * parameter_loss
          + self.f0_weight * f0_loss
          + self.crowding_weight * crowding_loss
          + self.render_weight * render_feature_loss
          + self.render_rms_weight * render_rms_loss
      )
      metrics = {
          "loss": float(total.detach().cpu()),
          "activity_loss": float(activity_loss.detach().cpu()),
          "parameter_loss": float(parameter_loss.detach().cpu()),
          "f0_loss": float(f0_loss.detach().cpu()),
          "f0_cents_mae": float(f0_cents_mae.detach().cpu()),
          "crowding_loss": float(crowding_loss.detach().cpu()),
          "render_feature_loss": float(render_feature_loss.detach().cpu()),
          "render_rms_loss": float(render_rms_loss.detach().cpu()),
      }
      return total, metrics

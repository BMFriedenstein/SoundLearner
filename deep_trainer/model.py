from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .dataset import OSCILLATOR_PARAMETER_COUNT


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3
    max_oscillators: int = 64
    width: int = 64
    dropout: float = 0.1
    coordinate_channels: bool = False
    normalization: str = "batch"


def model_config_from_mapping(data: dict[str, Any] | None) -> ModelConfig:
    """Build a ModelConfig while ignoring stale/future checkpoint keys."""
    if data is None:
      raise ValueError("Checkpoint is missing model_config")
    valid_keys = {field.name for field in fields(ModelConfig)}
    filtered = {key: value for key, value in data.items() if key in valid_keys}
    return ModelConfig(**filtered)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
      super().__init__()
      self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
      self.norm = nn.LayerNorm(channels)
      self.pointwise_1 = nn.Linear(channels, channels * 4)
      self.pointwise_2 = nn.Linear(channels * 4, channels)
      self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)
      self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      residual = x
      x = self.depthwise(x)
      x = x.permute(0, 2, 3, 1)
      x = self.norm(x)
      x = self.pointwise_1(x)
      x = F.gelu(x)
      x = self.pointwise_2(x)
      x = self.gamma * x
      x = x.permute(0, 3, 1, 2)
      return residual + x


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
      if channels % groups == 0:
        return groups
    return 1


def spatial_norm(channels: int, normalization: str) -> nn.Module:
    if normalization == "batch":
      return nn.BatchNorm2d(channels)
    if normalization == "group":
      return nn.GroupNorm(_group_count(channels), channels)
    raise ValueError(f"Unsupported normalization: {normalization}")


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, normalization: str) -> None:
      super().__init__()
      layers: list[nn.Module] = [
          nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
          spatial_norm(out_channels, normalization),
      ]
      layers.extend(ConvNeXtBlock(out_channels) for _ in range(blocks))
      self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layers(x)


class SoundLearnerNet(nn.Module):
    """Small ConvNeXt-style baseline for oscillator parameter prediction."""

    def __init__(self, config: ModelConfig) -> None:
      super().__init__()
      self.config = config
      width = config.width
      stem_input_channels = config.input_channels + (2 if config.coordinate_channels else 0)
      self.stem = nn.Sequential(
          nn.Conv2d(stem_input_channels, width, kernel_size=4, stride=4),
          spatial_norm(width, config.normalization),
          ConvNeXtBlock(width),
      )
      self.encoder = nn.Sequential(
          DownsampleStage(width, width * 2, blocks=2, normalization=config.normalization),
          DownsampleStage(width * 2, width * 4, blocks=2, normalization=config.normalization),
          DownsampleStage(width * 4, width * 8, blocks=3, normalization=config.normalization),
      )
      self.pool = nn.AdaptiveAvgPool2d(1)
      self.head = nn.Sequential(
          nn.Flatten(),
          nn.LayerNorm(width * 8),
          nn.Dropout(config.dropout),
          nn.Linear(width * 8, width * 8),
          nn.GELU(),
          nn.Dropout(config.dropout),
      )
      self.activity_head = nn.Linear(width * 8, config.max_oscillators)
      self.parameter_head = nn.Linear(width * 8, config.max_oscillators * OSCILLATOR_PARAMETER_COUNT)
      self.f0_head = nn.Linear(width * 8, 1)

    def append_coordinate_channels(self, features: torch.Tensor) -> torch.Tensor:
      if not self.config.coordinate_channels:
        return features
      batch_size, _, frequency_bins, time_frames = features.shape
      frequency = torch.linspace(-1.0, 1.0, frequency_bins, device=features.device, dtype=features.dtype)
      time = torch.linspace(-1.0, 1.0, time_frames, device=features.device, dtype=features.dtype)
      frequency_channel = frequency.view(1, 1, frequency_bins, 1).expand(batch_size, 1, frequency_bins, time_frames)
      time_channel = time.view(1, 1, 1, time_frames).expand(batch_size, 1, frequency_bins, time_frames)
      return torch.cat((features, frequency_channel, time_channel), dim=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
      features = self.append_coordinate_channels(features)
      x = self.stem(features)
      x = self.encoder(x)
      x = self.pool(x)
      x = self.head(x)
      activity_logits = self.activity_head(x)
      parameters = torch.sigmoid(self.parameter_head(x))
      parameters = parameters.view(-1, self.config.max_oscillators, OSCILLATOR_PARAMETER_COUNT)
      f0_normalized = torch.sigmoid(self.f0_head(x)).squeeze(-1)
      return {
          "activity_logits": activity_logits,
          "parameters": parameters,
          "f0_normalized": f0_normalized,
      }


def build_model(
    input_channels: int,
    max_oscillators: int,
    width: int = 64,
    dropout: float = 0.1,
    coordinate_channels: bool = False,
    normalization: str = "batch",
) -> SoundLearnerNet:
    return SoundLearnerNet(
        ModelConfig(
            input_channels=input_channels,
            max_oscillators=max_oscillators,
            width=width,
            dropout=dropout,
            coordinate_channels=coordinate_channels,
            normalization=normalization,
        )
    )

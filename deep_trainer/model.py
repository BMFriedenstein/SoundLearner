from __future__ import annotations

from dataclasses import dataclass

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


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
      super().__init__()
      layers: list[nn.Module] = [
          nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
          nn.BatchNorm2d(out_channels),
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
      self.stem = nn.Sequential(
          nn.Conv2d(config.input_channels, width, kernel_size=4, stride=4),
          nn.BatchNorm2d(width),
          ConvNeXtBlock(width),
      )
      self.encoder = nn.Sequential(
          DownsampleStage(width, width * 2, blocks=2),
          DownsampleStage(width * 2, width * 4, blocks=2),
          DownsampleStage(width * 4, width * 8, blocks=3),
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

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
      x = self.stem(features)
      x = self.encoder(x)
      x = self.pool(x)
      x = self.head(x)
      activity_logits = self.activity_head(x)
      parameters = torch.sigmoid(self.parameter_head(x))
      parameters = parameters.view(-1, self.config.max_oscillators, OSCILLATOR_PARAMETER_COUNT)
      return {
          "activity_logits": activity_logits,
          "parameters": parameters,
      }


def build_model(input_channels: int, max_oscillators: int, width: int = 64, dropout: float = 0.1) -> SoundLearnerNet:
    return SoundLearnerNet(ModelConfig(input_channels=input_channels, max_oscillators=max_oscillators, width=width, dropout=dropout))


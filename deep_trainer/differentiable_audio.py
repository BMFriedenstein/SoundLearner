from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


K_SAMPLE_RATE = 44100.0
K_SAMPLE_INCREMENT = 1.0 / K_SAMPLE_RATE
K_MIN_AMP_CUTOFF = 0.0
K_MAX_AMP_CUTOFF = 1.0
K_MIN_AMP_DECAY_RATE = 0.99999842823
K_MAX_AMP_DECAY_RATE = 0.99921442756
K_MIN_FREQ_DECAY_RATE = 1.0
K_MAX_FREQ_DECAY_RATE = 0.99999991268
K_MAX_AMP_ATTACK_RATE = 20.0 * K_SAMPLE_INCREMENT
K_MIN_AMP_ATTACK_RATE = K_SAMPLE_INCREMENT / 150.0
FREQUENCY_ANCHORS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
K_COUPLED_DETUNE_RATIO = 0.025
K_UNCOUPLED_DETUNE_RATIO = 0.05
F0_MIN_FREQUENCY = 40.0
F0_MAX_FREQUENCY = 2000.0


@dataclass(frozen=True)
class RenderLossConfig:
    frequency_bins: int
    time_frames: int
    sample_rate: int = 11025
    render_seconds: float = 1.0
    fft_size_multiplier: int = 4
    resolution_scales: tuple[int, ...] = (1, 2, 4)


def normalize_log_frequency(frequency: torch.Tensor, minimum: float = F0_MIN_FREQUENCY, maximum: float = F0_MAX_FREQUENCY) -> torch.Tensor:
    log_minimum = math.log2(minimum)
    log_maximum = math.log2(maximum)
    clamped = frequency.clamp(min=minimum, max=maximum)
    return ((torch.log2(clamped) - log_minimum) / (log_maximum - log_minimum)).clamp(0.0, 1.0)


def denormalize_log_frequency(normalized: torch.Tensor, minimum: float = F0_MIN_FREQUENCY, maximum: float = F0_MAX_FREQUENCY) -> torch.Tensor:
    log_minimum = math.log2(minimum)
    log_maximum = math.log2(maximum)
    log_frequency = log_minimum + normalized.clamp(0.0, 1.0) * (log_maximum - log_minimum)
    return torch.pow(torch.full_like(normalized, 2.0), log_frequency)


def decode_frequency_factors(parameters: torch.Tensor) -> torch.Tensor:
    freq_factor_norm = parameters[:, :, 1].clamp(0.0, 1.0)
    coupled_probability = parameters[:, :, 6].clamp(0.0, 1.0)
    anchors = torch.tensor(FREQUENCY_ANCHORS, device=parameters.device, dtype=parameters.dtype)
    scaled = freq_factor_norm * float(len(FREQUENCY_ANCHORS))
    anchor_index = torch.floor(scaled).long().clamp(0, len(FREQUENCY_ANCHORS) - 1)
    local = (scaled - anchor_index.to(parameters.dtype)).clamp(0.0, 1.0)
    base_factor = anchors[anchor_index]

    detune = 1.0 + (local - 0.5) * 2.0 * (
        coupled_probability * K_COUPLED_DETUNE_RATIO + (1.0 - coupled_probability) * K_UNCOUPLED_DETUNE_RATIO
    )
    return base_factor * detune


def frequency_crowding_loss(parameters: torch.Tensor, activity_logits: torch.Tensor, sigma_octaves: float = 0.18) -> torch.Tensor:
    activity = torch.sigmoid(activity_logits)
    activity = activity.detach()
    factors = decode_frequency_factors(parameters).clamp_min(1e-6)
    log_factors = torch.log2(factors)
    distance = log_factors.unsqueeze(2) - log_factors.unsqueeze(1)
    pair_activity = activity.unsqueeze(2) * activity.unsqueeze(1)
    oscillator_count = parameters.shape[1]
    eye = torch.eye(oscillator_count, device=parameters.device, dtype=parameters.dtype).unsqueeze(0)
    pair_activity = pair_activity * (1.0 - eye)
    crowding = torch.exp(-torch.square(distance) / (2.0 * sigma_octaves * sigma_octaves)) * pair_activity
    normalizer = pair_activity.sum().clamp_min(1.0)
    return crowding.sum() / normalizer


def _build_log_frequency_projection(sample_rate: int, n_fft: int, frequency_bins: int) -> torch.Tensor:
    source_count = n_fft // 2 + 1
    source_freqs = torch.linspace(0.0, sample_rate / 2.0, source_count, dtype=torch.float32)
    minimum_frequency = max(20.0, float(source_freqs[1].item()) if source_count > 1 else 20.0)
    target_freqs = torch.exp(
        torch.linspace(
            math.log(minimum_frequency),
            math.log(sample_rate / 2.0),
            frequency_bins,
            dtype=torch.float32,
        )
    )

    weights = torch.zeros((frequency_bins, source_count), dtype=torch.float32)
    valid_source = source_freqs[1:]
    for row_index, target in enumerate(target_freqs):
      insertion = torch.searchsorted(valid_source, target)
      right = int(torch.clamp(insertion, 0, valid_source.numel() - 1).item()) + 1
      left = max(1, right - 1)
      left_freq = source_freqs[left]
      right_freq = source_freqs[right]
      if right == left or float(right_freq - left_freq) <= 1e-12:
        weights[row_index, left] = 1.0
      else:
        right_weight = float((target - left_freq) / (right_freq - left_freq))
        weights[row_index, left] = 1.0 - right_weight
        weights[row_index, right] = right_weight
    return weights


class DifferentiableFeatureExtractor(nn.Module):
    def __init__(self, config: RenderLossConfig) -> None:
      super().__init__()
      self.config = config
      n_fft = max(256, int(config.frequency_bins * config.fft_size_multiplier))
      if n_fft % 2 != 0:
        n_fft += 1
      self.n_fft = n_fft
      self.sample_count = max(1, int(round(config.sample_rate * config.render_seconds)))

      window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
      self.register_buffer("window", window, persistent=False)
      projection = _build_log_frequency_projection(config.sample_rate, n_fft, config.frequency_bins)
      self.register_buffer("projection", projection, persistent=False)

      padded_count = max(self.sample_count, self.n_fft)
      max_start = max(0, padded_count - self.n_fft)
      if config.time_frames <= 1:
        starts = torch.zeros((1,), dtype=torch.long)
      else:
        starts = torch.linspace(0, max_start, config.time_frames, dtype=torch.long)
      frame_offsets = torch.arange(self.n_fft, dtype=torch.long).unsqueeze(0)
      frame_index = starts.unsqueeze(1) + frame_offsets
      self.register_buffer("frame_index", frame_index, persistent=False)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
      if waveforms.dim() != 2:
        raise ValueError(f"Expected [batch, samples] waveforms, got {tuple(waveforms.shape)}")
      if waveforms.shape[1] < self.n_fft:
        waveforms = torch.nn.functional.pad(waveforms, (0, self.n_fft - waveforms.shape[1]))

      frames = waveforms[:, self.frame_index]
      windowed = frames * self.window.unsqueeze(0).unsqueeze(0)
      spectrum = torch.fft.rfft(windowed, dim=-1)
      magnitude = torch.abs(spectrum).permute(0, 2, 1)
      log_frequency = torch.einsum("fs,bst->bft", self.projection, magnitude)

      decibels = 20.0 * torch.log10(log_frequency.clamp_min(1e-6))
      decibels = decibels - decibels.amax(dim=(1, 2), keepdim=True)
      decibels = decibels.clamp(-80.0, 0.0)
      log_frequency = (decibels + 80.0) / 80.0

      delta = torch.diff(log_frequency, dim=2, prepend=log_frequency[:, :, :1])
      temporal_delta = ((delta + 1.0) * 0.5).clamp(0.0, 1.0)
      onset = delta.clamp_min(0.0).clamp_max(1.0)
      return torch.stack([log_frequency, temporal_delta, onset], dim=1)


class MultiResolutionFeatureExtractor(nn.Module):
    def __init__(self, config: RenderLossConfig) -> None:
      super().__init__()
      scales = []
      extractors: list[DifferentiableFeatureExtractor] = []
      for scale in config.resolution_scales:
        frequency_bins = max(16, config.frequency_bins // max(int(scale), 1))
        time_frames = max(8, config.time_frames // max(int(scale), 1))
        stage_config = RenderLossConfig(
            frequency_bins=frequency_bins,
            time_frames=time_frames,
            sample_rate=config.sample_rate,
            render_seconds=config.render_seconds,
            fft_size_multiplier=config.fft_size_multiplier,
            resolution_scales=(1,),
        )
        scales.append((frequency_bins, time_frames))
        extractors.append(DifferentiableFeatureExtractor(stage_config))
      self.scales = tuple(scales)
      self.extractors = nn.ModuleList(extractors)

    def forward(self, waveforms: torch.Tensor) -> list[torch.Tensor]:
      return [extractor(waveforms) for extractor in self.extractors]


class DifferentiableOscillatorBank(nn.Module):
    def __init__(self, sample_rate: int, render_seconds: float) -> None:
      super().__init__()
      self.sample_rate = float(sample_rate)
      sample_count = max(1, int(round(sample_rate * render_seconds)))
      time_axis = torch.arange(sample_count, dtype=torch.float32)
      self.register_buffer("sample_index", time_axis, persistent=False)
      self.register_buffer("time_seconds", time_axis / self.sample_rate, persistent=False)

    def forward(self, parameters: torch.Tensor, activity_logits: torch.Tensor, note_frequency: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
      if parameters.dim() != 3:
        raise ValueError(f"Expected [batch, oscillators, parameters], got {tuple(parameters.shape)}")
      batch, oscillators, parameter_count = parameters.shape
      if parameter_count != 7:
        raise ValueError(f"Expected 7 oscillator parameters, got {parameter_count}")

      amp_factor = parameters[:, :, 0]
      phase_factor = parameters[:, :, 2]
      amp_decay_factor = parameters[:, :, 3]
      amp_attack_factor = parameters[:, :, 4]
      freq_decay_factor = parameters[:, :, 5]
      activity = torch.sigmoid(activity_logits)

      amp_decay_rate = K_MAX_AMP_DECAY_RATE + (K_MIN_AMP_DECAY_RATE - K_MAX_AMP_DECAY_RATE) * amp_decay_factor
      freq_decay_rate = K_MAX_FREQ_DECAY_RATE + (K_MIN_FREQ_DECAY_RATE - K_MAX_FREQ_DECAY_RATE) * freq_decay_factor
      amp_attack_rate = K_MIN_AMP_ATTACK_RATE + (K_MAX_AMP_ATTACK_RATE - K_MIN_AMP_ATTACK_RATE) * amp_attack_factor
      frequency_factor = decode_frequency_factors(parameters)
      max_amplitude = velocity.unsqueeze(1) * (K_MIN_AMP_CUTOFF + (K_MAX_AMP_CUTOFF - K_MIN_AMP_CUTOFF) * amp_factor) * activity

      attack_length = (1.0 / amp_attack_rate.clamp_min(1e-8)).clamp_min(1.0)
      sample_index = self.sample_index.view(1, 1, -1)
      time_seconds = self.time_seconds.view(1, 1, -1)
      attack_progress = ((sample_index + 1.0) / attack_length.unsqueeze(-1)).clamp(0.0, 1.0)
      decay_steps = torch.relu(sample_index - attack_length.unsqueeze(-1))
      amplitude = max_amplitude.unsqueeze(-1) * attack_progress * torch.pow(amp_decay_rate.unsqueeze(-1), decay_steps)

      base_frequency = note_frequency.unsqueeze(1) * frequency_factor
      max_rendered_frequency = (self.sample_rate / 2.0) - 1.0
      start_frequency = base_frequency.clamp(0.0, max_rendered_frequency)
      frequency_multiplier = torch.pow(freq_decay_rate.unsqueeze(-1), decay_steps)
      instantaneous_frequency = (start_frequency.unsqueeze(-1) * frequency_multiplier).clamp(0.0, max_rendered_frequency)

      phase_offset = phase_factor.unsqueeze(-1) * math.tau
      theta = torch.cumsum(instantaneous_frequency * (1.0 / self.sample_rate), dim=2) * math.tau + phase_offset
      samples = amplitude * torch.sin(theta)
      mixed = samples.sum(dim=1)
      return mixed.clamp(-1.0, 1.0)

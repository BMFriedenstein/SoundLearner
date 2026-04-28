from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_features import read_pcm16_mono, resample_linear
from .slft import read_slft


OSCILLATOR_PARAMETER_COUNT = 7
TARGET_CHANNELS = 1 + OSCILLATOR_PARAMETER_COUNT


@dataclass(frozen=True)
class ExamplePath:
    feature_path: Path
    target_path: Path
    audio_path: Path | None = None
    audio_rms: float | None = None
    note_frequency: float | None = None
    velocity: float | None = None


def _resolve_dataset_path(root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
      return path
    return root / path


def _examples_from_metadata(root: Path) -> list[ExamplePath]:
    metadata_dir = root / "metadata"
    if not metadata_dir.exists():
      return []

    examples: list[ExamplePath] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
      metadata = json.loads(metadata_path.read_text())
      feature_path = _resolve_dataset_path(root, metadata["analysis"]["feature_path"])
      target_path = _resolve_dataset_path(root, metadata["target"]["oscillator_csv_path"])
      if feature_path.exists() and target_path.exists():
        target = metadata.get("target", {})
        audio = metadata.get("audio", {})
        audio_path = _resolve_dataset_path(root, audio["path"]) if "path" in audio else None
        examples.append(
            ExamplePath(
                feature_path=feature_path,
                target_path=target_path,
                audio_path=audio_path,
                audio_rms=float(audio["rms"]) if "rms" in audio else None,
                note_frequency=float(target["note_frequency"]) if "note_frequency" in target else None,
                velocity=float(target["velocity"]) if "velocity" in target else None,
            )
        )
    return examples


def _examples_from_features(root: Path) -> list[ExamplePath]:
    features_dir = root / "features"
    if not features_dir.exists():
      return []

    examples: list[ExamplePath] = []
    for feature_path in sorted(features_dir.glob("*.slft")):
      target_path = root / f"{feature_path.stem}.data"
      if target_path.exists():
        examples.append(ExamplePath(feature_path=feature_path, target_path=target_path))
    return examples


def discover_examples(root: str | Path) -> list[ExamplePath]:
    dataset_root = Path(root)
    examples = _examples_from_metadata(dataset_root)
    if not examples:
      examples = _examples_from_features(dataset_root)
    return examples


def read_oscillator_csv(path: str | Path, max_oscillators: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((max_oscillators, TARGET_CHANNELS), dtype=np.float32)
    mask = np.zeros((max_oscillators,), dtype=np.float32)

    rows = Path(path).read_text().splitlines()
    for row_index, row in enumerate(rows[:max_oscillators]):
      values = [float(value) for value in row.split(",")]
      if len(values) != OSCILLATOR_PARAMETER_COUNT:
        raise ValueError(f"{path} row {row_index} has {len(values)} values, expected {OSCILLATOR_PARAMETER_COUNT}")
      mask[row_index] = 1.0
      target[row_index, 0] = 1.0
      target[row_index, 1:] = np.asarray(values, dtype=np.float32)

    return target, mask


def read_audio_rms(path: Path, target_sample_rate: int = 44100) -> float:
    sample_rate, samples = read_pcm16_mono(path)
    samples = resample_linear(samples, sample_rate, target_sample_rate)
    if samples.size == 0:
      return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


class SoundLearnerDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[ExamplePath],
        max_oscillators: int,
        expected_resolution: int | None = None,
        expected_frequency_bins: int | None = None,
        expected_time_frames: int | None = None,
    ) -> None:
      self.examples = list(examples)
      self.max_oscillators = max_oscillators
      self.expected_frequency_bins = expected_frequency_bins if expected_frequency_bins is not None else expected_resolution
      self.expected_time_frames = expected_time_frames if expected_time_frames is not None else expected_resolution
      if not self.examples:
        raise ValueError("No SoundLearner examples found")

    def __len__(self) -> int:
      return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
      example = self.examples[index]
      slft = read_slft(example.feature_path)
      if self.expected_frequency_bins is not None or self.expected_time_frames is not None:
        expected_frequency_bins = self.expected_frequency_bins if self.expected_frequency_bins is not None else slft.frequency_bins
        expected_time_frames = self.expected_time_frames if self.expected_time_frames is not None else slft.time_frames
        if slft.frequency_bins != expected_frequency_bins or slft.time_frames != expected_time_frames:
          raise ValueError(
              f"{example.feature_path} is {slft.frequency_bins}x{slft.time_frames}, expected {expected_frequency_bins}x{expected_time_frames}"
          )

      target, mask = read_oscillator_csv(example.target_path, self.max_oscillators)
      active_count = max(float(mask.sum()), 1.0)
      note_frequency = example.note_frequency if example.note_frequency is not None else 1000.0
      velocity = example.velocity if example.velocity is not None else 1.0 / active_count
      source_rms = example.audio_rms
      if source_rms is None and example.audio_path is not None and example.audio_path.exists():
        source_rms = read_audio_rms(example.audio_path)
      if source_rms is None:
        source_rms = 0.0
      return {
          "features": torch.from_numpy(slft.data.astype(np.float32)),
          "target": torch.from_numpy(target),
          "mask": torch.from_numpy(mask),
          "note_frequency": torch.tensor(note_frequency, dtype=torch.float32),
          "velocity": torch.tensor(velocity, dtype=torch.float32),
          "source_rms": torch.tensor(source_rms, dtype=torch.float32),
          "feature_path": str(example.feature_path),
          "target_path": str(example.target_path),
      }

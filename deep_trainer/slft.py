from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class SlftTensor:
    sample_rate: int
    channels: int
    frequency_bins: int
    time_frames: int
    data: np.ndarray


def write_slft(path: str | Path, data: np.ndarray, sample_rate: int = 44100) -> Path:
    """Write a SoundLearner feature tensor."""
    tensor_path = Path(path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim != 3:
      raise ValueError(f"SLFT tensor must be 3D, got shape {data.shape}")

    channels, frequency_bins, time_frames = data.shape
    array = np.asarray(data, dtype="<f4", order="C")
    header = struct.pack(
        "<4s5I",
        b"SLFT",
        1,
        int(sample_rate),
        int(channels),
        int(frequency_bins),
        int(time_frames),
    )
    tensor_path.write_bytes(header + array.tobytes())
    return tensor_path


def read_slft(path: str | Path) -> SlftTensor:
    """Read a SoundLearner feature tensor."""
    tensor_path = Path(path)
    raw = tensor_path.read_bytes()
    header_size = struct.calcsize("<4s5I")
    if len(raw) < header_size:
      raise ValueError(f"{tensor_path} is too small to be an SLFT tensor")

    magic, version, sample_rate, channels, frequency_bins, time_frames = struct.unpack("<4s5I", raw[:header_size])
    if magic != b"SLFT":
      raise ValueError(f"{tensor_path} has invalid magic {magic!r}")
    if version != 1:
      raise ValueError(f"{tensor_path} uses unsupported SLFT version {version}")

    expected_values = channels * frequency_bins * time_frames
    expected_bytes = header_size + expected_values * np.dtype(np.float32).itemsize
    if len(raw) != expected_bytes:
      raise ValueError(f"{tensor_path} has {len(raw)} bytes, expected {expected_bytes}")

    data = np.frombuffer(raw, dtype="<f4", offset=header_size).copy()
    data = data.reshape((channels, frequency_bins, time_frames))
    return SlftTensor(
        sample_rate=sample_rate,
        channels=channels,
        frequency_bins=frequency_bins,
        time_frames=time_frames,
        data=data,
    )

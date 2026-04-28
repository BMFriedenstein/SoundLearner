from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
from PIL import Image

from .slft import write_slft


TARGET_SAMPLE_RATE = 44100
CHANNEL_NAMES = ("log_frequency_magnitude", "temporal_delta", "onset_emphasis")


@dataclass(frozen=True)
class FeatureSpec:
    frequency_bins: int
    time_frames: int
    crop_seconds: float = 5.0
    crop_start_seconds: float = 0.0
    fft_size_multiplier: int = 4
    sample_rate: int = TARGET_SAMPLE_RATE


def read_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
      channels = handle.getnchannels()
      sample_width = handle.getsampwidth()
      sample_rate = handle.getframerate()
      frame_count = handle.getnframes()
      frames = handle.readframes(frame_count)
    if sample_width != 2:
      raise ValueError(f"{path} uses {sample_width * 8}-bit samples; only 16-bit PCM WAV is supported.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
      samples = samples.reshape((-1, channels)).mean(axis=1)
    return sample_rate, samples


def estimate_fundamental_frequency(
    samples: np.ndarray,
    sample_rate: int,
    minimum_hz: float = 40.0,
    maximum_hz: float = 1200.0,
    frame_seconds: float = 0.08,
    hop_seconds: float = 0.04,
) -> float | None:
    samples = samples.astype(np.float32, copy=False)
    if samples.size == 0:
      return None
    samples = samples - float(np.mean(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < 1e-4:
      return None

    frame_size = max(256, int(round(frame_seconds * sample_rate)))
    hop_size = max(1, int(round(hop_seconds * sample_rate)))
    if samples.size < frame_size:
      samples = np.pad(samples, (0, frame_size - samples.size))

    min_lag = max(1, int(sample_rate / maximum_hz))
    max_lag = min(frame_size - 2, int(sample_rate / minimum_hz))
    if max_lag <= min_lag:
      return None

    estimates: list[float] = []
    strengths: list[float] = []
    window = np.hanning(frame_size).astype(np.float32)
    for start in range(0, max(1, samples.size - frame_size + 1), hop_size):
      frame = samples[start : start + frame_size]
      if frame.size < frame_size:
        frame = np.pad(frame, (0, frame_size - frame.size))
      frame = (frame - float(np.mean(frame))) * window
      energy = float(np.dot(frame, frame))
      if energy < 1e-7:
        continue
      correlation = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
      search = correlation[min_lag : max_lag + 1]
      if search.size == 0:
        continue
      lag = int(np.argmax(search)) + min_lag
      strength = float(correlation[lag] / max(correlation[0], 1e-8))
      if strength < 0.25:
        continue
      if 1 <= lag < correlation.size - 1:
        left = float(correlation[lag - 1])
        center = float(correlation[lag])
        right = float(correlation[lag + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
          lag = lag + int(0)
          offset = 0.5 * (left - right) / denominator
          lag_value = float(lag) + float(np.clip(offset, -0.5, 0.5))
        else:
          lag_value = float(lag)
      else:
        lag_value = float(lag)
      estimates.append(sample_rate / lag_value)
      strengths.append(strength)

    if not estimates:
      return None
    log_estimates = np.log2(np.asarray(estimates, dtype=np.float64))
    weights = np.asarray(strengths, dtype=np.float64)
    center = float(np.average(log_estimates, weights=weights))
    return float(2.0 ** center)


def estimate_fundamental_frequency_from_wav(path: Path, minimum_hz: float = 40.0, maximum_hz: float = 1200.0) -> float | None:
    sample_rate, samples = read_pcm16_mono(path)
    return estimate_fundamental_frequency(samples, sample_rate, minimum_hz=minimum_hz, maximum_hz=maximum_hz)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
      return samples.astype(np.float32, copy=False)
    duration_seconds = samples.size / float(source_rate)
    target_count = max(1, int(round(duration_seconds * target_rate)))
    source_positions = np.linspace(0.0, duration_seconds, samples.size, endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration_seconds, target_count, endpoint=False, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def crop_or_pad_samples(samples: np.ndarray, sample_rate: int, crop_seconds: float, crop_start_seconds: float) -> np.ndarray:
    start_sample = int(round(crop_start_seconds * sample_rate))
    sample_count = int(round(crop_seconds * sample_rate))
    if sample_count <= 0:
      raise ValueError("crop_seconds must be positive")
    cropped = np.zeros((sample_count,), dtype=np.float32)
    if start_sample >= samples.size:
      return cropped
    copy_count = min(sample_count, samples.size - start_sample)
    cropped[:copy_count] = samples[start_sample : start_sample + copy_count]
    return cropped


def _frame_starts(sample_count: int, n_fft: int, time_frames: int) -> np.ndarray:
    padded_count = max(sample_count, n_fft)
    max_start = max(0, padded_count - n_fft)
    if time_frames <= 1:
      return np.zeros((1,), dtype=np.int64)
    return np.linspace(0, max_start, time_frames, dtype=np.int64)


def _stft_magnitude(samples: np.ndarray, n_fft: int, time_frames: int) -> np.ndarray:
    working = samples
    if working.size < n_fft:
      working = np.pad(working, (0, n_fft - working.size))
    starts = _frame_starts(working.size, n_fft, time_frames)
    window = np.hanning(n_fft).astype(np.float32)
    magnitude = np.empty((n_fft // 2 + 1, time_frames), dtype=np.float32)
    for frame_index, start in enumerate(starts):
      frame = working[start : start + n_fft]
      if frame.size < n_fft:
        frame = np.pad(frame, (0, n_fft - frame.size))
      spectrum = np.fft.rfft(frame * window)
      magnitude[:, frame_index] = np.abs(spectrum).astype(np.float32)
    return magnitude


def _log_frequency_remap(magnitude: np.ndarray, sample_rate: int, frequency_bins: int) -> np.ndarray:
    linear_frequency_count = magnitude.shape[0]
    source_frequencies = np.linspace(0.0, sample_rate / 2.0, linear_frequency_count, dtype=np.float64)
    minimum_frequency = max(20.0, float(source_frequencies[1]) if linear_frequency_count > 1 else 20.0)
    target_frequencies = np.geomspace(minimum_frequency, sample_rate / 2.0, frequency_bins, dtype=np.float64)

    source = magnitude[1:, :]
    remapped = np.empty((frequency_bins, magnitude.shape[1]), dtype=np.float32)
    for frame_index in range(magnitude.shape[1]):
      remapped[:, frame_index] = np.interp(target_frequencies, source_frequencies[1:], source[:, frame_index]).astype(np.float32)
    return remapped


def _normalize_log_magnitude(magnitude: np.ndarray) -> np.ndarray:
    decibels = 20.0 * np.log10(np.maximum(magnitude, 1e-6))
    decibels -= float(np.max(decibels))
    decibels = np.clip(decibels, -80.0, 0.0)
    return ((decibels + 80.0) / 80.0).astype(np.float32)


def extract_feature_tensor_from_samples(samples: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    n_fft = max(256, int(spec.frequency_bins * spec.fft_size_multiplier))
    if n_fft % 2 != 0:
      n_fft += 1
    magnitude = _stft_magnitude(samples, n_fft=n_fft, time_frames=spec.time_frames)
    log_frequency = _log_frequency_remap(magnitude, spec.sample_rate, spec.frequency_bins)
    log_frequency = _normalize_log_magnitude(log_frequency)

    delta = np.diff(log_frequency, axis=1, prepend=log_frequency[:, :1])
    temporal_delta = np.clip((delta + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    onset = np.clip(delta, 0.0, 1.0).astype(np.float32)

    return np.stack([log_frequency, temporal_delta, onset], axis=0).astype(np.float32)


def extract_feature_tensor_from_wav(path: Path, spec: FeatureSpec) -> np.ndarray:
    sample_rate, samples = read_pcm16_mono(path)
    samples = resample_linear(samples, sample_rate, spec.sample_rate)
    cropped = crop_or_pad_samples(samples, spec.sample_rate, spec.crop_seconds, spec.crop_start_seconds)
    return extract_feature_tensor_from_samples(cropped, spec)


def write_feature_tensor(path: Path, features: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    return write_slft(path, features, sample_rate=sample_rate)


def inferno_like_colormap(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.0, 0.0, 4.0, 30.0],
            [0.2, 52.0, 16.0, 92.0],
            [0.4, 120.0, 28.0, 109.0],
            [0.6, 187.0, 55.0, 84.0],
            [0.8, 249.0, 142.0, 8.0],
            [1.0, 252.0, 255.0, 164.0],
        ],
        dtype=np.float32,
    )
    positions = stops[:, 0]
    red = np.interp(values, positions, stops[:, 1])
    green = np.interp(values, positions, stops[:, 2])
    blue = np.interp(values, positions, stops[:, 3])
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def feature_preview_image(features: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    channel = np.clip(features[0], 0.0, 1.0)
    base = inferno_like_colormap(channel)
    image = Image.fromarray(base, mode="RGB")
    if width is not None and height is not None:
      image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image)


def write_feature_preview_bmp(path: Path, features: np.ndarray, width: int | None = None, height: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(feature_preview_image(features, width=width, height=height), mode="RGB").save(path)
    return path

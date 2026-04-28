from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .audio_features import inferno_like_colormap, read_pcm16_mono


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int, f_min: float = 20.0, f_max: float | None = None) -> np.ndarray:
    maximum_frequency = f_max if f_max is not None else sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(np.array([f_min]))[0], hz_to_mel(np.array([maximum_frequency]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filter_bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_index in range(n_mels):
      left = bin_indices[mel_index]
      center = bin_indices[mel_index + 1]
      right = bin_indices[mel_index + 2]
      if center <= left:
        center = left + 1
      if right <= center:
        right = center + 1
      for freq_bin in range(left, min(center, filter_bank.shape[1])):
        filter_bank[mel_index, freq_bin] = (freq_bin - left) / max(center - left, 1)
      for freq_bin in range(center, min(right, filter_bank.shape[1])):
        filter_bank[mel_index, freq_bin] = (right - freq_bin) / max(right - center, 1)
    return filter_bank


def stft_power(samples: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    if samples.size == 0:
      return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    if samples.size < n_fft:
      samples = np.pad(samples, (0, n_fft - samples.size))
    frame_count = 1 + max(0, (samples.size - n_fft) // hop_length)
    if frame_count <= 0:
      frame_count = 1
    window = np.hanning(n_fft).astype(np.float32)
    spectrogram = np.empty((n_fft // 2 + 1, frame_count), dtype=np.float32)
    for frame_index in range(frame_count):
      start = frame_index * hop_length
      frame = samples[start : start + n_fft]
      if frame.size < n_fft:
        frame = np.pad(frame, (0, n_fft - frame.size))
      spectrum = np.fft.rfft(frame * window)
      spectrogram[:, frame_index] = (np.abs(spectrum) ** 2).astype(np.float32)
    return spectrogram


def mel_spectrogram(samples: np.ndarray, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    power = stft_power(samples, n_fft=n_fft, hop_length=hop_length)
    filters = create_mel_filter_bank(sample_rate, n_fft, n_mels)
    mel = filters @ power
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    mel_db -= float(np.max(mel_db))
    mel_db = np.clip(mel_db, -80.0, 0.0)
    normalized = (mel_db + 80.0) / 80.0
    return np.flipud(normalized.astype(np.float32))


def spectrogram_image_array(path: Path, width: int = 900, height: int = 320) -> np.ndarray:
    sample_rate, samples = read_pcm16_mono(path)
    mel = mel_spectrogram(samples, sample_rate)
    base = inferno_like_colormap(mel)
    image = Image.fromarray(base, mode="RGB").resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image)


def write_mel_preview(wav_path: Path, output_path: Path, width: int = 900, height: int = 320) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(spectrogram_image_array(wav_path, width=width, height=height), mode="RGB")
    image.save(output_path)
    return output_path


def write_ab_mel_preview(source_wav: Path, rendered_wav: Path, output_path: Path, width: int = 900, height: int = 320, gap: int = 16) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = spectrogram_image_array(source_wav, width=width, height=height)
    rendered = spectrogram_image_array(rendered_wav, width=width, height=height)
    separator = np.full((height, gap, 3), 20, dtype=np.uint8)
    combined = np.concatenate([source, separator, rendered], axis=1)
    Image.fromarray(combined, mode="RGB").save(output_path)
    return output_path

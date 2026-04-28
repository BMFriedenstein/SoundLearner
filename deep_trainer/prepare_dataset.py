from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audio_features import CHANNEL_NAMES, FeatureSpec, extract_feature_tensor_from_wav, read_pcm16_mono, resample_linear, write_feature_preview_bmp, write_feature_tensor
from .audio_preview import write_mel_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a SoundLearner dataset with Python feature tensors and previews.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--freq-bins", type=int, default=None)
    parser.add_argument("--time-frames", type=int, default=None)
    parser.add_argument("--crop-seconds", type=float, default=5.0)
    parser.add_argument("--crop-start-seconds", type=float, default=0.0)
    parser.add_argument("--fft-size-multiplier", type=int, default=4)
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--skip-mel-previews", action="store_true")
    return parser.parse_args()


def path_for_record(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def discover_sample_stems(root: Path) -> list[str]:
    stems = sorted(path.stem for path in root.glob("data*.wav"))
    if not stems:
      raise ValueError(f"No data*.wav files found in {root}")
    return stems


def parse_legacy_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
      return {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    values = [float(line) for line in lines]
    metadata: dict[str, Any] = {}
    if len(values) >= 1:
      metadata["note_frequency"] = values[0]
    if len(values) >= 2:
      metadata["velocity"] = values[1]
    if len(values) >= 3:
      metadata["coupled_oscillator_count"] = int(values[2])
    if len(values) >= 4:
      metadata["uncoupled_oscillator_count"] = int(values[3])
    return metadata


def build_metadata(
    sample_id: str,
    dataset_root: Path,
    wav_path: Path,
    target_path: Path,
    feature_path: Path,
    legacy_meta: dict[str, Any],
    spec: FeatureSpec,
    sample_count: int,
    audio_rms: float,
    include_previews: bool,
    include_mel_preview: bool,
) -> dict[str, Any]:
    previews: dict[str, str] = {}
    if include_previews:
      previews["spectrogram_rgb"] = f"preview/{sample_id}_rgb.bmp"
      previews["log_frequency_rgb"] = f"preview/{sample_id}_logfreq_rgb.bmp"
    if include_mel_preview:
      previews["mel_spectrogram_png"] = f"mel_preview/{sample_id}_mel.png"

    coupled = int(legacy_meta.get("coupled_oscillator_count", 0))
    uncoupled = int(legacy_meta.get("uncoupled_oscillator_count", 0))

    return {
        "id": sample_id,
        "audio": {
            "path": path_for_record(wav_path, dataset_root),
            "sample_rate": spec.sample_rate,
            "sample_count": sample_count,
            "rms": audio_rms,
        },
        "analysis": {
            "feature_path": path_for_record(feature_path, dataset_root),
            "format": "SLFT.float32.v1",
            "frequency_scale": "python_log_frequency",
            "frequency_bins": spec.frequency_bins,
            "time_frames": spec.time_frames,
            "channels": list(CHANNEL_NAMES),
        },
        "target": {
            "note_frequency": float(legacy_meta.get("note_frequency", 440.0)),
            "velocity": float(legacy_meta.get("velocity", 0.0)),
            "coupled_oscillator_count": coupled,
            "uncoupled_oscillator_count": uncoupled,
            "total_oscillator_count": coupled + uncoupled,
            "oscillator_csv_path": path_for_record(target_path, dataset_root),
        },
        "previews": previews,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    freq_bins = args.freq_bins if args.freq_bins is not None else args.resolution
    time_frames = args.time_frames if args.time_frames is not None else args.resolution
    if freq_bins is None or time_frames is None:
      raise ValueError("Use --resolution or both --freq-bins and --time-frames.")

    spec = FeatureSpec(
        frequency_bins=freq_bins,
        time_frames=time_frames,
        crop_seconds=args.crop_seconds,
        crop_start_seconds=args.crop_start_seconds,
        fft_size_multiplier=args.fft_size_multiplier,
    )

    features_dir = dataset_root / "features"
    metadata_dir = dataset_root / "metadata"
    preview_dir = dataset_root / "preview"
    mel_dir = dataset_root / "mel_preview"
    features_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_previews:
      preview_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mel_previews:
      mel_dir.mkdir(parents=True, exist_ok=True)

    stems = discover_sample_stems(dataset_root)
    for index, stem in enumerate(stems, start=1):
      wav_path = dataset_root / f"{stem}.wav"
      target_path = dataset_root / f"{stem}.data"
      feature_path = features_dir / f"{stem}.slft"
      metadata_path = metadata_dir / f"{stem}.json"
      legacy_meta_path = dataset_root / f"{stem}.meta"

      features = extract_feature_tensor_from_wav(wav_path, spec)
      write_feature_tensor(feature_path, features, sample_rate=spec.sample_rate)

      if not args.skip_previews:
        rgb_path = preview_dir / f"{stem}_rgb.bmp"
        logfreq_path = preview_dir / f"{stem}_logfreq_rgb.bmp"
        write_feature_preview_bmp(rgb_path, features)
        write_feature_preview_bmp(logfreq_path, features)
      if not args.skip_mel_previews:
        write_mel_preview(wav_path, mel_dir / f"{stem}_mel.png")

      sample_count = int(round(spec.crop_seconds * spec.sample_rate))
      input_sample_rate, samples = read_pcm16_mono(wav_path)
      resampled = resample_linear(samples, input_sample_rate, spec.sample_rate)
      audio_rms = float((resampled.astype("float64") ** 2).mean() ** 0.5) if resampled.size else 0.0
      metadata = build_metadata(
          sample_id=stem,
          dataset_root=dataset_root,
          wav_path=wav_path,
          target_path=target_path,
          feature_path=feature_path,
          legacy_meta=parse_legacy_meta(legacy_meta_path),
          spec=spec,
          sample_count=sample_count,
          audio_rms=audio_rms,
          include_previews=not args.skip_previews,
          include_mel_preview=not args.skip_mel_previews,
      )
      metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
      print(f"[{index}/{len(stems)}] prepared {stem}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
import wave

import numpy as np

from .audio_features import FeatureSpec, extract_feature_tensor_from_wav, write_feature_preview_bmp, write_feature_tensor
from .audio_preview import write_mel_preview

@dataclass(frozen=True)
class DatasetItem:
    metadata_path: Path
    metadata: dict[str, Any]
    input_wav: Path
    target_csv: Path
    source_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a clean SoundLearner dataset into a more real-world-ish sibling dataset.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants-per-input", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--freq-bins", type=int, default=None)
    parser.add_argument("--time-frames", type=int, default=None)
    parser.add_argument("--crop-seconds", type=float, default=None)
    parser.add_argument("--crop-start-seconds", type=float, default=0.0)
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--skip-mel-previews", action="store_true")

    parser.add_argument("--gain-db-min", type=float, default=-3.0)
    parser.add_argument("--gain-db-max", type=float, default=3.0)
    parser.add_argument("--snr-db-min", type=float, default=28.0)
    parser.add_argument("--snr-db-max", type=float, default=42.0)
    parser.add_argument("--low-shelf-db-min", type=float, default=-3.0)
    parser.add_argument("--low-shelf-db-max", type=float, default=3.0)
    parser.add_argument("--high-shelf-db-min", type=float, default=-3.0)
    parser.add_argument("--high-shelf-db-max", type=float, default=3.0)
    parser.add_argument("--reverb-mix-max", type=float, default=0.08)
    parser.add_argument("--saturation-max", type=float, default=0.12)

    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--disable-tone", action="store_true")
    parser.add_argument("--disable-reverb", action="store_true")
    parser.add_argument("--disable-saturation", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def path_for_record(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def db_to_linear(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def clamp_audio(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0.999:
        signal = signal / peak * 0.999
    return np.clip(signal, -0.999, 0.999)


def read_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)
    if channels != 1:
        raise ValueError(f"{path} has {channels} channels; dataset_augmentor currently expects mono WAVs.")
    if sample_width != 2:
        raise ValueError(f"{path} uses {sample_width * 8}-bit samples; dataset_augmentor currently expects 16-bit PCM.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return sample_rate, samples


def write_pcm16_mono(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def lowpass_one_pole(signal: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty_like(signal)
    state = 0.0
    for index, sample in enumerate(signal):
        state += alpha * (float(sample) - state)
        output[index] = state
    return output


def sparse_reverb(signal: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    wet = np.copy(signal)
    delay_count = int(rng.integers(2, 5))
    for _ in range(delay_count):
        delay_ms = float(rng.uniform(8.0, 55.0))
        delay_samples = max(1, int(sample_rate * delay_ms / 1000.0))
        feedback = float(rng.uniform(0.08, 0.22))
        if delay_samples >= len(signal):
            continue
        wet[delay_samples:] += signal[:-delay_samples] * feedback
    return wet


def augment_audio(signal: np.ndarray, sample_rate: int, args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    augmented = np.copy(signal)
    params: dict[str, float] = {}

    gain_db = float(rng.uniform(args.gain_db_min, args.gain_db_max))
    params["gain_db"] = gain_db
    augmented *= db_to_linear(gain_db)

    if not args.disable_tone:
        low_alpha = float(rng.uniform(0.004, 0.035))
        low_shelf_db = float(rng.uniform(args.low_shelf_db_min, args.low_shelf_db_max))
        high_shelf_db = float(rng.uniform(args.high_shelf_db_min, args.high_shelf_db_max))
        low = lowpass_one_pole(augmented, low_alpha)
        high = augmented - low
        augmented = db_to_linear(low_shelf_db) * low + db_to_linear(high_shelf_db) * high
        params["low_alpha"] = low_alpha
        params["low_shelf_db"] = low_shelf_db
        params["high_shelf_db"] = high_shelf_db

    if not args.disable_noise:
        signal_rms = float(np.sqrt(np.mean(np.square(augmented))) + 1e-8)
        snr_db = float(rng.uniform(args.snr_db_min, args.snr_db_max))
        noise = rng.standard_normal(augmented.shape[0]).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-8)
        target_noise_rms = signal_rms / db_to_linear(snr_db)
        augmented = augmented + noise * (target_noise_rms / noise_rms)
        params["snr_db"] = snr_db

    if not args.disable_reverb:
        reverb_mix = float(rng.uniform(0.0, args.reverb_mix_max))
        params["reverb_mix"] = reverb_mix
        if reverb_mix > 0.0:
            wet = sparse_reverb(augmented, sample_rate, rng)
            augmented = (1.0 - reverb_mix) * augmented + reverb_mix * wet

    if not args.disable_saturation:
        saturation_drive = float(rng.uniform(0.0, args.saturation_max))
        params["saturation_drive"] = saturation_drive
        if saturation_drive > 0.0:
            drive = 1.0 + saturation_drive * 6.0
            augmented = np.tanh(augmented * drive) / np.tanh(drive)

    return clamp_audio(augmented), params


def discover_items(input_root: Path, limit: int | None) -> list[DatasetItem]:
    metadata_dir = input_root / "metadata"
    if not metadata_dir.exists():
        raise ValueError(f"{input_root} has no metadata directory; dataset_augmentor expects a generated dataset with metadata/*.json.")

    items: list[DatasetItem] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        source_id = str(metadata.get("id", metadata_path.stem))
        input_wav = input_root / metadata["audio"]["path"]
        target_csv = input_root / metadata["target"]["oscillator_csv_path"]
        if input_wav.exists() and target_csv.exists():
            items.append(
                DatasetItem(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    input_wav=input_wav,
                    target_csv=target_csv,
                    source_id=source_id,
                )
            )
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError(f"No dataset items found in {input_root}")
    return items


def output_feature_shape(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[int, int, int]:
    analysis = metadata["analysis"]
    resolution = args.resolution if args.resolution is not None else int(analysis.get("preview_resolution", 512))
    frequency_bins = args.freq_bins if args.freq_bins is not None else int(analysis["frequency_bins"])
    time_frames = args.time_frames if args.time_frames is not None else int(analysis["time_frames"])
    return resolution, frequency_bins, time_frames


def output_crop_seconds(args: argparse.Namespace, metadata: dict[str, Any]) -> float:
    if args.crop_seconds is not None:
        return args.crop_seconds
    audio = metadata["audio"]
    return float(audio["sample_count"]) / float(audio["sample_rate"])


def build_metadata(
    sample_id: str,
    source_item: DatasetItem,
    wav_path: str,
    feature_path: str,
    target_csv_path: str,
    preview_prefix: str,
    preview_resolution: int,
    frequency_bins: int,
    time_frames: int,
    augmentation: dict[str, float],
) -> dict[str, Any]:
    source_metadata = source_item.metadata
    source_target = dict(source_metadata.get("target", {}))
    previews: dict[str, str] = {
        "spectrogram_rgb": f"{preview_prefix}_rgb.bmp",
        "log_frequency_rgb": f"{preview_prefix}_logfreq_rgb.bmp",
    }

    return {
        "id": sample_id,
        "source": {
            "dataset_id": source_item.source_id,
            "metadata_path": path_for_record(source_item.metadata_path, repo_root()),
            "audio_path": path_for_record(source_item.input_wav, repo_root()),
        },
        "audio": {
            "path": wav_path,
            "sample_rate": int(source_metadata["audio"]["sample_rate"]),
            "sample_count": int(source_metadata["audio"]["sample_count"]),
        },
        "analysis": {
            "feature_path": feature_path,
            "format": "SLFT.float32.v1",
            "frequency_scale": "log_fft_bin",
            "frequency_bins": frequency_bins,
            "time_frames": time_frames,
            "preview_resolution": preview_resolution,
            "channels": ["log_frequency_magnitude", "temporal_delta", "onset_emphasis"],
        },
        "target": {
            **source_target,
            "oscillator_csv_path": target_csv_path,
        },
        "augmentation": augmentation,
        "previews": previews,
    }


def main() -> None:
    args = parse_args()
    if args.variants_per_input < 1:
        raise ValueError("--variants-per-input must be at least 1")

    root = repo_root()
    rng = np.random.default_rng(args.seed)
    items = discover_items(args.input_root, args.limit)

    output_root = args.output_root
    (output_root / "features").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    if not args.skip_previews:
        (output_root / "preview").mkdir(parents=True, exist_ok=True)
    if not args.skip_mel_previews:
        (output_root / "mel_preview").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    sample_index = 0

    for item in items:
        for variant in range(args.variants_per_input):
            sample_id = f"{item.source_id}_aug{variant:02d}"
            output_wav = output_root / f"{sample_id}.wav"
            output_target = output_root / f"{sample_id}.data"
            output_meta_legacy = output_root / f"{sample_id}.meta"
            output_feature = output_root / "features" / f"{sample_id}.slft"
            output_metadata = output_root / "metadata" / f"{sample_id}.json"
            preview_prefix = output_root / "preview" / sample_id

            sample_rate, signal = read_pcm16_mono(item.input_wav)
            augmented_signal, augmentation = augment_audio(signal, sample_rate, args, rng)
            write_pcm16_mono(output_wav, sample_rate, augmented_signal)

            shutil.copyfile(item.target_csv, output_target)
            source_legacy_meta = item.input_wav.with_suffix(".meta")
            if source_legacy_meta.exists():
                shutil.copyfile(source_legacy_meta, output_meta_legacy)

            resolution, frequency_bins, time_frames = output_feature_shape(args, item.metadata)
            crop_seconds = output_crop_seconds(args, item.metadata)
            spec = FeatureSpec(
                frequency_bins=frequency_bins,
                time_frames=time_frames,
                crop_seconds=crop_seconds,
                crop_start_seconds=args.crop_start_seconds,
            )
            features = extract_feature_tensor_from_wav(output_wav, spec)
            write_feature_tensor(output_feature, features)
            if not args.skip_previews:
                write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_rgb.bmp"), features)
                write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_logfreq_rgb.bmp"), features)
            mel_preview_path: str | None = None
            if not args.skip_mel_previews:
                mel_output = output_root / "mel_preview" / f"{sample_id}_mel.png"
                write_mel_preview(output_wav, mel_output)
                mel_preview_path = path_for_record(mel_output, root)

            preview_record_prefix = path_for_record(preview_prefix, root)
            metadata = build_metadata(
                sample_id=sample_id,
                source_item=item,
                wav_path=path_for_record(output_wav, root),
                feature_path=path_for_record(output_feature, root),
                target_csv_path=path_for_record(output_target, root),
                preview_prefix=preview_record_prefix,
                preview_resolution=resolution,
                frequency_bins=frequency_bins,
                time_frames=time_frames,
                augmentation=augmentation,
            )
            if mel_preview_path is not None:
                metadata.setdefault("previews", {})["mel_spectrogram_png"] = mel_preview_path
            output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")

            manifest_rows.append(
                {
                    "id": sample_id,
                    "source_id": item.source_id,
                    "variant": str(variant),
                    "input_wav": path_for_record(output_wav, root),
                    "feature_tensor": path_for_record(output_feature, root),
                    "target_csv": path_for_record(output_target, root),
                    "metadata_json": path_for_record(output_metadata, root),
                    "gain_db": f"{augmentation.get('gain_db', 0.0):.6f}",
                    "snr_db": f"{augmentation.get('snr_db', 0.0):.6f}",
                    "low_shelf_db": f"{augmentation.get('low_shelf_db', 0.0):.6f}",
                    "high_shelf_db": f"{augmentation.get('high_shelf_db', 0.0):.6f}",
                    "reverb_mix": f"{augmentation.get('reverb_mix', 0.0):.6f}",
                    "saturation_drive": f"{augmentation.get('saturation_drive', 0.0):.6f}",
                }
            )
            sample_index += 1
            print(f"[{sample_index}/{len(items) * args.variants_per_input}] wrote {sample_id}")

    manifest_path = output_root / "augmentation_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    config = {
        "input_root": path_for_record(args.input_root, root),
        "output_root": path_for_record(args.output_root, root),
        "variants_per_input": args.variants_per_input,
        "limit": args.limit,
        "seed": args.seed,
        "resolution": args.resolution,
        "freq_bins": args.freq_bins,
        "time_frames": args.time_frames,
        "crop_seconds": args.crop_seconds,
        "crop_start_seconds": args.crop_start_seconds,
        "skip_previews": args.skip_previews,
        "skip_mel_previews": args.skip_mel_previews,
        "gain_db_range": [args.gain_db_min, args.gain_db_max],
        "snr_db_range": [args.snr_db_min, args.snr_db_max],
        "low_shelf_db_range": [args.low_shelf_db_min, args.low_shelf_db_max],
        "high_shelf_db_range": [args.high_shelf_db_min, args.high_shelf_db_max],
        "reverb_mix_max": args.reverb_mix_max,
        "saturation_max": args.saturation_max,
        "disabled_effects": {
            "noise": args.disable_noise,
            "tone": args.disable_tone,
            "reverb": args.disable_reverb,
            "saturation": args.disable_saturation,
        },
    }
    (output_root / "augmentation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote augmented dataset to {output_root}")


if __name__ == "__main__":
    main()

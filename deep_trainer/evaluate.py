from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any
import wave

import numpy as np
import torch

from .audio_features import FeatureSpec, extract_feature_tensor_from_wav, write_feature_preview_bmp, write_feature_tensor
from .audio_preview import write_ab_mel_preview, write_mel_preview
from .differentiable_audio import denormalize_log_frequency
from .model import SoundLearnerNet, model_config_from_mapping
from .slft import read_slft


NOTE_OFFSETS = {
    "c": -9,
    "cs": -8,
    "db": -8,
    "d": -7,
    "ds": -6,
    "eb": -6,
    "e": -5,
    "f": -4,
    "fs": -3,
    "gb": -3,
    "g": -2,
    "gs": -1,
    "ab": -1,
    "a": 0,
    "as": 1,
    "bb": 1,
    "b": 2,
}


@dataclass(frozen=True)
class EvaluationItem:
    name: str
    input_wav: Path
    source_feature: Path | None = None
    note_frequency: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint by predicting, rendering, and comparing real-audio examples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="CSV manifest with at least name and input_wav columns.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="One WAV file to evaluate. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, default=Path("sounds/eval/latest"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--freq-bins", type=int, default=None)
    parser.add_argument("--time-frames", type=int, default=None)
    parser.add_argument("--crop-seconds", type=float, default=5.0)
    parser.add_argument("--crop-start-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--write-all-slots", action="store_true")
    parser.add_argument("--note-frequency", type=float, default=None, help="Override rendered note frequency for all examples.")
    parser.add_argument("--velocity", type=int, default=2, help="Player velocity percent. Synthetic training used 2.")
    parser.add_argument("--length", type=int, default=5, help="Rendered length in seconds.")
    parser.add_argument("--tool-mode", choices=["wsl", "native"], default="wsl")
    parser.add_argument("--player", type=Path, default=Path("build/player/player"))
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def frequency_bins(args: argparse.Namespace) -> int:
    return args.freq_bins if args.freq_bins is not None else args.resolution


def time_frames(args: argparse.Namespace) -> int:
    return args.time_frames if args.time_frames is not None else args.resolution


def path_for_record(path: Path, root: Path) -> str:
    try:
      return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
      return str(path)


def path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[-1]
    if drive:
      return f"/mnt/{drive}{rest}"
    return resolved.as_posix()


def run_tool(args: argparse.Namespace, command: list[str]) -> None:
    root = repo_root()
    if args.tool_mode == "native":
      completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
      if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
      return

    quoted = " ".join(shlex.quote(arg) for arg in command)
    bash_command = f"cd {shlex.quote(path_to_wsl(root))} && {quoted}"
    completed = subprocess.run(["bash", "-lc", bash_command], text=True, capture_output=True)
    if completed.returncode != 0:
      print(completed.stdout)
      print(completed.stderr)
      completed.check_returncode()


def format_tool_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
      return str(int(numeric))
    return str(numeric)


def read_manifest(path: Path) -> list[EvaluationItem]:
    root = repo_root()
    items: list[EvaluationItem] = []
    with path.open(newline="") as handle:
      for row in csv.DictReader(handle):
        name = row.get("name") or Path(row["input_wav"]).stem
        input_wav = root / row["input_wav"]
        feature_value = row.get("feature_tensor") or ""
        source_feature = root / feature_value if feature_value else None
        note_value = row.get("note_frequency") or ""
        note_frequency = float(note_value) if note_value else None
        items.append(EvaluationItem(name=name, input_wav=input_wav, source_feature=source_feature, note_frequency=note_frequency))
    return items


def discover_items(args: argparse.Namespace) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    if args.manifest is not None:
      items.extend(read_manifest(args.manifest))
    for input_path in args.input:
      items.append(EvaluationItem(name=input_path.stem, input_wav=input_path))
    if args.limit is not None:
      items = items[: args.limit]
    if not items:
      raise ValueError("No evaluation inputs. Use --manifest or --input.")
    return items


def infer_note_frequency(name: str) -> float:
    match = re.match(r"^o_([a-g](?:s|b)?)(0?\d)", name.lower())
    if not match:
      return 440.0
    note, octave_text = match.groups()
    octave = 0 if octave_text.startswith("0") else int(octave_text)
    semitones_from_a4 = (octave - 4) * 12 + NOTE_OFFSETS[note]
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


def load_model(checkpoint_path: Path, requested_device: str) -> tuple[SoundLearnerNet, torch.device]:
    if requested_device == "auto":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoundLearnerNet(model_config_from_mapping(checkpoint.get("model_config"))).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys:
      print(f"Checkpoint is missing new model keys; initialized randomly: {', '.join(incompatible.missing_keys)}")
    model.eval()
    return model, device


def predict_instrument(
    model: SoundLearnerNet,
    device: torch.device,
    feature_path: Path,
    output_path: Path,
    activity_threshold: float,
    write_all_slots: bool,
) -> tuple[int, float | None]:
    slft = read_slft(feature_path)
    features = torch.from_numpy(slft.data).unsqueeze(0).to(device)
    with torch.no_grad():
      prediction = model(features)
      activity = torch.sigmoid(prediction["activity_logits"])[0].cpu()
      parameters = prediction["parameters"][0].cpu()
      predicted_note = None
      if "f0_normalized" in prediction:
        predicted_note = float(denormalize_log_frequency(prediction["f0_normalized"])[0].detach().cpu())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for slot_index in range(model.config.max_oscillators):
      if not write_all_slots and activity[slot_index].item() < activity_threshold:
        continue
      values = parameters[slot_index].tolist()
      values[-1] = 1.0 if values[-1] >= 0.5 else 0.0
      rows.append(",".join(f"{value:.6f}" for value in values))
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows), predicted_note


def ensure_source_feature(args: argparse.Namespace, item: EvaluationItem, feature_path: Path, preview_prefix: Path) -> Path:
    if item.source_feature is not None and item.source_feature.exists():
      try:
        source = read_slft(item.source_feature)
        if source.frequency_bins == frequency_bins(args) and source.time_frames == time_frames(args):
          feature_path.parent.mkdir(parents=True, exist_ok=True)
          shutil.copyfile(item.source_feature, feature_path)
          return feature_path
      except ValueError:
        pass

    spec = FeatureSpec(
        frequency_bins=frequency_bins(args),
        time_frames=time_frames(args),
        crop_seconds=args.crop_seconds,
        crop_start_seconds=args.crop_start_seconds,
    )
    features = extract_feature_tensor_from_wav(item.input_wav, spec)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    preview_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_feature_tensor(feature_path, features)
    write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_rgb.bmp"), features)
    write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_logfreq_rgb.bmp"), features)
    return feature_path


def render_prediction(args: argparse.Namespace, instrument_path: Path, note_frequency: float, render_dir: Path) -> Path:
    render_dir.mkdir(parents=True, exist_ok=True)
    render_instrument = render_dir / instrument_path.name
    shutil.copyfile(instrument_path, render_instrument)
    run_tool(
        args,
        [
            str(args.player.as_posix()),
            "-f",
            path_for_record(render_instrument, repo_root()),
            "-n",
            f"{note_frequency:.6f}",
            "-v",
            str(args.velocity),
            "-l",
            str(args.length),
        ],
    )
    rendered_wav = render_instrument.with_suffix(render_instrument.suffix + ".wav")
    friendly_wav = render_dir / f"{instrument_path.stem}.wav"
    if rendered_wav.exists():
      shutil.copyfile(rendered_wav, friendly_wav)
      return friendly_wav
    return rendered_wav


def extract_render_feature(args: argparse.Namespace, rendered_wav: Path, feature_path: Path, preview_prefix: Path) -> Path:
    spec = FeatureSpec(
        frequency_bins=frequency_bins(args),
        time_frames=time_frames(args),
        crop_seconds=args.crop_seconds,
        crop_start_seconds=args.crop_start_seconds,
    )
    features = extract_feature_tensor_from_wav(rendered_wav, spec)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    preview_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_feature_tensor(feature_path, features)
    write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_rgb.bmp"), features)
    write_feature_preview_bmp(preview_prefix.with_name(preview_prefix.name + "_logfreq_rgb.bmp"), features)
    return feature_path


def tensor_metrics(source_feature: Path, rendered_feature: Path) -> dict[str, float]:
    source = read_slft(source_feature).data
    rendered = read_slft(rendered_feature).data
    if source.shape != rendered.shape:
      raise ValueError(f"Feature shape mismatch: {source.shape} vs {rendered.shape}")
    diff = source - rendered
    metrics = {
        "feature_mae": float(np.mean(np.abs(diff))),
        "feature_mse": float(np.mean(np.square(diff))),
        "feature_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "feature_max_abs": float(np.max(np.abs(diff))),
    }
    for channel in range(source.shape[0]):
      channel_diff = diff[channel]
      metrics[f"channel_{channel}_mae"] = float(np.mean(np.abs(channel_diff)))
      metrics[f"channel_{channel}_mse"] = float(np.mean(np.square(channel_diff)))
    return metrics


def read_mono_wave(path: Path, sample_count: int) -> np.ndarray | None:
    try:
      with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(sample_count)
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
    except wave.Error:
      return None
    if sample_width != 2:
      return None
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if channels > 1:
      samples = samples.reshape((-1, channels)).mean(axis=1)
    if samples.size < sample_count:
      samples = np.pad(samples, (0, sample_count - samples.size))
    else:
      samples = samples[:sample_count]
    return samples / 32768.0


def waveform_metrics(source_wav: Path, rendered_wav: Path, sample_rate: int, seconds: float) -> dict[str, float]:
    sample_count = int(sample_rate * seconds)
    source = read_mono_wave(source_wav, sample_count)
    rendered = read_mono_wave(rendered_wav, sample_count)
    if source is None or rendered is None:
      return {}
    diff = source - rendered
    return {
        "waveform_mae": float(np.mean(np.abs(diff))),
        "waveform_mse": float(np.mean(np.square(diff))),
        "source_rms": float(np.sqrt(np.mean(np.square(source)))),
        "rendered_rms": float(np.sqrt(np.mean(np.square(rendered)))),
    }


def write_summary_row(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with summary_path.open("a", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
      if not file_exists:
        writer.writeheader()
      writer.writerow(row)


def to_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
      return str(value)
    if isinstance(value, list):
      return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
      return [to_json_safe(item) for item in value]
    if isinstance(value, dict):
      return {str(key): to_json_safe(item) for key, item in value.items()}
    return value


def write_ab_artifacts(output_dir: Path, item_name: str, input_wav: Path, rendered_wav: Path, row: dict[str, Any]) -> None:
    ab_dir = output_dir / "ab_listen"
    mel_dir = ab_dir / "mel"
    ab_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    original_out = ab_dir / f"{item_name}_original.wav"
    predicted_out = ab_dir / f"{item_name}_predicted.wav"
    shutil.copyfile(input_wav, original_out)
    shutil.copyfile(rendered_wav, predicted_out)

    original_mel = mel_dir / f"{item_name}_original_mel.png"
    predicted_mel = mel_dir / f"{item_name}_predicted_mel.png"
    combined_mel = mel_dir / f"{item_name}_ab_mel.png"
    write_mel_preview(original_out, original_mel)
    write_mel_preview(predicted_out, predicted_mel)
    write_ab_mel_preview(original_out, predicted_out, combined_mel)

    manifest_path = ab_dir / "ab_manifest.csv"
    manifest_row = {
        "name": item_name,
        "original_wav": path_for_record(original_out, repo_root()),
        "predicted_wav": path_for_record(predicted_out, repo_root()),
        "original_mel": path_for_record(original_mel, repo_root()),
        "predicted_mel": path_for_record(predicted_mel, repo_root()),
        "ab_mel": path_for_record(combined_mel, repo_root()),
        "feature_mae": row["feature_mae"],
        "feature_rmse": row["feature_rmse"],
        "note_frequency": row["note_frequency"],
        "note_source": row["note_source"],
        "predicted_note_frequency": row["predicted_note_frequency"],
    }
    write_summary_row(manifest_path, manifest_row)


def evaluate_item(args: argparse.Namespace, model: SoundLearnerNet, device: torch.device, item: EvaluationItem) -> dict[str, Any]:
    item_dir = args.output_dir / item.name
    source_feature = ensure_source_feature(args, item, item_dir / "features" / f"{item.name}_source.slft", item_dir / "previews" / f"{item.name}_source")
    prediction_path = item_dir / "predictions" / f"{item.name}_predicted.data"
    row_count, predicted_note_frequency = predict_instrument(model, device, source_feature, prediction_path, args.activity_threshold, args.write_all_slots)

    if args.note_frequency is not None:
      note_frequency = args.note_frequency
      note_source = "override"
    elif item.note_frequency is not None:
      note_frequency = item.note_frequency
      note_source = "manifest"
    elif predicted_note_frequency is not None:
      note_frequency = predicted_note_frequency
      note_source = "predicted"
    else:
      note_frequency = infer_note_frequency(item.name)
      note_source = "filename"
    rendered_wav = render_prediction(args, prediction_path, note_frequency, item_dir / "renders")
    rendered_feature = extract_render_feature(
        args,
        rendered_wav,
        item_dir / "features" / f"{item.name}_rendered.slft",
        item_dir / "previews" / f"{item.name}_rendered",
    )

    metrics = tensor_metrics(source_feature, rendered_feature)
    metrics.update(waveform_metrics(item.input_wav, rendered_wav, sample_rate=44100, seconds=args.crop_seconds))
    row = {
        "name": item.name,
        "input_wav": path_for_record(item.input_wav, repo_root()),
        "source_feature": path_for_record(source_feature, repo_root()),
        "prediction": path_for_record(prediction_path, repo_root()),
        "rendered_wav": path_for_record(rendered_wav, repo_root()),
        "rendered_feature": path_for_record(rendered_feature, repo_root()),
        "note_frequency": note_frequency,
        "note_source": note_source,
        "predicted_note_frequency": predicted_note_frequency,
        "predicted_rows": row_count,
        **metrics,
    }
    (item_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    mel_dir = item_dir / "mel"
    write_mel_preview(item.input_wav, mel_dir / f"{item.name}_source_mel.png")
    write_mel_preview(rendered_wav, mel_dir / f"{item.name}_rendered_mel.png")
    write_ab_mel_preview(item.input_wav, rendered_wav, mel_dir / f"{item.name}_ab_mel.png")
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, device = load_model(args.checkpoint, args.device)
    items = discover_items(args)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                **to_json_safe(vars(args)),
                "device": str(device),
                "model_config": asdict(model.config),
                "item_count": len(items),
            },
            indent=2,
        )
    )

    summary_path = args.output_dir / "summary.csv"
    if summary_path.exists():
      summary_path.unlink()
    for index, item in enumerate(items, start=1):
      print(f"[{index}/{len(items)}] {item.name}")
      row = evaluate_item(args, model, device, item)
      write_summary_row(summary_path, row)
      write_ab_artifacts(args.output_dir, item.name, item.input_wav, repo_root() / row["rendered_wav"], row)
      print(
          f"  feature_mae={row['feature_mae']:.6f} feature_rmse={row['feature_rmse']:.6f} "
          f"rows={row['predicted_rows']} note={row['note_frequency']:.3f} source={row['note_source']}"
      )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch

from .audio_features import (
    FeatureSpec,
    estimate_fundamental_frequency_from_wav,
    extract_feature_tensor_from_wav,
    write_feature_preview_bmp,
    write_feature_tensor,
)
from .audio_preview import write_ab_mel_preview, write_mel_preview
from .dataset import TARGET_CHANNELS, discover_examples, read_oscillator_csv
from .differentiable_audio import denormalize_log_frequency
from .evaluate import path_for_record
from .model import ModelConfig, SoundLearnerNet
from .slft import read_slft


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "analysis_frontend"
RUN_ROOT = APP_ROOT / "runs"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}" if drive else resolved.as_posix()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return slug or "sample"


def latest_checkpoint() -> Path | None:
    candidates = sorted((REPO_ROOT / "runs").glob("**/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def discover_checkpoints() -> list[Path]:
    return sorted((REPO_ROOT / "runs").glob("**/best.pt"), key=lambda path: str(path).lower())


def discover_dataset_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in sorted((REPO_ROOT / "datasets").glob("**")):
      if candidate.name.startswith("_"):
        continue
      if candidate.is_dir() and ((candidate / "metadata").exists() or (candidate / "features").exists()):
        try:
          if discover_examples(candidate):
            roots.append(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
          pass
    return roots


def rel(path: Path) -> str:
    try:
      return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
      return path.resolve().as_posix()


def resolve_user_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
      return path
    return REPO_ROOT / path


def load_model(checkpoint_path: Path, device_name: str) -> tuple[SoundLearnerNet, torch.device]:
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoundLearnerNet(ModelConfig(**checkpoint["model_config"])).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys:
      print(f"Checkpoint is missing model keys: {', '.join(incompatible.missing_keys)}")
    model.eval()
    return model, device


def predict_instrument(
    model: SoundLearnerNet,
    device: torch.device,
    feature_path: Path,
    output_path: Path,
    activity_threshold: float,
    write_all_slots: bool,
) -> tuple[int, float | None, float, list[dict[str, float | int]]]:
    slft = read_slft(feature_path)
    features = torch.from_numpy(slft.data).unsqueeze(0).to(device)
    with torch.no_grad():
      prediction = model(features)
      activity = torch.sigmoid(prediction["activity_logits"])[0].detach().cpu()
      parameters = prediction["parameters"][0].detach().cpu()
      predicted_note = None
      if "f0_normalized" in prediction:
        predicted_note = float(denormalize_log_frequency(prediction["f0_normalized"])[0].detach().cpu())

    rows: list[str] = []
    active_probabilities: list[float] = []
    oscillator_summaries: list[dict[str, float | int]] = []
    for slot_index in range(model.config.max_oscillators):
      probability = float(activity[slot_index].item())
      if not write_all_slots and probability < activity_threshold:
        continue
      values = parameters[slot_index].tolist()
      values[-1] = 1.0 if values[-1] >= 0.5 else 0.0
      rows.append(",".join(f"{value:.6f}" for value in values))
      active_probabilities.append(probability)
      oscillator_summaries.append(
          {
              "slot": slot_index,
              "activity": probability,
              "amplitude": float(values[0]),
              "frequency": float(values[1]),
              "frequency_factor": decoded_frequency_factor(float(values[1]), bool(values[6] >= 0.5)),
              "phase": float(values[2]),
              "amp_decay": float(values[3]),
              "amp_attack": float(values[4]),
              "freq_decay": float(values[5]),
              "coupled": int(values[6] >= 0.5),
          }
      )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    mean_active_probability = float(np.mean(active_probabilities)) if active_probabilities else 0.0
    return len(rows), predicted_note, mean_active_probability, oscillator_summaries


def decoded_frequency_factor(normalized_factor: float, is_coupled: bool) -> float:
    anchors = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    detune_ratio = 0.025 if is_coupled else 0.05
    clamped = min(max(normalized_factor, 0.0), 1.0)
    scaled = clamped * len(anchors)
    anchor_index = min(int(scaled), len(anchors) - 1)
    local = min(max(scaled - anchor_index, 0.0), 1.0)
    detune = 1.0 + ((local - 0.5) * 2.0 * detune_ratio)
    return float(anchors[anchor_index] * detune)


def render_prediction(player_path: Path, instrument_path: Path, note_frequency: float, velocity: int, length: int) -> Path:
    player_command = to_wsl_path(player_path) if player_path.is_absolute() else str(player_path.as_posix())
    command = [
        player_command,
        "-f",
        path_for_record(instrument_path, REPO_ROOT),
        "-n",
        f"{note_frequency:.6f}",
        "-v",
        str(velocity),
        "-l",
        str(length),
    ]
    quoted = " ".join(shlex.quote(item) for item in command)
    bash_command = f"cd {shlex.quote(to_wsl_path(REPO_ROOT))} && {quoted}"
    completed = subprocess.run(["wsl", "bash", "-lc", bash_command], text=True, capture_output=True)
    if completed.returncode != 0:
      raise RuntimeError(f"player failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    rendered = instrument_path.with_suffix(instrument_path.suffix + ".wav")
    if not rendered.exists():
      raise FileNotFoundError(f"Player did not produce {rendered}")
    return rendered


def tensor_metrics(source_feature: Path, rendered_feature: Path) -> dict[str, float]:
    source = read_slft(source_feature).data
    rendered = read_slft(rendered_feature).data
    diff = source - rendered
    return {
        "feature_mae": float(np.mean(np.abs(diff))),
        "feature_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "feature_max_abs": float(np.max(np.abs(diff))),
        "source_mean": float(np.mean(source)),
        "rendered_mean": float(np.mean(rendered)),
    }


def cents_error(predicted_hz: float | None, reference_hz: float | None) -> float | None:
    if predicted_hz is None or reference_hz is None or predicted_hz <= 0.0 or reference_hz <= 0.0:
      return None
    return float(1200.0 * math.log2(predicted_hz / reference_hz))


def flatten_oscillator_file(path: Path, max_oscillators: int) -> np.ndarray:
    target, _ = read_oscillator_csv(path, max_oscillators)
    return target.astype(np.float32).reshape(-1)


def pca_projection(dataset_root: Path | None, prediction_path: Path, max_oscillators: int, limit: int = 240) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    if dataset_root is not None and dataset_root.exists():
      examples = discover_examples(dataset_root)
      paths = [example.target_path for example in examples]
      if len(paths) > limit:
        indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[index] for index in indices.tolist()]
      if paths:
        matrix = np.stack([flatten_oscillator_file(path, max_oscillators) for path in paths], axis=0)
        groups.append({"name": "reference", "label": "Reference Dataset", "color": "#2f7d32", "paths": paths, "matrix": matrix})

    prediction_matrix = np.stack([flatten_oscillator_file(prediction_path, max_oscillators)], axis=0)
    groups.append({"name": "prediction", "label": "Current Prediction", "color": "#c2410c", "paths": [prediction_path], "matrix": prediction_matrix})

    combined = np.concatenate([group["matrix"] for group in groups], axis=0)
    if combined.shape[0] < 2:
      return {"points": [], "explained": [0.0, 0.0]}
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    energy = singular_values ** 2
    explained = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)

    points: list[dict[str, Any]] = []
    offset = 0
    for group in groups:
      for path_index, path in enumerate(group["paths"]):
        coordinate = coords[offset + path_index]
        points.append(
            {
                "group": group["name"],
                "label": group["label"],
                "color": group["color"],
                "path": rel(path),
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
            }
        )
      offset += len(group["paths"])

    return {
        "points": points,
        "explained": [float(explained[0]) if explained.size > 0 else 0.0, float(explained[1]) if explained.size > 1 else 0.0],
    }


def file_url(path: Path) -> str:
    return f"/artifact?path={rel(path)}"


def escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def escape_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def config_payload() -> dict[str, Any]:
    checkpoints = discover_checkpoints()
    default_checkpoint = latest_checkpoint()
    datasets = discover_dataset_roots()
    return {
        "checkpoints": [
            {"path": rel(path), "label": rel(path), "default": default_checkpoint is not None and path == default_checkpoint}
            for path in checkpoints
        ],
        "datasets": [{"path": rel(path), "label": rel(path)} for path in datasets],
    }


def option_html(items: list[dict[str, Any]], include_empty: bool = False) -> str:
    lines: list[str] = []
    if include_empty:
      lines.append('<option value="">None</option>')
    for item in items:
      selected = " selected" if item.get("default") else ""
      value = escape_attr(str(item["path"]))
      label = escape_text(str(item["label"]))
      lines.append(f'<option value="{value}"{selected}>{label}</option>')
    return "\n".join(lines)


def render_index_html() -> str:
    config = config_payload()
    default_checkpoint = next((item["path"] for item in config["checkpoints"] if item.get("default")), "")
    default_dataset = config["datasets"][0]["path"] if config["datasets"] else ""
    return (
        HTML.replace("__CHECKPOINT_OPTIONS__", option_html(config["checkpoints"]))
        .replace("__DATASET_OPTIONS__", option_html(config["datasets"], include_empty=True))
        .replace("__DEFAULT_CHECKPOINT__", escape_attr(str(default_checkpoint)))
        .replace("__DEFAULT_DATASET__", escape_attr(str(default_dataset)))
        .replace("__CONFIG_COUNTS__", f"{len(config['checkpoints'])} checkpoint(s), {len(config['datasets'])} reference dataset(s)")
        .replace("__CONFIG_JSON__", json.dumps(config))
    )


def analyze_sample(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
    if "wav" not in files:
      raise ValueError("Upload a WAV file.")
    original_name, wav_bytes = files["wav"]
    if len(wav_bytes) > MAX_UPLOAD_BYTES:
      raise ValueError("WAV upload is too large.")

    checkpoint = resolve_user_path(fields.get("checkpoint", ""))
    if not checkpoint.exists():
      raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    dataset_text = fields.get("dataset_root", "").strip()
    dataset_root = resolve_user_path(dataset_text) if dataset_text else None
    if dataset_root is not None and not dataset_root.exists():
      dataset_root = None

    freq_bins = int(fields.get("freq_bins", "512"))
    time_frames = int(fields.get("time_frames", "256"))
    crop_seconds = float(fields.get("crop_seconds", "5.0"))
    activity_threshold = float(fields.get("activity_threshold", "0.5"))
    velocity = int(fields.get("velocity", "2"))
    length = int(float(fields.get("length", "5")))
    device_name = fields.get("device", "cpu")
    render_note_source = fields.get("render_note_source", "estimated")
    manual_note = float(fields.get("manual_note_frequency", "440") or "440")
    player = resolve_user_path(fields.get("player", "build/player/player"))

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_slug(Path(original_name).stem)}_{random.randrange(1000, 9999)}"
    run_dir = RUN_ROOT / run_id
    source_wav = run_dir / "source" / safe_slug(original_name)
    source_feature = run_dir / "features" / "source.slft"
    rendered_feature = run_dir / "features" / "prediction.slft"
    prediction_path = run_dir / "prediction" / "instrument.data"
    source_preview = run_dir / "images" / "source_feature.bmp"
    rendered_preview = run_dir / "images" / "prediction_feature.bmp"
    source_mel = run_dir / "images" / "source_mel.png"
    rendered_mel = run_dir / "images" / "prediction_mel.png"
    ab_mel = run_dir / "images" / "ab_mel.png"

    source_wav.parent.mkdir(parents=True, exist_ok=True)
    source_wav.write_bytes(wav_bytes)

    spec = FeatureSpec(frequency_bins=freq_bins, time_frames=time_frames, crop_seconds=crop_seconds)
    source_tensor = extract_feature_tensor_from_wav(source_wav, spec)
    write_feature_tensor(source_feature, source_tensor)
    write_feature_preview_bmp(source_preview, source_tensor, width=900, height=320)
    write_mel_preview(source_wav, source_mel)
    estimated_note = estimate_fundamental_frequency_from_wav(source_wav)

    model, device = load_model(checkpoint, device_name)
    row_count, predicted_note, mean_activity, oscillators = predict_instrument(model, device, source_feature, prediction_path, activity_threshold, False)
    if render_note_source == "predicted":
      note_frequency = predicted_note if predicted_note is not None else estimated_note if estimated_note is not None else manual_note
    elif render_note_source == "manual":
      note_frequency = manual_note
    else:
      note_frequency = estimated_note if estimated_note is not None else predicted_note if predicted_note is not None else manual_note

    rendered_wav_raw = render_prediction(player, prediction_path, note_frequency, velocity, length)
    rendered_wav = run_dir / "prediction" / "prediction.wav"
    shutil.copyfile(rendered_wav_raw, rendered_wav)
    rendered_tensor = extract_feature_tensor_from_wav(rendered_wav, spec)
    write_feature_tensor(rendered_feature, rendered_tensor)
    write_feature_preview_bmp(rendered_preview, rendered_tensor, width=900, height=320)
    write_mel_preview(rendered_wav, rendered_mel)
    write_ab_mel_preview(source_wav, rendered_wav, ab_mel)

    metrics = tensor_metrics(source_feature, rendered_feature)
    pca = pca_projection(dataset_root, prediction_path, model.config.max_oscillators)
    response = {
        "run_id": run_id,
        "checkpoint": rel(checkpoint),
        "dataset_root": rel(dataset_root) if dataset_root is not None else None,
        "predicted_note_frequency": predicted_note,
        "estimated_note_frequency": estimated_note,
        "render_note_frequency": note_frequency,
        "render_note_source": render_note_source,
        "f0_error_cents": cents_error(predicted_note, estimated_note),
        "predicted_rows": row_count,
        "mean_active_probability": mean_activity,
        "oscillators": oscillators,
        "metrics": metrics,
        "artifacts": {
            "source_wav": file_url(source_wav),
            "predicted_wav": file_url(rendered_wav),
            "source_feature": file_url(source_preview),
            "predicted_feature": file_url(rendered_preview),
            "source_mel": file_url(source_mel),
            "predicted_mel": file_url(rendered_mel),
            "ab_mel": file_url(ab_mel),
            "prediction_data": file_url(prediction_path),
        },
        "pca": pca,
    }
    (run_dir / "result.json").write_text(json.dumps(response, indent=2))
    return response


def parse_header_options(value: str) -> tuple[str, dict[str, str]]:
    pieces = [piece.strip() for piece in value.split(";")]
    main = pieces[0].lower() if pieces else ""
    options: dict[str, str] = {}
    for piece in pieces[1:]:
      if "=" not in piece:
        continue
      key, raw = piece.split("=", 1)
      options[key.strip().lower()] = raw.strip().strip('"')
    return main, options


def parse_multipart(headers: Any, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    content_type = headers.get("content-type", "")
    _, options = parse_header_options(content_type)
    boundary = options.get("boundary")
    if not boundary:
      raise ValueError("Missing multipart boundary")
    boundary_bytes = ("--" + boundary).encode()
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in body.split(boundary_bytes):
      part = part.strip(b"\r\n")
      if not part or part == b"--":
        continue
      header_blob, _, payload = part.partition(b"\r\n\r\n")
      if not payload:
        continue
      header_lines = header_blob.decode("utf-8", errors="replace").split("\r\n")
      dispositions = [line for line in header_lines if line.lower().startswith("content-disposition:")]
      if not dispositions:
        continue
      _, disposition_options = parse_header_options(dispositions[0].split(":", 1)[1].strip())
      name = disposition_options.get("name")
      filename = disposition_options.get("filename")
      payload = payload.removesuffix(b"\r\n")
      if name is None:
        continue
      if filename:
        files[name] = (filename, payload)
      else:
        fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SoundLearner Analysis Workbench</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f2efe6;
      --muted: #b9b1a3;
      --line: #3f4039;
      --paper: #171815;
      --panel: #23241f;
      --panel-2: #2d2f29;
      --green: #70b36a;
      --rust: #c75f37;
      --gold: #d0a53a;
      --cyan: #6bb7a8;
      --button: #41765f;
      --track: #11120f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--paper);
      color: var(--ink);
      letter-spacing: 0;
    }
    main { max-width: 1380px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 6px; font-size: 28px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 0 0 10px; font-size: 14px; color: var(--muted); text-transform: uppercase; }
    p { color: var(--muted); line-height: 1.45; }
    section, .module {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }
    .topology { display: grid; grid-template-columns: 1fr 360px; gap: 14px; align-items: stretch; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .rack { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr); gap: 14px; }
    .transport { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 12px; }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #151612;
      color: var(--ink);
    }
    input[type="range"] { accent-color: var(--gold); padding: 0; height: 24px; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 800;
      color: #0e120d;
      background: var(--gold);
      cursor: pointer;
      min-height: 42px;
    }
    button.secondary { background: var(--panel-2); color: var(--ink); border: 1px solid var(--line); }
    button:disabled { opacity: 0.55; cursor: wait; }
    .drop {
      border: 2px dashed #77786f;
      border-radius: 8px;
      min-height: 144px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      background: #1d1f1a;
    }
    .drop.active { border-color: var(--green); background: #20291f; }
    .drop strong { display: block; font-size: 18px; color: var(--ink); margin-bottom: 8px; }
    .picker { display: grid; grid-template-columns: 42px minmax(0, 1fr) 42px; gap: 6px; align-items: end; }
    .picker input { font-size: 12px; }
    .control-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .slider-box {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .slider-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .slider-head output { color: var(--ink); font-variant-numeric: tabular-nums; }
    .segments { display: flex; gap: 6px; padding: 4px; background: #151612; border: 1px solid var(--line); border-radius: 8px; }
    .segments button { flex: 1; min-height: 34px; padding: 7px 10px; background: transparent; color: var(--muted); border: 0; }
    .segments button.active { background: var(--green); color: #0e120d; }
    .status { color: var(--muted); min-height: 24px; }
    .metric-row { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 76px;
      background: #1d1e1a;
    }
    .metric b { display: block; font-size: 18px; margin-top: 8px; color: var(--gold); }
    audio { width: 100%; filter: sepia(0.08); }
    img { max-width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); background: #111; }
    canvas { width: 100%; height: 460px; border: 1px solid var(--line); border-radius: 8px; background: #11120f; }
    .hidden { display: none; }
    .small { font-size: 13px; color: var(--muted); }
    .osc-rack { display: grid; gap: 8px; max-height: 560px; overflow: auto; padding-right: 4px; }
    .osc-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #1d1e1a;
      padding: 10px;
    }
    .osc-title { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .osc-title b { color: var(--gold); }
    .osc-pill { border: 1px solid var(--line); border-radius: 6px; padding: 3px 6px; color: var(--muted); font-size: 11px; }
    .osc-controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
    .knob { display: grid; justify-items: center; gap: 5px; color: var(--muted); font-size: 11px; }
    .dial {
      width: 52px;
      height: 52px;
      border-radius: 50%;
      border: 1px solid #55564e;
      background: conic-gradient(var(--gold) calc(var(--value) * 1turn), #11120f 0);
      position: relative;
    }
    .dial::after { content: ""; position: absolute; inset: 9px; border-radius: 50%; background: var(--panel-2); border: 1px solid var(--line); }
    .knob span { color: var(--ink); font-variant-numeric: tabular-nums; }
    .bar { height: 8px; border-radius: 4px; background: #11120f; overflow: hidden; border: 1px solid var(--line); }
    .bar div { height: 100%; background: var(--cyan); }
    .osc-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    @media (max-width: 980px) {
      main { padding: 14px; }
      .topology, .rack, .grid, .pair, .control-grid, .metric-row { grid-template-columns: 1fr; }
      .osc-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>SoundLearner Analysis Workbench</h1>
    <p>Drop in a WAV, predict an oscillator model, render it through the C++ player, and inspect the A/B audio, mel previews, feature previews, and PCA position.</p>
  </header>

  <section>
    <h2>Patch Input</h2>
    <div class="topology">
      <div id="drop" class="drop">
        <div>
          <strong>Drop a WAV here</strong>
          <span id="fileLabel">or choose one below</span>
          <input id="wav" type="file" accept=".wav,audio/wav">
        </div>
      </div>
      <div class="module">
        <h3>Sources</h3>
        <label>Checkpoint</label>
        <div class="picker">
          <button class="secondary" id="checkpointPrev" type="button">Prev</button>
          <input id="checkpoint" list="checkpointList" value="__DEFAULT_CHECKPOINT__" spellcheck="false">
          <button class="secondary" id="checkpointNext" type="button">Next</button>
        </div>
        <datalist id="checkpointList">__CHECKPOINT_OPTIONS__</datalist>
        <label style="margin-top:10px">PCA Reference</label>
        <div class="picker">
          <button class="secondary" id="datasetPrev" type="button">Prev</button>
          <input id="dataset" list="datasetList" value="__DEFAULT_DATASET__" spellcheck="false">
          <button class="secondary" id="datasetNext" type="button">Next</button>
        </div>
        <datalist id="datasetList"><option value="">None</option>__DATASET_OPTIONS__</datalist>
      </div>
    </div>

    <div class="rack">
      <div class="module">
        <h3>Analysis Controls</h3>
        <div class="control-grid">
          <div class="slider-box"><div class="slider-head"><span>Frequency Bins</span><output id="freqBinsOut">512</output></div><input id="freqBins" type="range" min="64" max="2048" step="64" value="512"></div>
          <div class="slider-box"><div class="slider-head"><span>Time Frames</span><output id="timeFramesOut">256</output></div><input id="timeFrames" type="range" min="32" max="1024" step="32" value="256"></div>
          <div class="slider-box"><div class="slider-head"><span>Crop Seconds</span><output id="cropSecondsOut">5.0</output></div><input id="cropSeconds" type="range" min="0.5" max="8" step="0.5" value="5"></div>
          <div class="slider-box"><div class="slider-head"><span>Activity Threshold</span><output id="activityThresholdOut">0.25</output></div><input id="activityThreshold" type="range" min="0" max="1" step="0.01" value="0.25"></div>
          <div class="slider-box"><div class="slider-head"><span>Render Velocity</span><output id="velocityOut">2</output></div><input id="velocity" type="range" min="1" max="100" step="1" value="2"></div>
          <div class="slider-box"><div class="slider-head"><span>Manual Note Hz</span><output id="manualNoteFrequencyOut">440</output></div><input id="manualNoteFrequency" type="range" min="20" max="2000" step="1" value="440"></div>
        </div>
      </div>
      <div class="module">
        <h3>Performance</h3>
        <label>Render Note</label>
        <div class="segments" data-target="renderNoteSource">
          <button type="button" data-value="estimated" class="active">audio pitch</button>
          <button type="button" data-value="predicted">model f0</button>
          <button type="button" data-value="manual">manual</button>
        </div>
        <input id="renderNoteSource" type="hidden" value="estimated">
        <label style="margin-top:12px">Device</label>
        <div class="segments" data-target="device">
          <button type="button" data-value="cpu" class="active">cpu</button>
          <button type="button" data-value="cuda">cuda</button>
        </div>
        <input id="device" type="hidden" value="cpu">
        <div class="transport" style="margin-top:16px">
          <button id="analyze">Analyze</button>
          <span id="status" class="status">Ready. Found __CONFIG_COUNTS__.</span>
        </div>
      </div>
    </div>
  </section>

  <section id="results" class="hidden">
    <h2>Result</h2>
    <div class="metric-row">
      <div class="metric">Model f0<b id="modelNote">-</b></div>
      <div class="metric">Audio pitch estimate<b id="estimatedNote">-</b></div>
      <div class="metric">Render f0<b id="renderNote">-</b></div>
      <div class="metric">Oscillators<b id="rows">-</b></div>
      <div class="metric">Feature MAE<b id="mae">-</b></div>
      <div class="metric">F0 Error<b id="f0Error">-</b></div>
    </div>
  </section>

  <section id="audioSection" class="hidden">
    <h2>A/B Listening</h2>
    <div class="grid">
      <label>Original<audio id="sourceAudio" controls></audio></label>
      <label>Predicted<audio id="predAudio" controls></audio></label>
    </div>
  </section>

  <section id="oscillatorSection" class="hidden">
    <h2>Predicted Oscillator Rack</h2>
    <p class="small">Read-only for now: these controls mirror the generated .data rows that the C++ player rendered.</p>
    <div id="oscillatorRack" class="osc-rack"></div>
  </section>

  <section id="melSection" class="hidden">
    <h2>Mel Spectrograms</h2>
    <img id="abMel" alt="A/B mel spectrogram comparison">
    <div class="pair" style="margin-top:12px">
      <img id="sourceMel" alt="Original mel spectrogram">
      <img id="predMel" alt="Predicted mel spectrogram">
    </div>
  </section>

  <section id="featureSection" class="hidden">
    <h2>SLFT Feature Preview</h2>
    <div class="pair">
      <img id="sourceFeature" alt="Original feature preview">
      <img id="predFeature" alt="Predicted feature preview">
    </div>
  </section>

  <section id="pcaSection" class="hidden">
    <h2>2D PCA Space</h2>
    <canvas id="pca" width="1100" height="460"></canvas>
    <p id="pcaNote" class="small"></p>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let selectedFile = null;
let appConfig = __CONFIG_JSON__;

function setStatus(text) { $('status').textContent = text; }
function artifact(url) { return `${url}&t=${Date.now()}`; }
function setOutput(id, value) { const node = $(id + 'Out'); if (node) node.textContent = value; }
function syncSliders() {
  for (const id of ['freqBins', 'timeFrames', 'cropSeconds', 'activityThreshold', 'velocity', 'manualNoteFrequency']) {
    const input = $(id);
    if (!input) continue;
    const format = id === 'cropSeconds' ? Number(input.value).toFixed(1) : id === 'activityThreshold' ? Number(input.value).toFixed(2) : input.value;
    setOutput(id, format);
    input.addEventListener('input', () => {
      const value = id === 'cropSeconds' ? Number(input.value).toFixed(1) : id === 'activityThreshold' ? Number(input.value).toFixed(2) : input.value;
      setOutput(id, value);
    });
  }
}
function bindSegments() {
  for (const group of document.querySelectorAll('.segments')) {
    const target = $(group.dataset.target);
    group.addEventListener('click', event => {
      const button = event.target.closest('button[data-value]');
      if (!button) return;
      for (const item of group.querySelectorAll('button')) item.classList.toggle('active', item === button);
      target.value = button.dataset.value;
    });
  }
}
function bindPicker(inputId, prevId, nextId, items) {
  const input = $(inputId);
  const previous = $(prevId);
  const next = $(nextId);
  const values = items.map(item => item.path);
  const move = direction => {
    if (!values.length) return;
    let index = values.indexOf(input.value);
    if (index < 0) index = direction > 0 ? -1 : 0;
    index = (index + direction + values.length) % values.length;
    input.value = values[index];
  };
  previous.addEventListener('click', () => move(-1));
  next.addEventListener('click', () => move(1));
}
async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    appConfig = await response.json();
  } catch (error) {
    setStatus(`Using embedded config. ${error.message}`);
  }
  const checkpointDefault = appConfig.checkpoints.find(item => item.default) || appConfig.checkpoints[0];
  if (checkpointDefault && !$('checkpoint').value) $('checkpoint').value = checkpointDefault.path;
  if (appConfig.datasets.length && !$('dataset').value) $('dataset').value = appConfig.datasets[0].path;
  bindPicker('checkpoint', 'checkpointPrev', 'checkpointNext', appConfig.checkpoints);
  bindPicker('dataset', 'datasetPrev', 'datasetNext', [{ path: '' }, ...appConfig.datasets]);
  setStatus(`Ready. Found ${appConfig.checkpoints.length} checkpoint(s), ${appConfig.datasets.length} reference dataset(s).`);
}

function drawKnob(label, value, text) {
  const clamped = Math.max(0, Math.min(1, Number(value) || 0));
  return `<div class="knob"><div class="dial" style="--value:${clamped}"></div><small>${label}</small><span>${text ?? clamped.toFixed(2)}</span></div>`;
}
function renderOscillators(oscillators) {
  const rack = $('oscillatorRack');
  if (!oscillators || !oscillators.length) {
    rack.innerHTML = '<p class="small">No active oscillator rows passed the activity threshold.</p>';
    return;
  }
  rack.innerHTML = oscillators.map(osc => `
    <div class="osc-card">
      <div class="osc-title"><b>OSC ${osc.slot}</b><span class="osc-pill">${osc.coupled ? 'coupled' : 'free'} ? ${osc.frequency_factor.toFixed(2)}x</span></div>
      <div class="osc-controls">
        ${drawKnob('amp', osc.amplitude)}
        ${drawKnob('freq', osc.frequency, `${osc.frequency_factor.toFixed(2)}x`)}
        ${drawKnob('phase', osc.phase)}
        ${drawKnob('attack', osc.amp_attack)}
        ${drawKnob('amp decay', osc.amp_decay)}
        ${drawKnob('freq decay', osc.freq_decay)}
        ${drawKnob('activity', osc.activity)}
        ${drawKnob('couple', osc.coupled, osc.coupled ? 'on' : 'off')}
      </div>
      <div class="osc-summary">
        <span>activity ${(osc.activity * 100).toFixed(1)}%</span>
        <span>norm f ${osc.frequency.toFixed(3)}</span>
        <span>amp ${osc.amplitude.toFixed(3)}</span>
      </div>
    </div>
  `).join('');
}

function drawPca(pca) {
  const canvas = $('pca');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const points = pca.points || [];
  ctx.fillStyle = '#11120f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#3f4039';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const x = 70 + i * (canvas.width - 180) / 5;
    const y = 40 + i * (canvas.height - 110) / 5;
    ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x, canvas.height - 70); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(70, y); ctx.lineTo(canvas.width - 110, y); ctx.stroke();
  }
  if (!points.length) return;
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const sx = x => 70 + (x - minX) / (maxX - minX) * (canvas.width - 180);
  const sy = y => canvas.height - 70 - (y - minY) / (maxY - minY) * (canvas.height - 110);
  for (const point of points) {
    ctx.fillStyle = point.color;
    ctx.globalAlpha = point.group === 'prediction' ? 1 : 0.55;
    ctx.beginPath();
    ctx.arc(sx(point.x), sy(point.y), point.group === 'prediction' ? 9 : 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#f2efe6';
  ctx.font = '14px Segoe UI, Arial';
  ctx.fillText('Reference Dataset', canvas.width - 260, 62);
  ctx.fillStyle = '#2f7d32';
  ctx.fillRect(canvas.width - 284, 51, 14, 14);
  ctx.fillStyle = '#f2efe6';
  ctx.fillText('Current Prediction', canvas.width - 260, 88);
  ctx.fillStyle = '#c2410c';
  ctx.fillRect(canvas.width - 284, 77, 14, 14);
  $('pcaNote').textContent = `PC1 ${(pca.explained[0] * 100).toFixed(1)}% variance, PC2 ${(pca.explained[1] * 100).toFixed(1)}% variance.`;
}

async function analyze() {
  if (!selectedFile) {
    const input = $('wav');
    selectedFile = input.files[0];
  }
  if (!selectedFile) {
    setStatus('Choose a WAV first.');
    return;
  }
  const form = new FormData();
  form.append('wav', selectedFile);
  form.append('checkpoint', $('checkpoint').value);
  form.append('dataset_root', $('dataset').value);
  form.append('freq_bins', $('freqBins').value);
  form.append('time_frames', $('timeFrames').value);
  form.append('crop_seconds', $('cropSeconds').value);
  form.append('activity_threshold', $('activityThreshold').value);
  form.append('velocity', $('velocity').value);
  form.append('length', $('cropSeconds').value);
  form.append('device', $('device').value);
  form.append('render_note_source', $('renderNoteSource').value);
  form.append('manual_note_frequency', $('manualNoteFrequency').value);
  form.append('player', 'build/player/player');
  $('analyze').disabled = true;
  setStatus('Analyzing. This can take a moment.');
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed.');
    $('results').classList.remove('hidden');
    $('audioSection').classList.remove('hidden');
    $('oscillatorSection').classList.remove('hidden');
    $('melSection').classList.remove('hidden');
    $('featureSection').classList.remove('hidden');
    $('pcaSection').classList.remove('hidden');
    $('modelNote').textContent = result.predicted_note_frequency ? `${result.predicted_note_frequency.toFixed(2)} Hz` : '-';
    $('estimatedNote').textContent = result.estimated_note_frequency ? `${result.estimated_note_frequency.toFixed(2)} Hz` : '-';
    $('renderNote').textContent = `${result.render_note_frequency.toFixed(2)} Hz`;
    $('rows').textContent = result.predicted_rows;
    $('mae').textContent = result.metrics.feature_mae.toFixed(4);
    $('f0Error').textContent = result.f0_error_cents === null ? '-' : `${result.f0_error_cents.toFixed(0)} cents`;
    $('sourceAudio').src = artifact(result.artifacts.source_wav);
    $('predAudio').src = artifact(result.artifacts.predicted_wav);
    $('abMel').src = artifact(result.artifacts.ab_mel);
    $('sourceMel').src = artifact(result.artifacts.source_mel);
    $('predMel').src = artifact(result.artifacts.predicted_mel);
    $('sourceFeature').src = artifact(result.artifacts.source_feature);
    $('predFeature').src = artifact(result.artifacts.predicted_feature);
    renderOscillators(result.oscillators);
    drawPca(result.pca);
    setStatus(`Done. Run ${result.run_id}. Rendered with ${result.render_note_source} f0.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    $('analyze').disabled = false;
  }
}

$('wav').addEventListener('change', event => {
  selectedFile = event.target.files[0];
  $('fileLabel').textContent = selectedFile ? selectedFile.name : 'or choose one below';
});
$('analyze').addEventListener('click', analyze);
for (const eventName of ['dragenter', 'dragover']) {
  $('drop').addEventListener(eventName, event => { event.preventDefault(); $('drop').classList.add('active'); });
}
for (const eventName of ['dragleave', 'drop']) {
  $('drop').addEventListener(eventName, event => { event.preventDefault(); $('drop').classList.remove('active'); });
}
$('drop').addEventListener('drop', event => {
  selectedFile = event.dataTransfer.files[0];
  $('fileLabel').textContent = selectedFile ? selectedFile.name : 'or choose one below';
});
syncSliders();
bindSegments();
loadConfig().catch(error => setStatus(error.message));
</script>
</body>
</html>
"""


class AnalysisHandler(BaseHTTPRequestHandler):
    server_version = "SoundLearnerAnalysis/0.1"

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
      data = json.dumps(payload).encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def send_text(self, text: str, content_type: str = "text/html") -> None:
      data = text.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", f"{content_type}; charset=utf-8")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    def do_GET(self) -> None:
      parsed = urlparse(self.path)
      if parsed.path == "/":
        self.send_text(render_index_html())
        return
      if parsed.path == "/api/config":
        self.send_json(config_payload())
        return
      if parsed.path == "/artifact":
        query = parse_qs(parsed.query)
        path_values = query.get("path", [])
        if not path_values:
          self.send_error(HTTPStatus.BAD_REQUEST, "Missing artifact path")
          return
        artifact_path = resolve_user_path(unquote(path_values[0])).resolve()
        try:
          artifact_path.relative_to(REPO_ROOT.resolve())
        except ValueError:
          self.send_error(HTTPStatus.FORBIDDEN, "Artifact outside repository")
          return
        if not artifact_path.exists():
          self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
          return
        content_type = "application/octet-stream"
        if artifact_path.suffix.lower() == ".wav":
          content_type = "audio/wav"
        elif artifact_path.suffix.lower() == ".png":
          content_type = "image/png"
        elif artifact_path.suffix.lower() == ".bmp":
          content_type = "image/bmp"
        elif artifact_path.suffix.lower() in {".json", ".data", ".csv"}:
          content_type = "text/plain"
        data = artifact_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return
      self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
      parsed = urlparse(self.path)
      if parsed.path != "/api/analyze":
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        return
      try:
        length = int(self.headers.get("content-length", "0"))
        if length > MAX_UPLOAD_BYTES + 4096:
          raise ValueError("Request is too large")
        body = self.rfile.read(length)
        fields, files = parse_multipart(self.headers, body)
        result = analyze_sample(fields, files)
        self.send_json(result)
      except Exception as exc:  # noqa: BLE001 - local UI should surface the precise failure.
        self.send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
      print(f"[analysis] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SoundLearner local analysis frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), AnalysisHandler)
    print(f"SoundLearner analysis frontend running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

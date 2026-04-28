from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .differentiable_audio import denormalize_log_frequency
from .model import ModelConfig, SoundLearnerNet
from .slft import read_slft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict oscillator CSV rows from an SLFT tensor.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--write-all-slots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = SoundLearnerNet(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys:
      print(f"Checkpoint is missing new model keys; initialized randomly: {', '.join(incompatible.missing_keys)}")
    model.eval()

    slft = read_slft(args.feature)
    features = torch.from_numpy(slft.data).unsqueeze(0).to(device)
    with torch.no_grad():
      prediction = model(features)
      activity = torch.sigmoid(prediction["activity_logits"])[0].cpu()
      parameters = prediction["parameters"][0].cpu()
      predicted_note = None
      if "f0_normalized" in prediction:
        predicted_note = float(denormalize_log_frequency(prediction["f0_normalized"])[0].detach().cpu())

    rows: list[str] = []
    for slot_index in range(config.max_oscillators):
      if not args.write_all_slots and activity[slot_index].item() < args.activity_threshold:
        continue
      values = parameters[slot_index].tolist()
      values[-1] = 1.0 if values[-1] >= 0.5 else 0.0
      rows.append(",".join(f"{value:.6f}" for value in values))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"Wrote {len(rows)} oscillator rows to {args.output}")
    if predicted_note is not None:
      print(f"Predicted f0: {predicted_note:.3f} Hz")


if __name__ == "__main__":
    main()

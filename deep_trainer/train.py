from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import tomllib
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .dataset import SoundLearnerDataset, discover_examples
from .differentiable_audio import RenderLossConfig
from .losses import OscillatorLoss
from .model import ModelConfig, SoundLearnerNet


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
      raise FileNotFoundError(f"Training config file does not exist: {path}")
    if path.suffix.lower() == ".json":
      data = json.loads(path.read_text())
    else:
      data = tomllib.loads(path.read_text())
    if not isinstance(data, dict):
      raise ValueError(f"Training config {path} did not produce a top-level table/object")
    return data


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="Root containing features/, metadata/, and dataN.data files.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"), help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--max-oscillators", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=None, help="Optional strict SLFT frequency/time resolution.")
    parser.add_argument("--freq-bins", type=int, default=None, help="Optional strict SLFT frequency bin count.")
    parser.add_argument("--time-frames", type=int, default=None, help="Optional strict SLFT time frame count.")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--tensorboard", action="store_true", help="Write TensorBoard event logs if tensorboard is installed.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a run from a checkpoint, including optimizer state and epoch.")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize model weights from a checkpoint, but start a fresh run.")
    parser.add_argument("--activity-loss-weight", type=float, default=1.0)
    parser.add_argument("--parameter-loss-weight", type=float, default=10.0)
    parser.add_argument("--activity-positive-weight", type=float, default=0.0, help="Positive class weight for activity BCE. Use 0 for dynamic batch balancing.")
    parser.add_argument("--f0-loss-weight", type=float, default=1.0, help="Weight for predicted fundamental-frequency supervision.")
    parser.add_argument("--crowding-loss-weight", type=float, default=0.0, help="Penalty for predicted oscillators clustering on the same frequency factor.")
    parser.add_argument("--f0-min-frequency", type=float, default=40.0, help="Minimum Hz used for log-frequency f0 normalization.")
    parser.add_argument("--f0-max-frequency", type=float, default=2000.0, help="Maximum Hz used for log-frequency f0 normalization.")
    parser.add_argument("--render-loss-weight", type=float, default=0.0, help="Weight for differentiable rendered-feature reconstruction loss.")
    parser.add_argument("--render-rms-loss-weight", type=float, default=0.0, help="Weight for rendered-vs-source RMS penalty.")
    parser.add_argument("--render-loss-sample-rate", type=int, default=11025, help="Sample rate for differentiable surrogate render.")
    parser.add_argument("--render-loss-seconds", type=float, default=1.0, help="Seconds of surrogate audio to render for feature loss.")
    parser.add_argument("--render-loss-fft-size-multiplier", type=int, default=4, help="FFT multiplier for differentiable rendered-feature extraction.")


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Optional TOML or JSON training config file.")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Train the SoundLearner oscillator inverse model.")
    parser.add_argument("--config", type=Path, default=None, help="Optional TOML or JSON training config file.")
    add_train_arguments(parser)
    if pre_args.config is not None:
      config_defaults = load_config_file(pre_args.config)
      parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
      raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    return args


def split_dataset(dataset: SoundLearnerDataset, validation_split: float, seed: int) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    validation_size = max(1, int(len(dataset) * validation_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - validation_size
    if validation_size == 0:
      return dataset, dataset
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, validation_size], generator=generator)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
      moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def run_epoch(
    model: SoundLearnerNet,
    loader: DataLoader,
    criterion: OscillatorLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {
        "loss": 0.0,
        "activity_loss": 0.0,
        "parameter_loss": 0.0,
        "f0_loss": 0.0,
        "f0_cents_mae": 0.0,
        "crowding_loss": 0.0,
        "render_feature_loss": 0.0,
        "render_rms_loss": 0.0,
    }
    count = 0

    for batch in loader:
      batch = move_batch(batch, device)
      with torch.set_grad_enabled(is_training):
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
          predictions = model(batch["features"])
          loss, metrics = criterion(
              predictions,
              batch["target"],
              batch["mask"],
              source_features=batch["features"],
              note_frequency=batch["note_frequency"],
              velocity=batch["velocity"],
              source_rms=batch["source_rms"],
          )

        if is_training:
          optimizer.zero_grad(set_to_none=True)
          if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
          else:
            loss.backward()
            optimizer.step()

      batch_size = int(batch["features"].shape[0])
      count += batch_size
      for key in totals:
        totals[key] += metrics[key] * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: SoundLearnerNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "args": serializable_args,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def validate_checkpoint_config(checkpoint: dict[str, Any], expected: ModelConfig) -> None:
    checkpoint_config = checkpoint.get("model_config")
    if checkpoint_config is None:
      raise ValueError("Checkpoint is missing model_config")
    loaded = ModelConfig(**checkpoint_config)
    if loaded != expected:
      raise ValueError(
          "Checkpoint model_config does not match requested training config: "
          f"checkpoint={loaded}, requested={expected}"
      )


def load_model_state(model: SoundLearnerNet, checkpoint: dict[str, Any]) -> None:
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys:
      print(f"Checkpoint is missing new model keys; initialized randomly: {', '.join(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
      print(f"Checkpoint has unused model keys: {', '.join(incompatible.unexpected_keys)}")


def write_metrics_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
      if not file_exists:
        writer.writeheader()
      writer.writerow(row)


def log_tensorboard(writer: Any, epoch: int, train_metrics: dict[str, float], validation_metrics: dict[str, float], learning_rate: float) -> None:
    if writer is None:
      return
    writer.add_scalar("train/loss", train_metrics["loss"], epoch)
    writer.add_scalar("train/activity_loss", train_metrics["activity_loss"], epoch)
    writer.add_scalar("train/parameter_loss", train_metrics["parameter_loss"], epoch)
    writer.add_scalar("train/f0_loss", train_metrics["f0_loss"], epoch)
    writer.add_scalar("train/f0_cents_mae", train_metrics["f0_cents_mae"], epoch)
    writer.add_scalar("train/crowding_loss", train_metrics["crowding_loss"], epoch)
    writer.add_scalar("val/loss", validation_metrics["loss"], epoch)
    writer.add_scalar("val/activity_loss", validation_metrics["activity_loss"], epoch)
    writer.add_scalar("val/parameter_loss", validation_metrics["parameter_loss"], epoch)
    writer.add_scalar("val/f0_loss", validation_metrics["f0_loss"], epoch)
    writer.add_scalar("val/f0_cents_mae", validation_metrics["f0_cents_mae"], epoch)
    writer.add_scalar("val/crowding_loss", validation_metrics["crowding_loss"], epoch)
    writer.add_scalar("train/render_feature_loss", train_metrics["render_feature_loss"], epoch)
    writer.add_scalar("val/render_feature_loss", validation_metrics["render_feature_loss"], epoch)
    writer.add_scalar("train/render_rms_loss", train_metrics["render_rms_loss"], epoch)
    writer.add_scalar("val/render_rms_loss", validation_metrics["render_rms_loss"], epoch)
    writer.add_scalar("optim/learning_rate", learning_rate, epoch)
    writer.flush()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = discover_examples(args.dataset_root)
    dataset = SoundLearnerDataset(
        examples,
        max_oscillators=args.max_oscillators,
        expected_resolution=args.resolution,
        expected_frequency_bins=args.freq_bins,
        expected_time_frames=args.time_frames,
    )
    train_dataset, validation_dataset = split_dataset(dataset, args.validation_split, args.seed)

    first_sample = dataset[0]
    input_channels = int(first_sample["features"].shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(input_channels=input_channels, max_oscillators=args.max_oscillators, width=args.width, dropout=args.dropout)
    model = SoundLearnerNet(config).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    render_config = None
    if args.render_loss_weight > 0.0:
      expected_frequency_bins = args.freq_bins if args.freq_bins is not None else args.resolution
      expected_time_frames = args.time_frames if args.time_frames is not None else args.resolution
      if expected_frequency_bins is None or expected_time_frames is None:
        expected_frequency_bins = int(first_sample["features"].shape[1])
        expected_time_frames = int(first_sample["features"].shape[2])
      render_config = RenderLossConfig(
          frequency_bins=expected_frequency_bins,
          time_frames=expected_time_frames,
          sample_rate=args.render_loss_sample_rate,
          render_seconds=args.render_loss_seconds,
          fft_size_multiplier=args.render_loss_fft_size_multiplier,
      )
    criterion = OscillatorLoss(
        activity_weight=args.activity_loss_weight,
        parameter_weight=args.parameter_loss_weight,
        activity_positive_weight=args.activity_positive_weight,
        f0_weight=args.f0_loss_weight,
        crowding_weight=args.crowding_loss_weight,
        render_weight=args.render_loss_weight,
        render_rms_weight=args.render_rms_loss_weight,
        f0_min_frequency=args.f0_min_frequency,
        f0_max_frequency=args.f0_max_frequency,
        render_config=render_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    best_validation_loss = float("inf")
    start_epoch = 1
    metrics_path = args.output_dir / "metrics.csv"
    writer = None
    if args.tensorboard:
      if SummaryWriter is None:
        print("TensorBoard requested, but tensorboard is not installed. CSV metrics will still be written.")
      else:
        writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))

    if args.resume is not None:
      checkpoint = load_checkpoint(args.resume, device)
      validate_checkpoint_config(checkpoint, config)
      load_model_state(model, checkpoint)
      try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
      except ValueError as exc:
        print(f"Checkpoint optimizer state is incompatible with the current model; starting optimizer fresh: {exc}")
      start_epoch = int(checkpoint["epoch"]) + 1
      best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
      print(f"Resuming from {args.resume} at epoch {start_epoch} with best_val={best_validation_loss:.6f}")
    elif args.init_checkpoint is not None:
      checkpoint = load_checkpoint(args.init_checkpoint, device)
      validate_checkpoint_config(checkpoint, config)
      load_model_state(model, checkpoint)
      print(f"Initialized model weights from {args.init_checkpoint}")

    print(f"Training on {len(train_dataset)} examples, validating on {len(validation_dataset)} examples, device={device}")
    try:
      for epoch in range(start_epoch, start_epoch + args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad():
          validation_metrics = run_epoch(model, validation_loader, criterion, device, None, None)

        learning_rate = optimizer.param_groups[0]["lr"]
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
          best_validation_loss = validation_metrics["loss"]

        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_activity={validation_metrics['activity_loss']:.6f} "
            f"val_params={validation_metrics['parameter_loss']:.6f} "
            f"val_f0={validation_metrics['f0_loss']:.6f} "
            f"val_f0_cents={validation_metrics['f0_cents_mae']:.1f} "
            f"val_crowd={validation_metrics['crowding_loss']:.6f} "
            f"val_render={validation_metrics['render_feature_loss']:.6f} "
            f"val_rms={validation_metrics['render_rms_loss']:.6f} "
            f"best_val={best_validation_loss:.6f}"
        )

        write_metrics_row(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_activity_loss": train_metrics["activity_loss"],
                "train_parameter_loss": train_metrics["parameter_loss"],
                "train_f0_loss": train_metrics["f0_loss"],
                "train_f0_cents_mae": train_metrics["f0_cents_mae"],
                "train_crowding_loss": train_metrics["crowding_loss"],
                "train_render_feature_loss": train_metrics["render_feature_loss"],
                "train_render_rms_loss": train_metrics["render_rms_loss"],
                "val_loss": validation_metrics["loss"],
                "val_activity_loss": validation_metrics["activity_loss"],
                "val_parameter_loss": validation_metrics["parameter_loss"],
                "val_f0_loss": validation_metrics["f0_loss"],
                "val_f0_cents_mae": validation_metrics["f0_cents_mae"],
                "val_crowding_loss": validation_metrics["crowding_loss"],
                "val_render_feature_loss": validation_metrics["render_feature_loss"],
                "val_render_rms_loss": validation_metrics["render_rms_loss"],
                "best_validation_loss": best_validation_loss,
                "learning_rate": learning_rate,
            },
        )
        log_tensorboard(writer, epoch, train_metrics, validation_metrics, learning_rate)

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, best_validation_loss, args)
        if improved:
          save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_validation_loss, args)
    finally:
      if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

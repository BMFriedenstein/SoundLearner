from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import SoundLearnerDataset, discover_examples
from .losses import OscillatorLoss
from .model import ModelConfig, SoundLearnerNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SoundLearner oscillator inverse model.")
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="Root containing features/, metadata/, and dataN.data files.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"), help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--max-oscillators", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=None, help="Optional strict SLFT frequency/time resolution.")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    return parser.parse_args()


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
    totals = {"loss": 0.0, "activity_loss": 0.0, "parameter_loss": 0.0}
    count = 0

    for batch in loader:
      batch = move_batch(batch, device)
      with torch.set_grad_enabled(is_training):
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
          predictions = model(batch["features"])
          loss, metrics = criterion(predictions, batch["target"], batch["mask"])

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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = discover_examples(args.dataset_root)
    dataset = SoundLearnerDataset(examples, max_oscillators=args.max_oscillators, expected_resolution=args.resolution)
    train_dataset, validation_dataset = split_dataset(dataset, args.validation_split, args.seed)

    first_sample = dataset[0]
    input_channels = int(first_sample["features"].shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(input_channels=input_channels, max_oscillators=args.max_oscillators, width=args.width, dropout=args.dropout)
    model = SoundLearnerNet(config).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    criterion = OscillatorLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    best_validation_loss = float("inf")

    print(f"Training on {len(train_dataset)} examples, validating on {len(validation_dataset)} examples, device={device}")
    for epoch in range(1, args.epochs + 1):
      train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
      with torch.no_grad():
        validation_metrics = run_epoch(model, validation_loader, criterion, device, None, None)

      print(
          f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
          f"val_loss={validation_metrics['loss']:.6f} "
          f"val_activity={validation_metrics['activity_loss']:.6f} "
          f"val_params={validation_metrics['parameter_loss']:.6f}"
      )

      save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, best_validation_loss, args)
      if validation_metrics["loss"] < best_validation_loss:
        best_validation_loss = validation_metrics["loss"]
        save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_validation_loss, args)


if __name__ == "__main__":
    main()

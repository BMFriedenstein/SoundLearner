from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "deep_trainer" / "configs" / "adaptive_curriculum_v2_2048.toml"

PROMOTION_COLUMNS = {
    "max_val_loss": "val_loss",
    "max_val_activity_loss": "val_activity_loss",
    "max_val_activity_count_loss": "val_activity_count_loss",
    "max_val_activity_soft_count_mae": "val_activity_soft_count_mae",
    "max_val_activity_hard_count_mae": "val_activity_hard_count_mae",
    "max_val_activity_probability_mae": "val_activity_probability_mae",
    "max_val_parameter_loss": "val_parameter_loss",
    "max_val_f0_loss": "val_f0_loss",
    "max_val_f0_cents_mae": "val_f0_cents_mae",
    "max_val_crowding_loss": "val_crowding_loss",
    "max_val_render_feature_loss": "val_render_feature_loss",
    "max_val_render_rms_loss": "val_render_rms_loss",
}

TRAIN_ARGUMENTS = {
    "batch_size",
    "learning_rate",
    "weight_decay",
    "validation_split",
    "max_oscillators",
    "freq_bins",
    "time_frames",
    "resolution",
    "width",
    "dropout",
    "coordinate_channels",
    "normalization",
    "num_workers",
    "seed",
    "amp",
    "tensorboard",
    "activity_loss_weight",
    "activity_count_loss_weight",
    "parameter_loss_weight",
    "activity_positive_weight",
    "f0_loss_weight",
    "crowding_loss_weight",
    "f0_min_frequency",
    "f0_max_frequency",
    "render_loss_weight",
    "render_rms_loss_weight",
    "render_loss_sample_rate",
    "render_loss_seconds",
    "render_loss_fft_size_multiplier",
}


@dataclass(frozen=True)
class DatasetDefaults:
    root: Path
    samples_per_grade: int = 1000
    sample_time: int = 5
    freq_bins: int = 1024
    time_frames: int = 512
    fft_size_multiplier: int = 4
    workers: int = 8
    min_note_frequency: float = 55.0
    max_note_frequency: float = 440.0
    builder: str | None = None


@dataclass(frozen=True)
class TrainingDefaults:
    run_root: Path
    epochs_per_attempt: int = 5
    max_attempts_per_grade: int = 6
    train_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealEvaluation:
    manifest: Path | None = None
    limit: int | None = None
    device: str = "cpu"
    enabled: bool = False


@dataclass(frozen=True)
class Grade:
    name: str
    rel_path: str
    min_instrument_size: int
    max_instrument_size: int
    min_uncoupled_oscilators: int = 0
    max_uncoupled_oscilators: int = 0
    min_frequency_factor: float | None = None
    max_frequency_factor: float | None = None
    coupled_frequency_factors: tuple[float, ...] = ()
    require_fundamental: bool = False
    samples: int | None = None
    max_attempts: int | None = None
    min_epochs: int = 1
    promotion: dict[str, float] = field(default_factory=dict)
    train_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupervisorConfig:
    name: str
    dataset: DatasetDefaults
    training: TrainingDefaults
    grades: list[Grade]
    real_evaluation: RealEvaluation = field(default_factory=RealEvaluation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive dataset/training supervisor for SoundLearner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=Path(".venv/Scripts/python.exe"))
    parser.add_argument("--start-grade", default=None, help="Grade name to start from, overriding saved state.")
    parser.add_argument("--stop-grade", default=None, help="Grade name to stop after.")
    parser.add_argument("--state", type=Path, default=None, help="Override supervisor state path.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild each grade dataset even if it looks complete.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore prior supervisor state and start fresh.")
    parser.add_argument("--reset-grade", action="append", default=[], help="Forget one grade in supervisor state without resetting earlier grades.")
    parser.add_argument("--skip-native-build", action="store_true", help="Do not run meson compile before dataset generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run without changing files.")
    parser.add_argument("--workers", type=int, default=None, help="Override dataset worker count.")
    parser.add_argument("--samples-per-grade", type=int, default=None, help="Override generated samples per grade.")
    parser.add_argument("--epochs-per-attempt", type=int, default=None, help="Override training epochs per supervisor attempt.")
    parser.add_argument("--max-attempts-per-grade", type=int, default=None, help="Override max attempts per grade.")
    parser.add_argument("--real-eval", action="store_true", help="Run real-audio evaluation after promoted grades when configured.")
    return parser.parse_args()


def as_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
      return path
    return REPO_ROOT / path


def normalize_train_key(key: str) -> str:
    return key.replace("-", "_")


def load_config(path: Path) -> SupervisorConfig:
    if not path.exists():
      raise FileNotFoundError(f"Adaptive curriculum config does not exist: {path}")
    data = tomllib.loads(path.read_text())

    dataset_data = data.get("dataset", {})
    dataset = DatasetDefaults(
        root=as_repo_path(dataset_data.get("root", "datasets/curricula/adaptive_curriculum_v1")),
        samples_per_grade=int(dataset_data.get("samples_per_grade", 1000)),
        sample_time=int(dataset_data.get("sample_time", 5)),
        freq_bins=int(dataset_data.get("freq_bins", 1024)),
        time_frames=int(dataset_data.get("time_frames", 512)),
        fft_size_multiplier=int(dataset_data.get("fft_size_multiplier", 4)),
        workers=int(dataset_data.get("workers", 8)),
        min_note_frequency=float(dataset_data.get("min_note_frequency", 55.0)),
        max_note_frequency=float(dataset_data.get("max_note_frequency", 440.0)),
        builder=dataset_data.get("builder"),
    )

    training_data = data.get("training", {})
    train_args = {
        normalize_train_key(key): value
        for key, value in training_data.items()
        if normalize_train_key(key) in TRAIN_ARGUMENTS
    }
    training = TrainingDefaults(
        run_root=as_repo_path(training_data.get("run_root", "runs/adaptive_curriculum_v1")),
        epochs_per_attempt=int(training_data.get("epochs_per_attempt", 5)),
        max_attempts_per_grade=int(training_data.get("max_attempts_per_grade", 6)),
        train_args=train_args,
    )

    real_eval_data = data.get("real_evaluation", {})
    manifest_value = real_eval_data.get("manifest")
    real_evaluation = RealEvaluation(
        manifest=as_repo_path(manifest_value) if manifest_value else None,
        limit=int(real_eval_data["limit"]) if "limit" in real_eval_data else None,
        device=str(real_eval_data.get("device", "cpu")),
        enabled=bool(real_eval_data.get("enabled", False)),
    )

    grades = []
    for grade_data in data.get("grades", []):
      promotion = {
          key: float(value)
          for key, value in grade_data.items()
          if key in PROMOTION_COLUMNS
      }
      train_args = {
          normalize_train_key(key): value
          for key, value in grade_data.items()
          if normalize_train_key(key) in TRAIN_ARGUMENTS
      }
      coupled_frequency_factors = tuple(float(value) for value in grade_data.get("coupled_frequency_factors", []))
      grades.append(
          Grade(
              name=str(grade_data["name"]),
              rel_path=str(grade_data["rel_path"]),
              min_instrument_size=int(grade_data["min_instrument_size"]),
              max_instrument_size=int(grade_data["max_instrument_size"]),
              min_uncoupled_oscilators=int(grade_data.get("min_uncoupled_oscilators", 0)),
              max_uncoupled_oscilators=int(grade_data.get("max_uncoupled_oscilators", 0)),
              min_frequency_factor=float(grade_data["min_frequency_factor"]) if "min_frequency_factor" in grade_data else None,
              max_frequency_factor=float(grade_data["max_frequency_factor"]) if "max_frequency_factor" in grade_data else None,
              coupled_frequency_factors=coupled_frequency_factors,
              require_fundamental=bool(grade_data.get("require_fundamental", False)),
              samples=int(grade_data["samples"]) if "samples" in grade_data else None,
              max_attempts=int(grade_data["max_attempts"]) if "max_attempts" in grade_data else None,
              min_epochs=int(grade_data.get("min_epochs", 1)),
              promotion=promotion,
              train_args=train_args,
          )
      )
    if not grades:
      raise ValueError("Adaptive curriculum config must contain at least one [[grades]] entry")
    return SupervisorConfig(
        name=str(data.get("name", path.stem)),
        dataset=dataset,
        training=training,
        grades=grades,
        real_evaluation=real_evaluation,
    )


def ensure_python(path: Path) -> Path:
    resolved = as_repo_path(path) if not path.is_absolute() else path
    if not resolved.exists():
      raise FileNotFoundError(f"Missing Python executable: {resolved}")
    return resolved


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    if not drive:
      return resolved.as_posix()
    return f"/mnt/{drive}{tail}"


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: Sequence[str], *, dry_run: bool = False) -> None:
    print(f"$ {quote_command(command)}")
    if dry_run:
      return
    completed = subprocess.run(list(map(str, command)), cwd=REPO_ROOT, text=True)
    if completed.returncode != 0:
      raise subprocess.CalledProcessError(completed.returncode, command)


def run_native_build(dry_run: bool) -> None:
    command = f"cd {shlex.quote(to_wsl_path(REPO_ROOT))} && meson compile -C build"
    run_command(["wsl", "bash", "-lc", command], dry_run=dry_run)


def check_dataset_ready(python_exe: Path, dataset_root: Path, expected_samples: int) -> bool:
    completed = subprocess.run(
        [
            str(python_exe),
            str(REPO_ROOT / "scripts" / "check_dataset_ready.py"),
            "--dataset-root",
            str(dataset_root),
            "--expected-samples",
            str(expected_samples),
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    return completed.returncode == 0


def dataset_builder_path(config: SupervisorConfig) -> str:
    if config.dataset.builder:
      return config.dataset.builder
    return to_wsl_path(REPO_ROOT / "build" / "dataset_builder" / "dataset_builder")


def build_grade_dataset(
    python_exe: Path,
    config: SupervisorConfig,
    grade: Grade,
    workers: int,
    samples: int,
    force_rebuild: bool,
    dry_run: bool,
) -> Path:
    output_root = config.dataset.root / grade.rel_path
    if not force_rebuild and not dry_run and check_dataset_ready(python_exe, output_root, samples):
      print(f"[dataset] {grade.name}: ready at {output_root}")
      return output_root

    command = [
        str(python_exe),
        str(REPO_ROOT / "scripts" / "build_dataset_sharded.py"),
        "--output-root",
        str(output_root),
        "--num-samples",
        str(samples),
        "--sample-time",
        str(config.dataset.sample_time),
        "--min-instrument-size",
        str(grade.min_instrument_size),
        "--max-instrument-size",
        str(grade.max_instrument_size),
        "--min-uncoupled-oscilators",
        str(grade.min_uncoupled_oscilators),
        "--max-uncoupled-oscilators",
        str(grade.max_uncoupled_oscilators),
        "--min-note-frequency",
        str(config.dataset.min_note_frequency),
        "--max-note-frequency",
        str(config.dataset.max_note_frequency),
        "--min-frequency-factor",
        str(grade.min_frequency_factor if grade.min_frequency_factor is not None else 0.0),
        "--max-frequency-factor",
        str(grade.max_frequency_factor if grade.max_frequency_factor is not None else 1.0),
        "--freq-bins",
        str(config.dataset.freq_bins),
        "--time-frames",
        str(config.dataset.time_frames),
        "--fft-size-multiplier",
        str(config.dataset.fft_size_multiplier),
        "--workers",
        str(workers),
        "--python-exe",
        str(python_exe),
        "--builder",
        dataset_builder_path(config),
    ]
    if grade.require_fundamental:
      command.append("--require-fundamental")
    if grade.coupled_frequency_factors:
      command.extend(["--coupled-frequency-factors", ",".join(str(value) for value in grade.coupled_frequency_factors)])
    run_command(command, dry_run=dry_run)
    return output_root


def append_cli_argument(command: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
      if value:
        command.append(f"--{name.replace('_', '-')}")
      return
    if value is None:
      return
    command.extend([f"--{name.replace('_', '-')}", str(value)])


def train_grade_attempt(
    python_exe: Path,
    config: SupervisorConfig,
    grade: Grade,
    dataset_root: Path,
    attempt: int,
    init_checkpoint: Path | None,
    dry_run: bool,
) -> Path:
    run_dir = config.training.run_root / grade.name
    last_checkpoint = run_dir / "last.pt"
    command = [
        str(python_exe),
        "-m",
        "deep_trainer.train",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(config.training.epochs_per_attempt),
    ]
    train_args = {**config.training.train_args, **grade.train_args}
    for key in sorted(train_args):
      append_cli_argument(command, key, train_args[key])
    if last_checkpoint.exists() and attempt > 1:
      command.extend(["--resume", str(last_checkpoint)])
    elif init_checkpoint is not None:
      command.extend(["--init-checkpoint", str(init_checkpoint)])
    run_command(command, dry_run=dry_run)
    return run_dir


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
      return []
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
      for row in csv.DictReader(handle):
        numeric: dict[str, float] = {}
        for key, value in row.items():
          try:
            numeric[key] = float(value)
          except (TypeError, ValueError):
            pass
        rows.append(numeric)
    return rows


def best_metric_row(rows: Iterable[dict[str, float]]) -> dict[str, float] | None:
    rows = list(rows)
    if not rows:
      return None
    return min(rows, key=lambda row: row.get("val_loss", float("inf")))


def threshold_failures_for_row(grade: Grade, row: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for threshold_key, metric_key in PROMOTION_COLUMNS.items():
      if threshold_key not in grade.promotion:
        continue
      value = row.get(metric_key)
      threshold = grade.promotion[threshold_key]
      if value is None:
        failures.append(f"{metric_key} missing")
      elif value > threshold:
        failures.append(f"{metric_key}={value:.6g} > {threshold:.6g}")
    return failures


def promotion_score(grade: Grade, row: dict[str, float]) -> float:
    ratios: list[float] = []
    for threshold_key, metric_key in PROMOTION_COLUMNS.items():
      if threshold_key not in grade.promotion:
        continue
      value = row.get(metric_key)
      threshold = grade.promotion[threshold_key]
      if value is None:
        ratios.append(float("inf"))
      else:
        ratios.append(value / max(threshold, 1e-12))
    return max(ratios, default=row.get("val_loss", float("inf")))


def best_promotion_row(grade: Grade, rows: Iterable[dict[str, float]]) -> dict[str, float] | None:
    rows = list(rows)
    if not rows:
      return None
    return min(rows, key=lambda row: promotion_score(grade, row))


def promotion_failures(grade: Grade, rows: list[dict[str, float]]) -> list[str]:
    best = best_promotion_row(grade, rows)
    if best is None:
      return ["no metrics written yet"]
    failures = []
    if len(rows) < grade.min_epochs:
      failures.append(f"needs at least {grade.min_epochs} epochs, has {len(rows)}")
    if len(rows) >= grade.min_epochs:
      for row in rows:
        if not threshold_failures_for_row(grade, row):
          return []
    failures.extend(threshold_failures_for_row(grade, best))
    return failures


def grade_passed(grade: Grade, rows: list[dict[str, float]]) -> bool:
    return not promotion_failures(grade, rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
      return str(value)
    if isinstance(value, dict):
      return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
      return [json_safe(item) for item in value]
    return value


def load_state(path: Path, reset: bool) -> dict[str, Any]:
    if reset or not path.exists():
      return {"grades": {}, "history": []}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(state), indent=2))


def write_history_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
      if not file_exists:
        writer.writeheader()
      writer.writerow(row)


def run_real_evaluation(
    python_exe: Path,
    config: SupervisorConfig,
    grade: Grade,
    checkpoint: Path,
    dry_run: bool,
) -> None:
    real_eval = config.real_evaluation
    if not real_eval.manifest:
      return
    if not dry_run and not real_eval.manifest.exists():
      print(f"[real-eval] manifest missing, skipping: {real_eval.manifest}")
      return
    output_dir = REPO_ROOT / "sounds" / "eval" / config.name / grade.name
    command = [
        str(python_exe),
        "-m",
        "deep_trainer.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(real_eval.manifest),
        "--output-dir",
        str(output_dir),
        "--freq-bins",
        str(config.dataset.freq_bins),
        "--time-frames",
        str(config.dataset.time_frames),
        "--device",
        real_eval.device,
    ]
    if real_eval.limit is not None:
      command.extend(["--limit", str(real_eval.limit)])
    run_command(command, dry_run=dry_run)


def grade_index(grades: list[Grade], name: str | None, default: int) -> int:
    if name is None:
      return default
    for index, grade in enumerate(grades):
      if grade.name == name:
        return index
    raise ValueError(f"Unknown grade {name!r}. Known grades: {', '.join(grade.name for grade in grades)}")


def state_start_index(config: SupervisorConfig, state: dict[str, Any]) -> int:
    completed = {
        name
        for name, info in state.get("grades", {}).items()
        if isinstance(info, dict) and info.get("promoted")
    }
    for index, grade in enumerate(config.grades):
      if grade.name not in completed:
        return index
    return len(config.grades)


def best_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    best = best_metric_row(rows) or {}
    keys = [
        "epoch",
        "val_loss",
        "val_activity_loss",
        "val_activity_soft_count_mae",
        "val_activity_hard_count_mae",
        "val_activity_probability_mae",
        "val_parameter_loss",
        "val_f0_cents_mae",
        "val_render_feature_loss",
        "val_render_rms_loss",
    ]
    return {key: best[key] for key in keys if key in best}


def run_supervisor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    python_exe = ensure_python(args.python_exe)

    workers = args.workers if args.workers is not None else config.dataset.workers
    samples_per_grade = args.samples_per_grade if args.samples_per_grade is not None else config.dataset.samples_per_grade
    epochs_per_attempt = args.epochs_per_attempt if args.epochs_per_attempt is not None else config.training.epochs_per_attempt
    max_attempts_per_grade = (
        args.max_attempts_per_grade
        if args.max_attempts_per_grade is not None
        else config.training.max_attempts_per_grade
    )
    config = SupervisorConfig(
        name=config.name,
        dataset=config.dataset,
        training=TrainingDefaults(
            run_root=config.training.run_root,
            epochs_per_attempt=epochs_per_attempt,
            max_attempts_per_grade=max_attempts_per_grade,
            train_args=config.training.train_args,
        ),
        grades=config.grades,
        real_evaluation=config.real_evaluation,
    )

    state_path = as_repo_path(args.state) if args.state is not None else config.training.run_root / "supervisor_state.json"
    history_path = config.training.run_root / "supervisor_history.csv"
    state = load_state(state_path, args.reset_state)
    for grade_name in args.reset_grade:
      state.get("grades", {}).pop(grade_name, None)
    state["config"] = str(args.config)
    state["curriculum"] = config.name

    if args.dry_run:
      print("[dry-run] No files will be changed.")
      if not args.skip_native_build:
        run_native_build(args.dry_run)
    elif not args.skip_native_build:
      run_native_build(args.dry_run)

    start_index = grade_index(config.grades, args.start_grade, state_start_index(config, state))
    stop_index = grade_index(config.grades, args.stop_grade, len(config.grades) - 1)
    if start_index >= len(config.grades):
      print("All configured grades are already promoted.")
      save_state(state_path, state)
      return
    if start_index > stop_index:
      raise ValueError("Start grade comes after stop grade")

    init_checkpoint: Path | None = None
    if start_index > 0:
      previous_info = state.get("grades", {}).get(config.grades[start_index - 1].name, {})
      checkpoint_value = previous_info.get("checkpoint") if isinstance(previous_info, dict) else None
      if checkpoint_value:
        init_checkpoint = Path(checkpoint_value)

    for grade in config.grades[start_index : stop_index + 1]:
      samples = grade.samples or samples_per_grade
      if args.max_attempts_per_grade is not None:
        max_attempts = config.training.max_attempts_per_grade
      else:
        max_attempts = grade.max_attempts or config.training.max_attempts_per_grade
      dataset_root = build_grade_dataset(
          python_exe,
          config,
          grade,
          workers=workers,
          samples=samples,
          force_rebuild=args.force_rebuild,
          dry_run=args.dry_run,
      )

      print(f"[grade] {grade.name}: {grade.min_instrument_size}..{grade.max_instrument_size} coupled, attempts <= {max_attempts}")
      promoted = False
      run_dir = config.training.run_root / grade.name
      starting_attempt = int(state.get("grades", {}).get(grade.name, {}).get("attempts", 0)) + 1
      for attempt in range(starting_attempt, max_attempts + 1):
        if args.dry_run:
          train_grade_attempt(python_exe, config, grade, dataset_root, attempt, init_checkpoint, args.dry_run)
          print("[dry-run] Stopping after command preview for this grade.")
          return

        run_dir = train_grade_attempt(python_exe, config, grade, dataset_root, attempt, init_checkpoint, args.dry_run)
        rows = read_metrics(run_dir / "metrics.csv")
        best = best_summary(rows)
        failures = promotion_failures(grade, rows)
        promoted = not failures
        checkpoint = run_dir / "best.pt"
        info = {
            "attempts": attempt,
            "promoted": promoted,
            "dataset_root": dataset_root,
            "run_dir": run_dir,
            "checkpoint": checkpoint if checkpoint.exists() else None,
            "best": best,
            "failures": failures,
        }
        state.setdefault("grades", {})[grade.name] = info
        state.setdefault("history", []).append({"grade": grade.name, "attempt": attempt, "promoted": promoted, **best})
        save_state(state_path, state)
        write_history_row(history_path, {"grade": grade.name, "attempt": attempt, "promoted": promoted, **best})

        if promoted:
          print(f"[promote] {grade.name}: {best}")
          if (config.real_evaluation.enabled or args.real_eval) and checkpoint.exists():
            run_real_evaluation(python_exe, config, grade, checkpoint, args.dry_run)
          init_checkpoint = checkpoint
          break

        print(f"[hold] {grade.name}: " + "; ".join(failures))

      if not promoted:
        print(f"[stop] {grade.name} did not pass after {max_attempts} attempt(s).")
        break

    save_state(state_path, state)
    print(f"Adaptive curriculum supervisor state: {state_path}")


def main() -> None:
    args = parse_args()
    run_supervisor(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CleanStage:
    name: str
    rel_path: str
    min_coupled: int
    max_coupled: int
    min_uncoupled: int
    max_uncoupled: int


@dataclass(frozen=True)
class AugmentStage:
    name: str
    rel_path: str
    gain_min: float
    gain_max: float
    snr_min: float
    snr_max: float
    low_shelf_min: float
    low_shelf_max: float
    high_shelf_min: float
    high_shelf_max: float
    reverb_mix_max: float
    saturation_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SoundLearner dataset curricula from one Python entrypoint.")
    parser.add_argument("--curriculum", choices=["oscillator_v1", "complexity_v1"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-native-build", action="store_true")
    parser.add_argument("--python-exe", type=Path, default=Path(".venv/Scripts/python.exe"))
    return parser.parse_args()


def run_command(command: Sequence[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(list(command), cwd=cwd or REPO_ROOT, text=True)
    if completed.returncode != 0:
      raise subprocess.CalledProcessError(completed.returncode, command)


def run_wsl(command: str) -> None:
    run_command(["wsl", "bash", "-lc", command], cwd=REPO_ROOT)


def ensure_python(python_exe: Path) -> Path:
    resolved = (REPO_ROOT / python_exe).resolve() if not python_exe.is_absolute() else python_exe
    if not resolved.exists():
      raise FileNotFoundError(f"Missing Python executable: {resolved}")
    return resolved


def ensure_native_build() -> None:
    print("[native] meson compile -C build")
    run_wsl(f"cd {shlex.quote(to_wsl_path(REPO_ROOT))} && meson compile -C build")


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
      path.write_text(text)


def check_dataset_ready(python_exe: Path, dataset_root: Path, expected_samples: int) -> bool:
    completed = subprocess.run(
        [str(python_exe), str(REPO_ROOT / "scripts" / "check_dataset_ready.py"), "--dataset-root", str(dataset_root), "--expected-samples", str(expected_samples)],
        cwd=REPO_ROOT,
        text=True,
    )
    return completed.returncode == 0


def build_sharded_dataset(
    python_exe: Path,
    output_root: Path,
    expected_samples: int,
    workers: int,
    stage: CleanStage,
) -> None:
    run_command(
        [
            str(python_exe),
            str(REPO_ROOT / "scripts" / "build_dataset_sharded.py"),
            "--output-root",
            str(output_root),
            "--num-samples",
            str(expected_samples),
            "--sample-time",
            "5",
            "--min-instrument-size",
            str(stage.min_coupled),
            "--max-instrument-size",
            str(stage.max_coupled),
            "--min-uncoupled-oscilators",
            str(stage.min_uncoupled),
            "--max-uncoupled-oscilators",
            str(stage.max_uncoupled),
            "--min-note-frequency",
            "55.0",
            "--max-note-frequency",
            "440.0",
            "--freq-bins",
            "1024",
            "--time-frames",
            "512",
            "--fft-size-multiplier",
            "4",
            "--workers",
            str(workers),
            "--python-exe",
            str(python_exe),
        ]
    )


def build_augmented_dataset(
    python_exe: Path,
    input_root: Path,
    output_root: Path,
    stage: AugmentStage,
) -> None:
    run_command(
        [
            str(python_exe),
            "-m",
            "deep_trainer.dataset_augmentor",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--variants-per-input",
            "1",
            "--tool-mode",
            "wsl",
            "--skip-previews",
            "--gain-db-min",
            str(stage.gain_min),
            "--gain-db-max",
            str(stage.gain_max),
            "--snr-db-min",
            str(stage.snr_min),
            "--snr-db-max",
            str(stage.snr_max),
            "--low-shelf-db-min",
            str(stage.low_shelf_min),
            "--low-shelf-db-max",
            str(stage.low_shelf_max),
            "--high-shelf-db-min",
            str(stage.high_shelf_min),
            "--high-shelf-db-max",
            str(stage.high_shelf_max),
            "--reverb-mix-max",
            str(stage.reverb_mix_max),
            "--saturation-max",
            str(stage.saturation_max),
        ]
    )


def build_oscillator_curriculum(args: argparse.Namespace, python_exe: Path) -> None:
    root = REPO_ROOT / "datasets" / "curricula" / "oscillator_curriculum_v1"
    ensure_dir(root)
    write_if_missing(
        root / "README.md",
        "# Oscillator Curriculum v1\n\n"
        "This curriculum organizes synthetic dataset stages from clean to progressively more recording-like augmentation.\n\n"
        "Stages:\n\n"
        "```text\n"
        "00_clean_varcount_1024x512_1k\n"
        "10_realish_light_1024x512_1k\n"
        "20_realish_medium_1024x512_1k\n"
        "30_realish_heavy_1024x512_1k\n"
        "```\n",
    )

    clean_stage = CleanStage("clean", "00_clean_varcount_1024x512_1k", 8, 64, 0, 12)
    clean_root = root / clean_stage.rel_path
    if args.force or not check_dataset_ready(python_exe, clean_root, args.expected_samples):
      print(f"[build] {clean_stage.rel_path}")
      build_sharded_dataset(python_exe, clean_root, args.expected_samples, args.workers, clean_stage)
    else:
      print(f"[skip]  {clean_stage.rel_path}")

    augment_stages = [
        AugmentStage("light", "10_realish_light_1024x512_1k", -2, 2, 34, 44, -2, 2, -2, 2, 0.04, 0.05),
        AugmentStage("medium", "20_realish_medium_1024x512_1k", -4, 4, 26, 38, -4, 4, -4, 4, 0.10, 0.10),
        AugmentStage("heavy", "30_realish_heavy_1024x512_1k", -6, 6, 18, 30, -6, 6, -6, 6, 0.18, 0.18),
    ]
    for stage in augment_stages:
      stage_root = root / stage.rel_path
      if args.force or not check_dataset_ready(python_exe, stage_root, args.expected_samples):
        print(f"[build] {stage.rel_path}")
        build_augmented_dataset(python_exe, clean_root, stage_root, stage)
      else:
        print(f"[skip]  {stage.rel_path}")


def build_complexity_curriculum(args: argparse.Namespace, python_exe: Path) -> None:
    root = REPO_ROOT / "datasets" / "curricula" / "complexity_curriculum_v1"
    ensure_dir(root)
    write_if_missing(
        root / "README.md",
        "# Complexity Curriculum v1\n\n"
        "This curriculum increases oscillator-count complexity in stages so the model can learn simple decompositions before tackling dense mixtures.\n\n"
        "Stages:\n\n"
        "```text\n"
        "c1_1to3_clean_1024x512_1k\n"
        "c2_1to5_clean_1024x512_1k\n"
        "c3_1to8_clean_1024x512_1k\n"
        "c4_1to12_plus2_uncoupled_1024x512_1k\n"
        "c5_1to20_plus4_uncoupled_1024x512_1k\n"
        "c6_1to32_plus8_uncoupled_1024x512_1k\n"
        "c7_1to64_plus12_uncoupled_1024x512_1k\n"
        "```\n",
    )

    stages = [
        CleanStage("c1", "c1_1to3_clean_1024x512_1k", 1, 3, 0, 0),
        CleanStage("c2", "c2_1to5_clean_1024x512_1k", 1, 5, 0, 0),
        CleanStage("c3", "c3_1to8_clean_1024x512_1k", 1, 8, 0, 0),
        CleanStage("c4", "c4_1to12_plus2_uncoupled_1024x512_1k", 1, 12, 0, 2),
        CleanStage("c5", "c5_1to20_plus4_uncoupled_1024x512_1k", 1, 20, 0, 4),
        CleanStage("c6", "c6_1to32_plus8_uncoupled_1024x512_1k", 1, 32, 0, 8),
        CleanStage("c7", "c7_1to64_plus12_uncoupled_1024x512_1k", 1, 64, 0, 12),
    ]
    for stage in stages:
      stage_root = root / stage.rel_path
      if args.force or not check_dataset_ready(python_exe, stage_root, args.expected_samples):
        print(f"[build] {stage.rel_path}")
        build_sharded_dataset(python_exe, stage_root, args.expected_samples, args.workers, stage)
      else:
        print(f"[skip]  {stage.rel_path}")


def main() -> None:
    args = parse_args()
    python_exe = ensure_python(args.python_exe)
    ensure_dir(REPO_ROOT / "datasets" / "curricula")

    if not args.skip_native_build:
      ensure_native_build()

    if args.curriculum == "oscillator_v1":
      build_oscillator_curriculum(args, python_exe)
    else:
      build_complexity_curriculum(args, python_exe)

    print(f"{args.curriculum} dataset build complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dataset with multiple dataset_builder processes and merge the outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--sample-time", type=int, required=True)
    parser.add_argument("--min-instrument-size", type=int, required=True)
    parser.add_argument("--max-instrument-size", type=int, required=True)
    parser.add_argument("--min-uncoupled-oscilators", type=int, required=True)
    parser.add_argument("--max-uncoupled-oscilators", type=int, required=True)
    parser.add_argument("--min-note-frequency", type=float, default=55.0)
    parser.add_argument("--max-note-frequency", type=float, default=440.0)
    parser.add_argument("--min-frequency-factor", type=float, default=0.0)
    parser.add_argument("--max-frequency-factor", type=float, default=1.0)
    parser.add_argument("--coupled-frequency-factors", default="")
    parser.add_argument("--require-fundamental", action="store_true")
    parser.add_argument("--freq-bins", type=int, required=True)
    parser.add_argument("--time-frames", type=int, required=True)
    parser.add_argument("--fft-size-multiplier", type=int, default=4)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--builder",
        default="/mnt/c/Users/Brandon/Documents/wip/SoundLearner/build/dataset_builder/dataset_builder",
        help="WSL path to dataset_builder executable.",
    )
    return parser.parse_args()


def to_wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def shard_counts(total: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total))
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


def numeric_key(path: Path) -> int:
    return int(path.stem.removeprefix("data"))


def clean_output_root(output_root: Path) -> None:
    if output_root.exists():
      stubborn_paths: list[Path] = []
      for attempt in range(5):
        children = list(output_root.iterdir())
        if not children:
          break
        stubborn_paths.clear()
        for child in children:
          try:
            if child.is_dir():
              shutil.rmtree(child)
            else:
              child.unlink()
          except OSError:
            stubborn_paths.append(child)
        if not stubborn_paths:
          break
        time.sleep(0.25 * (attempt + 1))
      if stubborn_paths:
        joined = "\n".join(str(path) for path in stubborn_paths)
        raise RuntimeError(f"Unable to clean dataset output root after multiple attempts:\n{joined}")
    output_root.mkdir(parents=True, exist_ok=True)


def run_shard(shard_dir: Path, sample_count: int, args: argparse.Namespace, shard_index: int) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {to_wsl_path(shard_dir)} && "
        f"{args.builder} "
        f"-n {sample_count} "
        f"-t {args.sample_time} "
        f"--min-instrument-size {args.min_instrument_size} "
        f"--max-instrument-size {args.max_instrument_size} "
        f"--min-uncoupled-oscilators {args.min_uncoupled_oscilators} "
        f"--max-uncoupled-oscilators {args.max_uncoupled_oscilators} "
        f"--min-note-frequency {args.min_note_frequency} "
        f"--max-note-frequency {args.max_note_frequency} "
        f"--min-frequency-factor {args.min_frequency_factor} "
        f"--max-frequency-factor {args.max_frequency_factor}"
    )
    if args.require_fundamental:
      command += " --require-fundamental"
    if args.coupled_frequency_factors:
      command += f" --coupled-frequency-factors {args.coupled_frequency_factors}"
    result = subprocess.run(["wsl", "bash", "-lc", command], text=True, capture_output=True)
    if result.returncode != 0:
      raise RuntimeError(
          f"Shard {shard_index} failed with exit code {result.returncode}\n"
          f"stdout:\n{result.stdout}\n"
          f"stderr:\n{result.stderr}"
      )


def merge_shards(output_root: Path, shard_root: Path) -> None:
    next_index = 0
    for shard_dir in sorted(shard_root.iterdir()):
      for wav_path in sorted(shard_dir.glob("data*.wav"), key=numeric_key):
        local_index = numeric_key(wav_path)
        old_stem = f"data{local_index}"
        new_stem = f"data{next_index}"

        shutil.move(str(shard_dir / f"{old_stem}.wav"), output_root / f"{new_stem}.wav")
        if (shard_dir / f"{old_stem}.data").exists():
          shutil.move(str(shard_dir / f"{old_stem}.data"), output_root / f"{new_stem}.data")
        if (shard_dir / f"{old_stem}.meta").exists():
          shutil.move(str(shard_dir / f"{old_stem}.meta"), output_root / f"{new_stem}.meta")
        next_index += 1


def prepare_dataset(output_root: Path, args: argparse.Namespace) -> None:
    command = [
        str(args.python_exe),
        "-m",
        "deep_trainer.prepare_dataset",
        "--dataset-root",
        str(output_root),
        "--freq-bins",
        str(args.freq_bins),
        "--time-frames",
        str(args.time_frames),
        "--crop-seconds",
        str(args.sample_time),
        "--fft-size-multiplier",
        str(args.fft_size_multiplier),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
      raise RuntimeError(
          f"Dataset preparation failed with exit code {result.returncode}\n"
          f"stdout:\n{result.stdout}\n"
          f"stderr:\n{result.stderr}"
      )
    print(result.stdout, end="")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    counts = shard_counts(args.num_samples, args.workers)
    shard_root = Path(tempfile.mkdtemp(prefix=f"{output_root.name}_shards_", dir=str(output_root.parent)))

    print(f"Building {args.num_samples} samples into {output_root} with {len(counts)} shard(s): {counts}")
    clean_output_root(output_root)

    with ThreadPoolExecutor(max_workers=len(counts)) as executor:
      futures = []
      for shard_index, count in enumerate(counts):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        futures.append(executor.submit(run_shard, shard_dir, count, args, shard_index))
      for future in as_completed(futures):
        future.result()

    merge_shards(output_root, shard_root)
    prepare_dataset(output_root, args)
    shutil.rmtree(shard_root, ignore_errors=True)
    print("Sharded dataset build complete.")


if __name__ == "__main__":
    main()

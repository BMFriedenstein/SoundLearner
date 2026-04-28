from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a dataset folder looks complete enough to skip rebuilding.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    return parser.parse_args()


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
      return 0
    return sum(1 for _ in path.glob(pattern))


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    expected = args.expected_samples

    metadata_count = count_files(root / "metadata", "*.json")
    feature_count = count_files(root / "features", "*.slft")
    wav_count = count_files(root, "*.wav")

    ready = metadata_count >= expected and feature_count >= expected and wav_count >= expected
    print(
        f"{root}: metadata={metadata_count}, features={feature_count}, wavs={wav_count}, "
        f"expected>={expected}, ready={'yes' if ready else 'no'}"
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

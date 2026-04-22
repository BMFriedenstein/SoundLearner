from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import TARGET_CHANNELS, discover_examples, read_oscillator_csv


PARAMETER_NAMES = [
    "active",
    "start_amplitude_factor",
    "start_frequency_factor",
    "phase_factor",
    "amplitude_decay_factor",
    "amplitude_attack_factor",
    "frequency_decay_factor",
    "base_frequency_coupled",
]

GROUP_COLORS = {
    "synthetic_truth": "#2563eb",
    "holdout_prediction": "#dc2626",
}


@dataclass(frozen=True)
class GroupVectors:
    name: str
    label: str
    color: str
    paths: list[Path]
    matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze oscillator parameter-space collapse with PCA/SVD charts."
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Synthetic dataset root. Uses target .data files as the ground-truth parameter space.")
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Evaluation output root. Uses */predictions/*.data as predicted parameter vectors.")
    parser.add_argument("--predicted-data", type=Path, action="append", default=[], help="Individual predicted .data file. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-oscillators", type=int, default=64)
    parser.add_argument("--max-examples-per-group", type=int, default=256)
    return parser.parse_args()


def evenly_sample(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[index] for index in indices.tolist()]


def flatten_oscillator_csv(path: Path, max_oscillators: int) -> np.ndarray:
    target, _ = read_oscillator_csv(path, max_oscillators)
    return target.astype(np.float32).reshape(-1)


def load_group(
    *,
    name: str,
    label: str,
    color: str,
    paths: Iterable[Path],
    max_oscillators: int,
    limit: int,
) -> GroupVectors:
    selected_paths = evenly_sample(sorted(set(paths)), limit)
    if not selected_paths:
        raise ValueError(f"No parameter files found for group '{name}'")
    matrix = np.stack([flatten_oscillator_csv(path, max_oscillators) for path in selected_paths], axis=0)
    return GroupVectors(name=name, label=label, color=color, paths=selected_paths, matrix=matrix)


def gather_groups(args: argparse.Namespace) -> list[GroupVectors]:
    groups: list[GroupVectors] = []
    if args.dataset_root is not None:
        examples = discover_examples(args.dataset_root)
        groups.append(
            load_group(
                name="synthetic_truth",
                label="Synthetic Truth",
                color=GROUP_COLORS["synthetic_truth"],
                paths=[example.target_path for example in examples],
                max_oscillators=args.max_oscillators,
                limit=args.max_examples_per_group,
            )
        )

    predicted_paths = list(args.predicted_data)
    if args.evaluation_root is not None:
        predicted_paths.extend(args.evaluation_root.glob("*/predictions/*.data"))
    if predicted_paths:
        groups.append(
            load_group(
                name="holdout_prediction",
                label="Holdout Prediction",
                color=GROUP_COLORS["holdout_prediction"],
                paths=predicted_paths,
                max_oscillators=args.max_oscillators,
                limit=args.max_examples_per_group,
            )
        )

    if not groups:
        raise ValueError("Nothing to analyze. Provide --dataset-root and/or --evaluation-root / --predicted-data.")
    if sum(group.matrix.shape[0] for group in groups) < 2:
        raise ValueError("Need at least two vectors across all groups for PCA/SVD analysis.")
    return groups


def singular_value_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    energy = singular_values ** 2
    if np.allclose(energy.sum(), 0.0):
        explained = np.zeros_like(singular_values)
        effective_rank = 0.0
    else:
        explained = energy / energy.sum()
        safe = explained[explained > 0]
        entropy = -np.sum(safe * np.log(safe))
        effective_rank = float(np.exp(entropy))
    return singular_values, explained, effective_rank


def compute_pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)
    diffs = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    upper = np.triu_indices(matrix.shape[0], k=1)
    return distances[upper].astype(np.float32)


def compute_pca(groups: list[GroupVectors]) -> np.ndarray:
    combined = np.concatenate([group.matrix for group in groups], axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return centered @ components.T


def write_coordinates_csv(output_path: Path, groups: list[GroupVectors], coordinates: np.ndarray) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "label", "path", "pc1", "pc2"])
        index = 0
        for group in groups:
            for path in group.paths:
                row = coordinates[index]
                writer.writerow([group.name, group.label, str(path), f"{row[0]:.6f}", f"{row[1]:.6f}"])
                index += 1


def format_float(value: float) -> str:
    return f"{value:.4f}"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#111827} .small{font-size:12px} .label{font-size:13px;font-weight:bold} .title{font-size:18px;font-weight:bold} .axis{stroke:#9ca3af;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1} .line{fill:none;stroke-width:2}</style>',
    ]


def finish_svg(lines: list[str]) -> str:
    return "\n".join([*lines, "</svg>"])


def padded_range(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if math.isclose(minimum, maximum):
        pad = 1.0 if math.isclose(minimum, 0.0) else abs(minimum) * 0.1
        return minimum - pad, maximum + pad
    pad = (maximum - minimum) * 0.1
    return minimum - pad, maximum + pad


def project(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if math.isclose(source_min, source_max):
        return (target_min + target_max) / 2.0
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def write_scatter_svg(output_path: Path, groups: list[GroupVectors], coordinates: np.ndarray) -> None:
    width = 900
    height = 620
    margin_left = 80
    margin_right = 220
    margin_top = 60
    margin_bottom = 80
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    x_min, x_max = padded_range(coordinates[:, 0])
    y_min, y_max = padded_range(coordinates[:, 1])

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{margin_left}" y="32">Parameter Space PCA Scatter</text>')
    lines.append(f'<text class="small" x="{margin_left}" y="50">PC1 vs PC2 on flattened oscillator vectors (active flag + 7 parameters per slot)</text>')

    for step in range(6):
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_top + step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        lines.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')

    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')

    index = 0
    for group in groups:
        for _ in group.paths:
            x_value, y_value = coordinates[index]
            x = project(float(x_value), x_min, x_max, plot_left, plot_right)
            y = project(float(y_value), y_min, y_max, plot_bottom, plot_top)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{group.color}" fill-opacity="0.85"/>')
            index += 1

    for step in range(6):
        x_value = x_min + step * (x_max - x_min) / 5.0
        y_value = y_min + step * (y_max - y_min) / 5.0
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_bottom - step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<text class="small" x="{x:.2f}" y="{plot_bottom + 24}">{format_float(x_value)}</text>')
        lines.append(f'<text class="small" x="10" y="{y + 4:.2f}">{format_float(y_value)}</text>')

    lines.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.2f}" y="{height - 24}">PC1</text>')
    lines.append(f'<text class="label" transform="translate(24 {(plot_top + plot_bottom) / 2:.2f}) rotate(-90)">PC2</text>')

    legend_y = 110
    for group in groups:
        lines.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="14" height="14" fill="{group.color}"/>')
        lines.append(f'<text class="small" x="{width - 158}" y="{legend_y + 2}">{group.label} ({len(group.paths)})</text>')
        legend_y += 24

    output_path.write_text(finish_svg(lines))


def write_singular_values_svg(output_path: Path, group_stats: list[dict[str, object]]) -> None:
    width = 900
    height = 620
    margin_left = 80
    margin_right = 220
    margin_top = 60
    margin_bottom = 80
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    max_rank = max(len(stats["singular_values"]) for stats in group_stats)
    max_value = max(float(np.max(stats["singular_values"])) for stats in group_stats)

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{margin_left}" y="32">Singular Value Spectrum</text>')
    lines.append(f'<text class="small" x="{margin_left}" y="50">Sharper drop and lower effective rank usually mean less diversity in parameter space.</text>')

    for step in range(6):
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_top + step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        lines.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')

    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')

    for stats in group_stats:
        singular_values: np.ndarray = stats["singular_values"]  # type: ignore[assignment]
        color = str(stats["color"])
        points: list[str] = []
        for index, value in enumerate(singular_values, start=1):
            x = project(float(index), 1.0, float(max_rank), plot_left, plot_right)
            y = project(float(value), 0.0, max_value, plot_bottom, plot_top)
            points.append(f"{x:.2f} {y:.2f}")
        if points:
            lines.append(f'<path class="line" d="M {" L ".join(points)}" stroke="{color}"/>')

    for step in range(6):
        rank_value = 1.0 + step * max(0.0, float(max_rank - 1)) / 5.0
        singular_value = step * max_value / 5.0
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_bottom - step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<text class="small" x="{x:.2f}" y="{plot_bottom + 24}">{int(round(rank_value))}</text>')
        lines.append(f'<text class="small" x="10" y="{y + 4:.2f}">{format_float(singular_value)}</text>')

    lines.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.2f}" y="{height - 24}">Component Index</text>')
    lines.append(f'<text class="label" transform="translate(24 {(plot_top + plot_bottom) / 2:.2f}) rotate(-90)">Singular Value</text>')

    legend_y = 110
    for stats in group_stats:
        lines.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="14" height="14" fill="{stats["color"]}"/>')
        lines.append(f'<text class="small" x="{width - 158}" y="{legend_y + 2}">{stats["label"]} (rank {float(stats["effective_rank"]):.2f})</text>')
        legend_y += 24

    output_path.write_text(finish_svg(lines))


def histogram(values: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=np.float32), np.array([0], dtype=np.int32)
    counts, edges = np.histogram(values, bins=bins)
    return edges.astype(np.float32), counts.astype(np.int32)


def write_distance_histogram_svg(output_path: Path, group_stats: list[dict[str, object]]) -> None:
    width = 900
    height = 620
    margin_left = 80
    margin_right = 220
    margin_top = 60
    margin_bottom = 80
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    histograms = [histogram(np.asarray(stats["pairwise_distances"], dtype=np.float32)) for stats in group_stats]
    max_count = max(int(np.max(counts)) for _, counts in histograms)
    x_max = max(float(edges[-1]) for edges, _ in histograms)

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="{margin_left}" y="32">Pairwise Distance Histogram</text>')
    lines.append(f'<text class="small" x="{margin_left}" y="50">If predicted instruments cluster tightly, their distance histogram hugs the left side.</text>')

    for step in range(6):
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_top + step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        lines.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')

    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')

    for stats, (edges, counts) in zip(group_stats, histograms, strict=True):
        color = str(stats["color"])
        for index, count in enumerate(counts):
            x0 = project(float(edges[index]), 0.0, x_max, plot_left, plot_right)
            x1 = project(float(edges[index + 1]), 0.0, x_max, plot_left, plot_right)
            y = project(float(count), 0.0, float(max_count), plot_bottom, plot_top)
            width_rect = max(1.0, x1 - x0 - 1.5)
            lines.append(
                f'<rect x="{x0 + 0.75:.2f}" y="{y:.2f}" width="{width_rect:.2f}" height="{plot_bottom - y:.2f}" fill="{color}" fill-opacity="0.30" stroke="{color}" stroke-width="1"/>'
            )

    for step in range(6):
        distance_value = step * x_max / 5.0
        count_value = step * max_count / 5.0
        x = plot_left + step * (plot_right - plot_left) / 5.0
        y = plot_bottom - step * (plot_bottom - plot_top) / 5.0
        lines.append(f'<text class="small" x="{x:.2f}" y="{plot_bottom + 24}">{format_float(distance_value)}</text>')
        lines.append(f'<text class="small" x="10" y="{y + 4:.2f}">{int(round(count_value))}</text>')

    lines.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.2f}" y="{height - 24}">Pairwise L2 Distance</text>')
    lines.append(f'<text class="label" transform="translate(24 {(plot_top + plot_bottom) / 2:.2f}) rotate(-90)">Pair Count</text>')

    legend_y = 110
    for stats in group_stats:
        lines.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="14" height="14" fill="{stats["color"]}"/>')
        lines.append(f'<text class="small" x="{width - 158}" y="{legend_y + 2}">{stats["label"]} mean {float(stats["mean_pairwise_distance"]):.3f}</text>')
        legend_y += 24

    output_path.write_text(finish_svg(lines))


def write_parameter_variance_csv(output_path: Path, groups: list[GroupVectors], max_oscillators: int) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "parameter", "mean", "stddev"])
        for group in groups:
            reshaped = group.matrix.reshape(group.matrix.shape[0], max_oscillators, TARGET_CHANNELS)
            for parameter_index, name in enumerate(PARAMETER_NAMES):
                parameter_values = reshaped[:, :, parameter_index]
                writer.writerow(
                    [
                        group.name,
                        name,
                        f"{float(np.mean(parameter_values)):.6f}",
                        f"{float(np.std(parameter_values)):.6f}",
                    ]
                )


def write_summary_json(output_path: Path, group_stats: list[dict[str, object]]) -> None:
    serializable: dict[str, dict[str, object]] = {}
    for stats in group_stats:
        serializable[str(stats["name"])] = {
            "label": stats["label"],
            "count": stats["count"],
            "effective_rank": round(float(stats["effective_rank"]), 6),
            "mean_pairwise_distance": round(float(stats["mean_pairwise_distance"]), 6),
            "median_pairwise_distance": round(float(stats["median_pairwise_distance"]), 6),
            "explained_variance_first_two": [
                round(float(value), 6)
                for value in np.asarray(stats["explained_variance"], dtype=np.float32)[:2].tolist()
            ],
        }
    output_path.write_text(json.dumps(serializable, indent=2))


def write_report_markdown(output_path: Path, group_stats: list[dict[str, object]]) -> None:
    lines = ["# Parameter Space Report", ""]
    lines.append("This report compares flattened oscillator vectors across groups. Lower effective rank and much smaller pairwise distances are the fingerprints of collapse toward one average solution.")
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    lines.append("| Group | Count | Effective rank | Mean pairwise distance | Median pairwise distance | PC1+PC2 variance |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for stats in group_stats:
        explained = np.asarray(stats["explained_variance"], dtype=np.float32)
        lines.append(
            f"| {stats['label']} | {stats['count']} | {float(stats['effective_rank']):.2f} | {float(stats['mean_pairwise_distance']):.4f} | {float(stats['median_pairwise_distance']):.4f} | {float(explained[:2].sum()):.4f} |"
        )
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append("![PCA Scatter](scatter.svg)")
    lines.append("")
    lines.append("![Singular Values](singular_values.svg)")
    lines.append("")
    lines.append("![Pairwise Distances](pairwise_distance_histogram.svg)")
    lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    groups = gather_groups(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coordinates = compute_pca(groups)
    write_coordinates_csv(args.output_dir / "pc_coordinates.csv", groups, coordinates)

    group_stats: list[dict[str, object]] = []
    for group in groups:
        singular_values, explained_variance, effective_rank = singular_value_stats(group.matrix)
        pairwise_distances = compute_pairwise_distances(group.matrix)
        mean_pairwise_distance = float(np.mean(pairwise_distances)) if pairwise_distances.size else 0.0
        median_pairwise_distance = float(np.median(pairwise_distances)) if pairwise_distances.size else 0.0
        group_stats.append(
            {
                "name": group.name,
                "label": group.label,
                "color": group.color,
                "count": len(group.paths),
                "singular_values": singular_values,
                "explained_variance": explained_variance,
                "effective_rank": effective_rank,
                "pairwise_distances": pairwise_distances,
                "mean_pairwise_distance": mean_pairwise_distance,
                "median_pairwise_distance": median_pairwise_distance,
            }
        )

    write_scatter_svg(args.output_dir / "scatter.svg", groups, coordinates)
    write_singular_values_svg(args.output_dir / "singular_values.svg", group_stats)
    write_distance_histogram_svg(args.output_dir / "pairwise_distance_histogram.svg", group_stats)
    write_parameter_variance_csv(args.output_dir / "parameter_variance.csv", groups, args.max_oscillators)
    write_summary_json(args.output_dir / "summary.json", group_stats)
    write_report_markdown(args.output_dir / "report.md", group_stats)
    print(f"Wrote parameter-space analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

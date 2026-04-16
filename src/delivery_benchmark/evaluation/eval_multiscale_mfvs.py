#!/usr/bin/env python3
"""Offline multi-scale comparison for baseline, MFVS, and MFVS ablations.

This script evaluates three map scales (small / medium / large) by default and
produces:
1) Detailed per-building CSV.
2) Per-map summary CSV.
3) Overall summary CSV.
4) JSON summary.
5) Two figures for map-level and overall ablation comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_pkg import CoordinateTransformer, OSMHandler
from compare_entrance_planners import (
    build_boundary_geometry,
    calculate_vertex_saliency,
    calculate_yaw,
    cost_to_target,
    deduplicate_candidates,
    find_target_index,
    inflate_building,
    order_baseline,
    path_cost,
    point_on_boundary,
    score_candidate,
)


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MAPS = {
    "small": "src/utils_pkg/resource/osm/small.osm",
    "medium": "src/utils_pkg/resource/osm/medium.osm",
    "large": "src/utils_pkg/resource/osm/large.osm",
}

DEFAULT_WEIGHTS = {
    "distance": 0.45,
    "saliency": 0.32,
    "coverage": 0.10,
    "novelty": 0.05,
    "heading": 0.04,
    "corner": 0.04,
}

VARIANTS = [
    {
        "name": "baseline_fixed_order",
        "use_corner_augmentation": False,
        "enable_dynamic_scoring": False,
        "enable_coverage_novelty": False,
        "enable_pruning": False,
    },
    {
        "name": "mfvs_full",
        "use_corner_augmentation": True,
        "enable_dynamic_scoring": True,
        "enable_coverage_novelty": True,
        "enable_pruning": True,
    },
    {
        "name": "ablate_corner_augmentation",
        "use_corner_augmentation": False,
        "enable_dynamic_scoring": True,
        "enable_coverage_novelty": True,
        "enable_pruning": True,
    },
    {
        "name": "ablate_dynamic_scoring",
        "use_corner_augmentation": True,
        "enable_dynamic_scoring": False,
        "enable_coverage_novelty": True,
        "enable_pruning": True,
    },
    {
        "name": "ablate_coverage_novelty",
        "use_corner_augmentation": True,
        "enable_dynamic_scoring": True,
        "enable_coverage_novelty": False,
        "enable_pruning": True,
    },
    {
        "name": "ablate_pruning",
        "use_corner_augmentation": True,
        "enable_dynamic_scoring": True,
        "enable_coverage_novelty": True,
        "enable_pruning": False,
    },
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path

    if path.exists():
        return path.resolve()

    for parent in SCRIPT_DIR.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate.resolve()

    return path.resolve()


def build_candidate_pool_variant(
    inflated_building,
    sample_distance,
    exploration_radius,
    corner_saliency_threshold,
    candidate_offset_fraction,
    use_corner_augmentation,
):
    polygon_points, closed_points, segment_lengths, cumulative_lengths = build_boundary_geometry(inflated_building)
    perimeter = cumulative_lengths[-1]
    if perimeter == 0.0:
        return []

    base_step = max(sample_distance, 1e-6)
    fine_step = max(base_step * 0.5, exploration_radius)
    candidate_distances = list(np.arange(0.0, perimeter, base_step))
    candidate_distances += list(np.arange(base_step * 0.5, perimeter, fine_step))

    center = np.mean(np.asarray(inflated_building), axis=0)
    vertex_saliency = calculate_vertex_saliency(inflated_building)
    candidates = []

    for target_distance in candidate_distances:
        waypoint, arc_length, segment_index = point_on_boundary(
            closed_points,
            segment_lengths,
            cumulative_lengths,
            target_distance,
        )
        yaw = calculate_yaw(waypoint, center)
        nearest_vertex_index = min(
            range(len(polygon_points)),
            key=lambda index: np.linalg.norm(
                np.asarray(polygon_points[index], dtype=float) - np.asarray(waypoint, dtype=float)
            ),
        )
        saliency = vertex_saliency[nearest_vertex_index] if vertex_saliency else 0.0
        candidates.append(
            {
                "pose": (float(waypoint[0]), float(waypoint[1]), float(yaw)),
                "saliency": float(saliency),
                "arc_length": float(arc_length),
                "segment_length": float(
                    segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0
                ),
                "corner_focus": bool(saliency >= corner_saliency_threshold),
            }
        )

    if use_corner_augmentation and candidates:
        salient_vertices = [
            index for index, value in enumerate(vertex_saliency) if value >= corner_saliency_threshold
        ]
        for vertex_index in salient_vertices:
            vertex_arc = cumulative_lengths[min(vertex_index, len(cumulative_lengths) - 2)]
            for offset_ratio in (-candidate_offset_fraction, 0.0, candidate_offset_fraction):
                waypoint, arc_length, segment_index = point_on_boundary(
                    closed_points,
                    segment_lengths,
                    cumulative_lengths,
                    vertex_arc + offset_ratio * base_step,
                )
                yaw = calculate_yaw(waypoint, center)
                candidates.append(
                    {
                        "pose": (float(waypoint[0]), float(waypoint[1]), float(yaw)),
                        "saliency": float(vertex_saliency[vertex_index]),
                        "arc_length": float(arc_length),
                        "segment_length": float(
                            segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0
                        ),
                        "corner_focus": True,
                    }
                )

    return deduplicate_candidates(candidates, exploration_radius)


def select_dynamic_order_variant(
    candidates,
    robot_position,
    robot_heading,
    perimeter_length,
    weights,
    coverage_radius,
    enable_coverage_novelty,
    enable_pruning,
):
    remaining = list(candidates)
    ordered = []
    visited_positions = []
    visited_arc_positions = []

    while remaining:
        best_index = 0
        best_score = None

        for index, candidate in enumerate(remaining):
            current_weights = dict(weights)
            if not enable_coverage_novelty:
                current_weights["coverage"] = 0.0
                current_weights["novelty"] = 0.0

            candidate_score = score_candidate(
                candidate=candidate,
                robot_position=robot_position,
                robot_heading=robot_heading,
                visited_positions=visited_positions,
                visited_arc_positions=visited_arc_positions,
                perimeter_length=perimeter_length,
                weights=current_weights,
                coverage_radius=coverage_radius,
            )
            if best_score is None or candidate_score > best_score:
                best_index = index
                best_score = candidate_score

        selected = remaining.pop(best_index)
        ordered.append(selected)
        visited_positions.append(selected["pose"][:2])
        visited_arc_positions.append(selected["arc_length"])

        if enable_pruning:
            remaining = [
                candidate
                for candidate in remaining
                if math.sqrt(
                    (candidate["pose"][0] - selected["pose"][0]) ** 2
                    + (candidate["pose"][1] - selected["pose"][1]) ** 2
                )
                >= coverage_radius * 0.75
            ]

        robot_position = selected["pose"][:2]
        robot_heading = selected["pose"][2]

    return ordered


def evaluate_variant_on_building(
    handler,
    transform_matrix,
    building_way,
    robot_position,
    sample_distance,
    exploration_radius,
    coverage_radius,
    corner_saliency_threshold,
    candidate_offset_fraction,
    weights,
    use_corner_augmentation,
    enable_dynamic_scoring,
    enable_coverage_novelty,
    enable_pruning,
    target_pose_reference,
):
    coordinates = handler.get_way_nodes_locations(building_way["id"])
    if len(coordinates) < 3:
        return None

    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + [coordinates[0]]

    utm_coordinates = []
    for lon, lat in coordinates:
        easting, northing, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
        utm_coordinates.append((easting, northing))

    local_coordinates = [
        tuple(np.dot(transform_matrix, np.array([x, y, 1.0]))[:2]) for x, y in utm_coordinates
    ]
    inflated_building = inflate_building(local_coordinates, exploration_radius)

    candidates = build_candidate_pool_variant(
        inflated_building=inflated_building,
        sample_distance=sample_distance,
        exploration_radius=coverage_radius,
        corner_saliency_threshold=corner_saliency_threshold,
        candidate_offset_fraction=candidate_offset_fraction,
        use_corner_augmentation=use_corner_augmentation,
    )
    if not candidates:
        return None

    perimeter_length = max(candidate["arc_length"] for candidate in candidates)

    if enable_dynamic_scoring:
        order = select_dynamic_order_variant(
            candidates=candidates,
            robot_position=robot_position,
            robot_heading=0.0,
            perimeter_length=perimeter_length,
            weights=weights,
            coverage_radius=coverage_radius,
            enable_coverage_novelty=enable_coverage_novelty,
            enable_pruning=enable_pruning,
        )
    else:
        order = order_baseline(candidates, robot_position)

    target_index = find_target_index(order, target_pose_reference, coverage_radius)
    target_cost, target_steps = cost_to_target(order, robot_position, target_index)
    full_cost = path_cost(order, robot_position)

    return {
        "building_name": building_way["tags"].get("name", f"building_{building_way['id']}"),
        "candidate_count": len(candidates),
        "target_cost": target_cost,
        "target_steps": target_steps,
        "full_cost": full_cost,
        "target_saliency": float(max(candidate["saliency"] for candidate in candidates)),
    }


def summarize_frame(frame: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            count=("building_name", "count"),
            mean_target_cost=("target_cost", "mean"),
            mean_full_cost=("full_cost", "mean"),
            mean_target_steps=("target_steps", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
        )
        .reset_index()
    )
    return grouped


def add_baseline_delta(summary_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    merged_frames = []
    for group_value, group_frame in summary_frame.groupby(group_column):
        baseline_rows = group_frame[group_frame["variant"] == "baseline_fixed_order"]
        if baseline_rows.empty:
            merged_frames.append(group_frame)
            continue

        baseline_target = float(baseline_rows.iloc[0]["mean_target_cost"])
        baseline_full = float(baseline_rows.iloc[0]["mean_full_cost"])

        group_frame = group_frame.copy()
        group_frame["target_cost_delta_vs_baseline"] = group_frame["mean_target_cost"] - baseline_target
        group_frame["full_cost_delta_vs_baseline"] = group_frame["mean_full_cost"] - baseline_full
        merged_frames.append(group_frame)

    return pd.concat(merged_frames, ignore_index=True) if merged_frames else summary_frame


def plot_overall(summary_overall: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_frame = summary_overall.copy()
    order = [variant["name"] for variant in VARIANTS]
    plot_frame["variant"] = pd.Categorical(plot_frame["variant"], categories=order, ordered=True)
    plot_frame = plot_frame.sort_values("variant")

    x = np.arange(len(plot_frame))
    width = 0.36

    baseline_rows = plot_frame[plot_frame["variant"] == "baseline_fixed_order"]
    baseline_target = float(baseline_rows.iloc[0]["mean_target_cost"]) if not baseline_rows.empty else 0.0
    baseline_full = float(baseline_rows.iloc[0]["mean_full_cost"]) if not baseline_rows.empty else 0.0
    baseline_steps = float(baseline_rows.iloc[0]["mean_target_steps"]) if not baseline_rows.empty else 0.0
    baseline_candidates = float(baseline_rows.iloc[0]["mean_candidate_count"]) if not baseline_rows.empty else 0.0

    plot_frame["target_delta_vs_baseline"] = plot_frame["mean_target_cost"] - baseline_target
    plot_frame["full_delta_vs_baseline"] = plot_frame["mean_full_cost"] - baseline_full
    plot_frame["steps_delta_vs_baseline"] = plot_frame["mean_target_steps"] - baseline_steps
    plot_frame["candidates_delta_vs_baseline"] = plot_frame["mean_candidate_count"] - baseline_candidates

    def annotate_bars(axis, bars, span):
        for bar in bars:
            value = bar.get_height()
            offset = max(2.0, span * 0.01)
            axis.annotate(
                f"{value:+.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset if value >= 0 else -offset * 1.6),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                clip_on=True,
            )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    bars_t = axes[0].bar(
        x - width / 2,
        plot_frame["target_delta_vs_baseline"],
        width=width,
        label="Target cost delta",
        color="#2E86AB",
    )
    bars_f = axes[0].bar(
        x + width / 2,
        plot_frame["full_delta_vs_baseline"],
        width=width,
        label="Full cost delta",
        color="#1ABC9C",
    )
    top_values = np.concatenate([
        np.asarray(plot_frame["target_delta_vs_baseline"], dtype=float),
        np.asarray(plot_frame["full_delta_vs_baseline"], dtype=float),
    ])
    top_min = float(np.min(top_values)) if len(top_values) else 0.0
    top_max = float(np.max(top_values)) if len(top_values) else 0.0
    top_span = max(top_max - top_min, 1.0)
    axes[0].set_ylim(top_min - 0.25 * top_span, top_max + 0.25 * top_span)
    annotate_bars(axes[0], bars_t, top_span)
    annotate_bars(axes[0], bars_f, top_span)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Delta vs baseline")
    axes[0].set_title("Overall Delta Comparison (baseline = 0)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    axes[0].grid(axis="y", alpha=0.25)

    bars_s = axes[1].bar(
        x - width / 2,
        plot_frame["steps_delta_vs_baseline"],
        width=width,
        label="Target steps delta",
        color="#D35400",
    )
    bars_c = axes[1].bar(
        x + width / 2,
        plot_frame["candidates_delta_vs_baseline"],
        width=width,
        label="Candidate count delta",
        color="#7F8C8D",
    )
    bottom_values = np.concatenate([
        np.asarray(plot_frame["steps_delta_vs_baseline"], dtype=float),
        np.asarray(plot_frame["candidates_delta_vs_baseline"], dtype=float),
    ])
    bottom_min = float(np.min(bottom_values)) if len(bottom_values) else 0.0
    bottom_max = float(np.max(bottom_values)) if len(bottom_values) else 0.0
    bottom_span = max(bottom_max - bottom_min, 1.0)
    axes[1].set_ylim(bottom_min - 0.25 * bottom_span, bottom_max + 0.25 * bottom_span)
    annotate_bars(axes[1], bars_s, bottom_span)
    annotate_bars(axes[1], bars_c, bottom_span)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Delta vs baseline")
    axes[1].set_xlabel("Variant")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_frame["variant"], rotation=30, ha="right")

    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_map(summary_by_map: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = [variant["name"] for variant in VARIANTS]
    maps = list(summary_by_map["map_scale"].unique())

    fig, axes = plt.subplots(len(maps), 1, figsize=(14, 4 * max(1, len(maps))), sharex=True)
    if len(maps) == 1:
        axes = [axes]

    for axis, map_scale in zip(axes, maps):
        map_frame = summary_by_map[summary_by_map["map_scale"] == map_scale].copy()
        map_frame["variant"] = pd.Categorical(map_frame["variant"], categories=order, ordered=True)
        map_frame = map_frame.sort_values("variant")

        baseline_rows = map_frame[map_frame["variant"] == "baseline_fixed_order"]
        baseline_target = float(baseline_rows.iloc[0]["mean_target_cost"]) if not baseline_rows.empty else 0.0
        baseline_full = float(baseline_rows.iloc[0]["mean_full_cost"]) if not baseline_rows.empty else 0.0
        map_frame["target_delta_vs_baseline"] = map_frame["mean_target_cost"] - baseline_target
        map_frame["full_delta_vs_baseline"] = map_frame["mean_full_cost"] - baseline_full

        def annotate_bars(axis_obj, bars, span):
            for bar in bars:
                value = bar.get_height()
                offset = max(2.0, span * 0.01)
                axis_obj.annotate(
                    f"{value:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, offset if value >= 0 else -offset * 1.6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    clip_on=True,
                )

        x = np.arange(len(map_frame))
        width = 0.38

        bars_t = axis.bar(
            x - width / 2,
            map_frame["target_delta_vs_baseline"],
            width=width,
            label="Target cost delta",
            color="#2E86AB",
        )
        bars_f = axis.bar(
            x + width / 2,
            map_frame["full_delta_vs_baseline"],
            width=width,
            label="Full cost delta",
            color="#16A085",
        )
        map_values = np.concatenate([
            np.asarray(map_frame["target_delta_vs_baseline"], dtype=float),
            np.asarray(map_frame["full_delta_vs_baseline"], dtype=float),
        ])
        map_min = float(np.min(map_values)) if len(map_values) else 0.0
        map_max = float(np.max(map_values)) if len(map_values) else 0.0
        map_span = max(map_max - map_min, 1.0)
        axis.set_ylim(map_min - 0.25 * map_span, map_max + 0.25 * map_span)
        annotate_bars(axis, bars_t, map_span)
        annotate_bars(axis, bars_f, map_span)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"Map scale: {map_scale} (baseline = 0)")
        axis.set_ylabel("Delta vs baseline")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        axis.set_xticks(x)
        axis.set_xticklabels(map_frame["variant"], rotation=30, ha="right")

    axes[-1].set_xlabel("Variant")
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline multi-scale comparison for baseline, MFVS full model, and MFVS ablations."
    )
    parser.add_argument("--small-osm", type=str, default=DEFAULT_MAPS["small"])
    parser.add_argument("--medium-osm", type=str, default=DEFAULT_MAPS["medium"])
    parser.add_argument("--large-osm", type=str, default=DEFAULT_MAPS["large"])
    parser.add_argument("--output-dir", type=str, default="src/delivery_benchmark/result/multiscale_eval")
    parser.add_argument("--sample-distance", type=float, default=5.0)
    parser.add_argument("--exploration-radius", type=float, default=2.0)
    parser.add_argument("--coverage-radius", type=float, default=1.2)
    parser.add_argument("--corner-saliency-threshold", type=float, default=0.45)
    parser.add_argument("--candidate-offset-fraction", type=float, default=0.35)
    args = parser.parse_args()

    map_paths = {
        "small": resolve_path(args.small_osm),
        "medium": resolve_path(args.medium_osm),
        "large": resolve_path(args.large_osm),
    }

    for map_scale, map_path in map_paths.items():
        if not map_path.exists():
            raise FileNotFoundError(f"{map_scale} map not found: {map_path}")

    transform_matrix = np.array(
        [
            [1.0, 0.0, -500000.0],
            [0.0, 1.0, -4483000.0],
            [0.0, 0.0, 1.0],
        ]
    )

    rows = []
    weights = dict(DEFAULT_WEIGHTS)

    for map_scale, map_path in map_paths.items():
        handler = OSMHandler()
        handler.apply_file(str(map_path))
        buildings = [
            way for way in handler.ways_info if way.get("type") == "building" and way.get("tags", {}).get("name")
        ]

        for building_way in buildings:
            coordinates = handler.get_way_nodes_locations(building_way["id"])
            if len(coordinates) < 3:
                continue

            if coordinates[0] != coordinates[-1]:
                coordinates = coordinates + [coordinates[0]]

            utm_coordinates = []
            for lon, lat in coordinates:
                easting, northing, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
                utm_coordinates.append((easting, northing))

            local_coordinates = [
                tuple(np.dot(transform_matrix, np.array([x, y, 1.0]))[:2]) for x, y in utm_coordinates
            ]
            center = np.mean(np.asarray(local_coordinates), axis=0)
            robot_position = (float(center[0] - 15.0), float(center[1]))

            reference_candidates = build_candidate_pool_variant(
                inflated_building=inflate_building(local_coordinates, args.exploration_radius),
                sample_distance=args.sample_distance,
                exploration_radius=args.coverage_radius,
                corner_saliency_threshold=args.corner_saliency_threshold,
                candidate_offset_fraction=args.candidate_offset_fraction,
                use_corner_augmentation=True,
            )
            if not reference_candidates:
                continue

            target_reference_candidate = max(reference_candidates, key=lambda candidate: candidate["saliency"])

            for variant in VARIANTS:
                result = evaluate_variant_on_building(
                    handler=handler,
                    transform_matrix=transform_matrix,
                    building_way=building_way,
                    robot_position=robot_position,
                    sample_distance=args.sample_distance,
                    exploration_radius=args.exploration_radius,
                    coverage_radius=args.coverage_radius,
                    corner_saliency_threshold=args.corner_saliency_threshold,
                    candidate_offset_fraction=args.candidate_offset_fraction,
                    weights=weights,
                    use_corner_augmentation=variant["use_corner_augmentation"],
                    enable_dynamic_scoring=variant["enable_dynamic_scoring"],
                    enable_coverage_novelty=variant["enable_coverage_novelty"],
                    enable_pruning=variant["enable_pruning"],
                    target_pose_reference=target_reference_candidate["pose"],
                )
                if result is None:
                    continue

                rows.append(
                    {
                        "map_scale": map_scale,
                        "map_file": str(map_path),
                        "variant": variant["name"],
                        "target_reference_saliency": float(target_reference_candidate["saliency"]),
                        **result,
                    }
                )

    if not rows:
        raise RuntimeError("No valid evaluation rows were produced.")

    detailed_frame = pd.DataFrame(rows)
    summary_by_map = summarize_frame(detailed_frame, ["map_scale", "variant"])
    summary_by_map = add_baseline_delta(summary_by_map, "map_scale")

    summary_overall = summarize_frame(detailed_frame, ["variant"])
    summary_overall = add_baseline_delta(summary_overall.assign(scope="overall"), "scope")
    summary_overall = summary_overall.drop(columns=["scope"])

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = output_dir / "multiscale_detailed.csv"
    map_summary_csv = output_dir / "multiscale_map_summary.csv"
    overall_summary_csv = output_dir / "multiscale_overall_summary.csv"
    summary_json = output_dir / "multiscale_summary.json"
    overall_plot = output_dir / "multiscale_overall.png"
    per_map_plot = output_dir / "multiscale_per_map.png"

    detailed_frame.to_csv(detailed_csv, index=False)
    summary_by_map.to_csv(map_summary_csv, index=False)
    summary_overall.to_csv(overall_summary_csv, index=False)

    payload = {
        "experiment": {
            "maps": {key: str(value) for key, value in map_paths.items()},
            "sample_distance": args.sample_distance,
            "exploration_radius": args.exploration_radius,
            "coverage_radius": args.coverage_radius,
            "corner_saliency_threshold": args.corner_saliency_threshold,
            "candidate_offset_fraction": args.candidate_offset_fraction,
        },
        "variants": [variant["name"] for variant in VARIANTS],
        "summary_by_map": summary_by_map.to_dict(orient="records"),
        "summary_overall": summary_overall.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    plot_overall(summary_overall, overall_plot)
    plot_per_map(summary_by_map, per_map_plot)

    print(f"Detailed CSV: {detailed_csv}")
    print(f"Map summary CSV: {map_summary_csv}")
    print(f"Overall summary CSV: {overall_summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Overall plot: {overall_plot}")
    print(f"Per-map plot: {per_map_plot}")


if __name__ == "__main__":
    main()

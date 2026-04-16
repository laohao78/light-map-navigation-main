import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyclipper

from utils_pkg import CoordinateTransformer, OSMHandler
from compare_entrance_planners import (
    angular_distance,
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
from result_plots import plot_eval3_summary


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path

    if path.exists():
        return path.resolve()

    for parent in SCRIPT_DIR.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate

    return path


BASELINE_WEIGHTS = {
    'distance': 0.45,
    'saliency': 0.32,
    'coverage': 0.10,
    'novelty': 0.05,
    'heading': 0.04,
    'corner': 0.04,
}


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
        candidates.append({
            'pose': (float(waypoint[0]), float(waypoint[1]), float(yaw)),
            'saliency': float(saliency),
            'arc_length': float(arc_length),
            'segment_length': float(segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0),
            'corner_focus': bool(saliency >= corner_saliency_threshold),
        })

    if use_corner_augmentation and candidates:
        salient_vertices = [index for index, value in enumerate(vertex_saliency) if value >= corner_saliency_threshold]
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
                candidates.append({
                    'pose': (float(waypoint[0]), float(waypoint[1]), float(yaw)),
                    'saliency': float(vertex_saliency[vertex_index]),
                    'arc_length': float(arc_length),
                    'segment_length': float(segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0),
                    'corner_focus': True,
                })

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
            candidate_weights = dict(weights)
            if not enable_coverage_novelty:
                candidate_weights['coverage'] = 0.0
                candidate_weights['novelty'] = 0.0

            candidate_score = score_candidate(
                candidate=candidate,
                robot_position=robot_position,
                robot_heading=robot_heading,
                visited_positions=visited_positions,
                visited_arc_positions=visited_arc_positions,
                perimeter_length=perimeter_length,
                weights=candidate_weights,
                coverage_radius=coverage_radius,
            )
            if best_score is None or candidate_score > best_score:
                best_index = index
                best_score = candidate_score

        selected = remaining.pop(best_index)
        ordered.append(selected)
        visited_positions.append(selected['pose'][:2])
        visited_arc_positions.append(selected['arc_length'])

        if enable_pruning:
            remaining = [
                candidate for candidate in remaining
                if math.sqrt((candidate['pose'][0] - selected['pose'][0]) ** 2 + (candidate['pose'][1] - selected['pose'][1]) ** 2) >= coverage_radius * 0.75
            ]

        robot_position = selected['pose'][:2]
        robot_heading = selected['pose'][2]

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
    coordinates = handler.get_way_nodes_locations(building_way['id'])
    if len(coordinates) < 3:
        return None

    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + [coordinates[0]]

    utm_coordinates = []
    for lon, lat in coordinates:
        easting, northing, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
        utm_coordinates.append((easting, northing))

    local_coordinates = [tuple(np.dot(transform_matrix, np.array([x, y, 1.0]))[:2]) for x, y in utm_coordinates]
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

    perimeter_length = max(candidate['arc_length'] for candidate in candidates)

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
        'building_name': building_way['tags'].get('name', f"building_{building_way['id']}"),
        'candidate_count': len(candidates),
        'target_cost': target_cost,
        'target_steps': target_steps,
        'full_cost': full_cost,
        'target_saliency': float(max(candidate['saliency'] for candidate in candidates)),
    }


def summarize_variant(records):
    if not records:
        return {}

    return {
        'count': len(records),
        'mean_target_cost': float(np.mean([record['target_cost'] for record in records])),
        'mean_full_cost': float(np.mean([record['full_cost'] for record in records])),
        'mean_target_steps': float(np.mean([record['target_steps'] for record in records])),
        'mean_candidate_count': float(np.mean([record['candidate_count'] for record in records])),
    }


def evaluate_suite(records_by_variant):
    summary = {}
    for variant_name, records in records_by_variant.items():
        summary[variant_name] = summarize_variant(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for entrance exploration planner.')
    parser.add_argument('--osm-file', type=str, default='src/utils_pkg/resource/osm/medium.osm')
    parser.add_argument('--output-csv', type=str, default='src/delivery_benchmark/result/entrance_ablation_comparison.csv')
    parser.add_argument('--output-json', type=str, default='src/delivery_benchmark/result/entrance_ablation_summary.json')
    parser.add_argument('--plot-out', type=str, default='src/delivery_benchmark/result/entrance_ablation_summary.png')
    parser.add_argument('--sample-distance', type=float, default=5.0)
    parser.add_argument('--exploration-radius', type=float, default=2.0)
    parser.add_argument('--coverage-radius', type=float, default=1.2)
    parser.add_argument('--corner-saliency-threshold', type=float, default=0.45)
    parser.add_argument('--candidate-offset-fraction', type=float, default=0.35)
    args = parser.parse_args()

    osm_path = resolve_path(args.osm_file)
    if not osm_path.exists():
        raise FileNotFoundError(f'OSM file not found: {osm_path}')

    handler = OSMHandler()
    handler.apply_file(str(osm_path))

    transform_matrix = np.array([
        [1.0, 0.0, -500000.0],
        [0.0, 1.0, -4483000.0],
        [0.0, 0.0, 1.0],
    ])

    buildings = [
        way for way in handler.ways_info
        if way.get('type') == 'building' and way.get('tags', {}).get('name')
    ]

    variants = [
        {
            'name': 'baseline_fixed_order',
            'use_corner_augmentation': False,
            'enable_dynamic_scoring': False,
            'enable_coverage_novelty': False,
            'enable_pruning': False,
        },
        {
            'name': 'plus_corner_augmentation',
            'use_corner_augmentation': True,
            'enable_dynamic_scoring': False,
            'enable_coverage_novelty': False,
            'enable_pruning': False,
        },
        {
            'name': 'plus_dynamic_scoring',
            'use_corner_augmentation': True,
            'enable_dynamic_scoring': True,
            'enable_coverage_novelty': False,
            'enable_pruning': False,
        },
        {
            'name': 'plus_coverage_novelty',
            'use_corner_augmentation': True,
            'enable_dynamic_scoring': True,
            'enable_coverage_novelty': True,
            'enable_pruning': False,
        },
        {
            'name': 'full_model_with_pruning',
            'use_corner_augmentation': True,
            'enable_dynamic_scoring': True,
            'enable_coverage_novelty': True,
            'enable_pruning': True,
        },
    ]

    weights = dict(BASELINE_WEIGHTS)
    rows = []
    records_by_variant = {variant['name']: [] for variant in variants}

    for building_way in buildings:
        coordinates = handler.get_way_nodes_locations(building_way['id'])
        if len(coordinates) < 3:
            continue

        if coordinates[0] != coordinates[-1]:
            coordinates = coordinates + [coordinates[0]]

        utm_coordinates = []
        for lon, lat in coordinates:
            easting, northing, _ = CoordinateTransformer.wgs84_to_utm(lon, lat)
            utm_coordinates.append((easting, northing))

        local_coordinates = [tuple(np.dot(transform_matrix, np.array([x, y, 1.0]))[:2]) for x, y in utm_coordinates]
        center = np.mean(np.asarray(local_coordinates), axis=0)
        robot_position = (float(center[0] - 15.0), float(center[1]))

        full_reference_candidates = build_candidate_pool_variant(
            inflated_building=inflate_building(local_coordinates, args.exploration_radius),
            sample_distance=args.sample_distance,
            exploration_radius=args.coverage_radius,
            corner_saliency_threshold=args.corner_saliency_threshold,
            candidate_offset_fraction=args.candidate_offset_fraction,
            use_corner_augmentation=True,
        )
        if not full_reference_candidates:
            continue

        target_reference_candidate = max(full_reference_candidates, key=lambda candidate: candidate['saliency'])

        for variant in variants:
            record = evaluate_variant_on_building(
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
                use_corner_augmentation=variant['use_corner_augmentation'],
                enable_dynamic_scoring=variant['enable_dynamic_scoring'],
                enable_coverage_novelty=variant['enable_coverage_novelty'],
                enable_pruning=variant['enable_pruning'],
                target_pose_reference=target_reference_candidate['pose'],
            )
            if record is None:
                continue

            record['variant'] = variant['name']
            record['target_reference_saliency'] = float(target_reference_candidate['saliency'])
            records_by_variant[variant['name']].append(record)
            rows.append(record)

    if not rows:
        raise RuntimeError('No valid ablation records were produced')

    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    summary = evaluate_suite(records_by_variant)
    summary['experiment'] = {
        'osm_file': str(osm_path),
        'sample_distance': args.sample_distance,
        'exploration_radius': args.exploration_radius,
        'coverage_radius': args.coverage_radius,
        'corner_saliency_threshold': args.corner_saliency_threshold,
        'candidate_offset_fraction': args.candidate_offset_fraction,
    }

    plot_eval3_summary(summary, resolve_path(args.plot_out))

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    print(f'Ablation CSV saved to: {output_csv}')
    print(f'Ablation summary saved to: {output_json}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
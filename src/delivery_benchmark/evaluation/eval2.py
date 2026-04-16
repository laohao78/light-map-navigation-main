import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyclipper

from utils_pkg import CoordinateTransformer, OSMHandler
from result_plots import plot_eval2_summary


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


def inflate_building(coordinates, offset_distance):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(coordinates, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    inflated_polygon = pco.Execute(offset_distance)
    return inflated_polygon[0] if inflated_polygon else coordinates


def sample_waypoints_evenly(coordinates, distance):
    if len(coordinates) < 2:
        return [tuple(point) for point in coordinates]

    points = [np.asarray(point, dtype=float) for point in coordinates]
    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    if len(points) < 2:
        return [tuple(point) for point in points]

    closed_points = points + [points[0]]
    segment_lengths = []
    cumulative_lengths = [0.0]

    for index in range(len(points)):
        segment_length = float(np.linalg.norm(closed_points[index + 1] - closed_points[index]))
        segment_lengths.append(segment_length)
        cumulative_lengths.append(cumulative_lengths[-1] + segment_length)

    perimeter = cumulative_lengths[-1]
    if perimeter == 0.0:
        return [tuple(points[0])]

    sample_count = max(1, int(np.ceil(perimeter / max(distance, 1e-6))))
    sampled_waypoints = []

    for sample_index in range(sample_count):
        target_distance = sample_index * distance
        if target_distance >= perimeter:
            target_distance = perimeter - 1e-6

        segment_index = 0
        while segment_index < len(segment_lengths) - 1 and cumulative_lengths[segment_index + 1] < target_distance:
            segment_index += 1

        segment_start = closed_points[segment_index]
        segment_end = closed_points[segment_index + 1]
        segment_length = segment_lengths[segment_index]

        if segment_length == 0.0:
            sampled_waypoints.append(tuple(segment_start))
            continue

        ratio = (target_distance - cumulative_lengths[segment_index]) / segment_length
        ratio = np.clip(ratio, 0.0, 1.0)
        sampled_point = segment_start + ratio * (segment_end - segment_start)
        sampled_waypoints.append(tuple(sampled_point))

    return sampled_waypoints


def calculate_yaw(waypoint, target):
    vector_to_target = np.array(target) - np.array(waypoint)
    return math.atan2(vector_to_target[1], vector_to_target[0])


def calculate_vertex_saliency(coordinates):
    if len(coordinates) < 3:
        return [0.0 for _ in coordinates]

    points = [np.asarray(point, dtype=float) for point in coordinates]
    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    saliency = []
    for index in range(len(points)):
        prev_point = points[index - 1]
        current_point = points[index]
        next_point = points[(index + 1) % len(points)]

        vec_prev = prev_point - current_point
        vec_next = next_point - current_point
        prev_norm = np.linalg.norm(vec_prev)
        next_norm = np.linalg.norm(vec_next)

        if prev_norm == 0.0 or next_norm == 0.0:
            saliency.append(0.0)
            continue

        cosine = np.clip(np.dot(vec_prev, vec_next) / (prev_norm * next_norm), -1.0, 1.0)
        interior_angle = np.arccos(cosine)
        saliency.append(float(np.pi - interior_angle))

    max_saliency = max(saliency) if saliency else 0.0
    if max_saliency <= 0.0:
        return [0.0 for _ in saliency]

    return [value / max_saliency for value in saliency]


def reorder_waypoints(values, start_index):
    return values[start_index:] + values[:start_index]


def reverse_keep_start(values):
    if len(values) <= 1:
        return list(values)
    return [values[0]] + list(reversed(values[1:]))


def calculate_prefix_cost(waypoints, robot_position, saliency, horizon=3, saliency_weight=0.15, exploration_radius=2.0):
    if not waypoints:
        return float('inf')

    total_cost = 0.0
    current_position = robot_position
    prefix_length = min(horizon, len(waypoints))

    for index in range(prefix_length):
        waypoint = waypoints[index]
        total_cost += math.sqrt((waypoint[0] - current_position[0]) ** 2 + (waypoint[1] - current_position[1]) ** 2)
        if saliency:
            total_cost -= saliency_weight * saliency[index] * max(exploration_radius, 1.0)
        current_position = (waypoint[0], waypoint[1])

    return total_cost


def angular_distance(angle_a, angle_b):
    delta = (angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi
    return abs(delta)


def build_boundary_geometry(coordinates):
    points = [np.asarray(point, dtype=float) for point in coordinates]
    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    if len(points) < 2:
        return points, points, [], [0.0]

    closed_points = points + [points[0]]
    segment_lengths = []
    cumulative_lengths = [0.0]

    for index in range(len(points)):
        segment_length = float(np.linalg.norm(closed_points[index + 1] - closed_points[index]))
        segment_lengths.append(segment_length)
        cumulative_lengths.append(cumulative_lengths[-1] + segment_length)

    return points, closed_points, segment_lengths, cumulative_lengths


def point_on_boundary(closed_points, segment_lengths, cumulative_lengths, target_distance):
    perimeter = cumulative_lengths[-1]
    if perimeter == 0.0:
        return tuple(closed_points[0]), 0.0, 0

    arc_distance = target_distance % perimeter
    segment_index = 0
    while segment_index < len(segment_lengths) - 1 and cumulative_lengths[segment_index + 1] < arc_distance:
        segment_index += 1

    segment_start = closed_points[segment_index]
    segment_end = closed_points[segment_index + 1]
    segment_length = segment_lengths[segment_index]
    if segment_length == 0.0:
        return tuple(segment_start), arc_distance, segment_index

    ratio = (arc_distance - cumulative_lengths[segment_index]) / segment_length
    ratio = np.clip(ratio, 0.0, 1.0)
    sampled_point = segment_start + ratio * (segment_end - segment_start)
    return tuple(sampled_point), arc_distance, segment_index


def deduplicate_candidates(candidates, coverage_radius):
    deduplicated = []
    for candidate in sorted(candidates, key=lambda item: (-item['saliency'], item['arc_length'])):
        if any(
            math.sqrt((candidate['pose'][0] - existing['pose'][0]) ** 2 + (candidate['pose'][1] - existing['pose'][1]) ** 2) < coverage_radius * 0.45
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return deduplicated


def build_candidate_pool(inflated_building, sample_distance, exploration_radius, corner_saliency_threshold, candidate_offset_fraction):
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
        waypoint, arc_length, segment_index = point_on_boundary(closed_points, segment_lengths, cumulative_lengths, target_distance)
        yaw = calculate_yaw(waypoint, center)
        nearest_vertex_index = min(
            range(len(polygon_points)),
            key=lambda index: np.linalg.norm(np.asarray(polygon_points[index], dtype=float) - np.asarray(waypoint, dtype=float))
        )
        saliency = vertex_saliency[nearest_vertex_index] if vertex_saliency else 0.0
        candidates.append({
            'pose': (float(waypoint[0]), float(waypoint[1]), float(yaw)),
            'saliency': float(saliency),
            'arc_length': float(arc_length),
            'segment_length': float(segment_lengths[segment_index] if segment_index < len(segment_lengths) else 0.0),
            'corner_focus': bool(saliency >= corner_saliency_threshold),
        })

    if candidates:
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


def score_candidate(candidate, robot_position, robot_heading, visited_positions, visited_arc_positions, perimeter_length, weights, coverage_radius):
    candidate_x, candidate_y, candidate_yaw = candidate['pose']
    distance = math.sqrt((candidate_x - robot_position[0]) ** 2 + (candidate_y - robot_position[1]) ** 2)
    distance_score = 1.0 / (1.0 + distance / max(coverage_radius, 1e-6))

    if visited_arc_positions:
        arc_deltas = [abs(candidate['arc_length'] - arc) for arc in visited_arc_positions]
        coverage_gap = min(min(delta, perimeter_length - delta) for delta in arc_deltas) / (0.5 * max(perimeter_length, 1e-6))
        coverage_score = max(0.0, min(coverage_gap, 1.0))
    else:
        coverage_score = 1.0

    if visited_positions:
        nearest_visit = min(math.sqrt((candidate_x - visited[0]) ** 2 + (candidate_y - visited[1]) ** 2) for visited in visited_positions)
        novelty_score = max(0.0, min(nearest_visit / max(coverage_radius, 1e-6), 1.0))
    else:
        novelty_score = 1.0

    heading_error = angular_distance(robot_heading, candidate_yaw)
    heading_score = 1.0 - min(heading_error / np.pi, 1.0)
    corner_score = 1.0 if candidate.get('corner_focus', False) else 0.0
    saliency_score = candidate.get('saliency', 0.0)

    return (
        weights['distance'] * distance_score
        + weights['saliency'] * saliency_score
        + weights['coverage'] * coverage_score
        + weights['novelty'] * novelty_score
        + weights['heading'] * heading_score
        + weights['corner'] * corner_score
    )


def select_dynamic_order(candidates, robot_position, robot_heading, perimeter_length, weights, coverage_radius):
    remaining = list(candidates)
    ordered = []
    visited_positions = []
    visited_arc_positions = []

    while remaining:
        best_index = 0
        best_score = None

        for index, candidate in enumerate(remaining):
            candidate_score = score_candidate(
                candidate=candidate,
                robot_position=robot_position,
                robot_heading=robot_heading,
                visited_positions=visited_positions,
                visited_arc_positions=visited_arc_positions,
                perimeter_length=perimeter_length,
                weights=weights,
                coverage_radius=coverage_radius,
            )
            if best_score is None or candidate_score > best_score:
                best_index = index
                best_score = candidate_score

        selected = remaining.pop(best_index)
        ordered.append(selected)
        visited_positions.append(selected['pose'][:2])
        visited_arc_positions.append(selected['arc_length'])

        remaining = [
            candidate for candidate in remaining
            if math.sqrt((candidate['pose'][0] - selected['pose'][0]) ** 2 + (candidate['pose'][1] - selected['pose'][1]) ** 2) >= coverage_radius * 0.75
        ]

        robot_position = selected['pose'][:2]
        robot_heading = selected['pose'][2]

    return ordered


def order_baseline(candidates, robot_position):
    if not candidates:
        return []

    distances = [math.sqrt((candidate['pose'][0] - robot_position[0]) ** 2 + (candidate['pose'][1] - robot_position[1]) ** 2) for candidate in candidates]
    closest_index = int(np.argmin(distances))
    return candidates[closest_index:] + candidates[:closest_index]


def path_cost(order, robot_position):
    total_cost = 0.0
    current_position = robot_position
    for candidate in order:
        candidate_position = candidate['pose'][:2]
        total_cost += math.sqrt((candidate_position[0] - current_position[0]) ** 2 + (candidate_position[1] - current_position[1]) ** 2)
        current_position = candidate_position
    return total_cost


def cost_to_target(order, robot_position, target_index):
    total_cost = 0.0
    current_position = robot_position
    for index, candidate in enumerate(order):
        candidate_position = candidate['pose'][:2]
        total_cost += math.sqrt((candidate_position[0] - current_position[0]) ** 2 + (candidate_position[1] - current_position[1]) ** 2)
        current_position = candidate_position
        if index == target_index:
            return total_cost, index + 1
    return total_cost, len(order)


def find_target_index(order, target_pose, coverage_radius):
    """Find the target in an order by proximity, with a deterministic nearest-neighbor fallback."""
    if not order:
        return 0

    for index, candidate in enumerate(order):
        candidate_position = np.asarray(candidate['pose'][:2], dtype=float)
        if np.linalg.norm(candidate_position - np.asarray(target_pose[:2], dtype=float)) < coverage_radius * 0.45:
            return index

    distances = [np.linalg.norm(np.asarray(candidate['pose'][:2], dtype=float) - np.asarray(target_pose[:2], dtype=float)) for candidate in order]
    return int(np.argmin(distances))


def evaluate_building(handler, transform_matrix, waypoints_distance, offset_distance, building_way, robot_position, weights, coverage_radius, corner_saliency_threshold, candidate_offset_fraction):
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
    inflated_building = inflate_building(local_coordinates, offset_distance)
    sampled_waypoints = sample_waypoints_evenly(inflated_building, waypoints_distance)
    if not sampled_waypoints:
        return None

    candidates = build_candidate_pool(
        inflated_building=inflated_building,
        sample_distance=waypoints_distance,
        exploration_radius=coverage_radius,
        corner_saliency_threshold=corner_saliency_threshold,
        candidate_offset_fraction=candidate_offset_fraction,
    )
    if not candidates:
        return None

    perimeter_length = max(candidate['arc_length'] for candidate in candidates)
    baseline_order = order_baseline(candidates, robot_position)
    optimized_order = select_dynamic_order(candidates, robot_position, 0.0, perimeter_length, weights, coverage_radius)

    target_index = int(np.argmax([candidate['saliency'] for candidate in candidates]))
    target_candidate = candidates[target_index]

    baseline_target_index = find_target_index(baseline_order, target_candidate['pose'], coverage_radius)
    optimized_target_index = find_target_index(optimized_order, target_candidate['pose'], coverage_radius)

    baseline_target_cost, baseline_target_steps = cost_to_target(baseline_order, robot_position, baseline_target_index)
    optimized_target_cost, optimized_target_steps = cost_to_target(optimized_order, robot_position, optimized_target_index)

    baseline_full_cost = path_cost(baseline_order, robot_position)
    optimized_full_cost = path_cost(optimized_order, robot_position)

    return {
        'building_name': building_way['tags'].get('name', f"building_{building_way['id']}"),
        'candidate_count': len(candidates),
        'baseline_target_cost': baseline_target_cost,
        'optimized_target_cost': optimized_target_cost,
        'target_cost_improvement': baseline_target_cost - optimized_target_cost,
        'baseline_target_steps': baseline_target_steps,
        'optimized_target_steps': optimized_target_steps,
        'baseline_full_cost': baseline_full_cost,
        'optimized_full_cost': optimized_full_cost,
        'full_cost_improvement': baseline_full_cost - optimized_full_cost,
        'target_saliency': float(target_candidate['saliency']),
    }


def summarize(records):
    if not records:
        return {}

    baseline_values = [record['baseline_target_cost'] for record in records]
    optimized_values = [record['optimized_target_cost'] for record in records]
    improvements = [record['target_cost_improvement'] for record in records]
    full_improvements = [record['full_cost_improvement'] for record in records]

    return {
        'count': len(records),
        'baseline_target_cost_mean': float(np.mean(baseline_values)),
        'optimized_target_cost_mean': float(np.mean(optimized_values)),
        'mean_improvement': float(np.mean(improvements)),
        'median_improvement': float(np.median(improvements)),
        'mean_full_cost_improvement': float(np.mean(full_improvements)),
        'improved_cases': int(sum(1 for value in improvements if value > 0)),
        'degraded_cases': int(sum(1 for value in improvements if value < 0)),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare the original entrance exploration planner with the optimized planner.')
    parser.add_argument('--osm-file', type=str, default='src/utils_pkg/resource/osm/medium.osm', help='OSM file used for the comparison')
    parser.add_argument('--output-csv', type=str, default='src/delivery_benchmark/result/entrance_planner_comparison.csv', help='Path to save the per-building comparison CSV')
    parser.add_argument('--output-json', type=str, default='src/delivery_benchmark/result/entrance_planner_comparison_summary.json', help='Path to save the summary JSON')
    parser.add_argument('--plot-out', type=str, default='src/delivery_benchmark/result/entrance_planner_comparison.png', help='Path to save the comparison plot')
    parser.add_argument('--exploration-radius', type=float, default=2.0, help='Exploration radius used by the optimized planner')
    parser.add_argument('--sample-distance', type=float, default=5.0, help='Distance between consecutive sampled waypoints')
    parser.add_argument('--coverage-radius', type=float, default=1.2, help='Candidate pruning radius for the optimized planner')
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

    records = []
    weights = {
        'distance': 0.45,
        'saliency': 0.32,
        'coverage': 0.10,
        'novelty': 0.05,
        'heading': 0.04,
        'corner': 0.04,
    }
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

        record = evaluate_building(
            handler=handler,
            transform_matrix=transform_matrix,
            waypoints_distance=args.sample_distance,
            offset_distance=args.exploration_radius,
            building_way=building_way,
            robot_position=robot_position,
            weights=weights,
            coverage_radius=args.coverage_radius,
            corner_saliency_threshold=0.45,
            candidate_offset_fraction=0.35,
        )
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError('No valid building records were produced')

    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)

    summary = summarize(records)
    summary.update({
        'osm_file': str(osm_path),
        'exploration_radius': args.exploration_radius,
        'sample_distance': args.sample_distance,
        'coverage_radius': args.coverage_radius,
    })

    plot_eval2_summary(records, summary, resolve_path(args.plot_out))

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    print(f'Comparison CSV saved to: {output_csv}')
    print(f'Summary JSON saved to: {output_json}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
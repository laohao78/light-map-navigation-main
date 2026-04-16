import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyclipper

from utils_pkg import CoordinateTransformer, OSMHandler
from result_plots import plot_eval1_summary


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


def build_waypoint_sequence(waypoints, robot_position):
    if not waypoints:
        return [], []

    distances = [math.sqrt((x - robot_position[0]) ** 2 + (y - robot_position[1]) ** 2) for x, y, _ in waypoints]
    closest_index = int(np.argmin(distances))

    forward_waypoints = reorder_waypoints(waypoints, closest_index)
    forward_saliency = None
    return forward_waypoints, forward_saliency


def evaluate_building(handler, transform_matrix, waypoints_distance, offset_distance, building_way, robot_position):
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

    center = np.mean(np.asarray(local_coordinates), axis=0)
    orientations = [calculate_yaw(waypoint, center) for waypoint in sampled_waypoints]
    waypoints_with_yaw = [(x, y, yaw) for (x, y), yaw in zip(sampled_waypoints, orientations)]

    vertex_saliency = calculate_vertex_saliency(inflated_building)
    inflated_vertices = [np.asarray(vertex, dtype=float) for vertex in inflated_building]
    waypoint_saliency = []
    for waypoint in sampled_waypoints:
        waypoint_array = np.asarray(waypoint, dtype=float)
        distances = [np.linalg.norm(vertex - waypoint_array) for vertex in inflated_vertices]
        closest_vertex_index = int(np.argmin(distances)) if distances else 0
        waypoint_saliency.append(vertex_saliency[min(closest_vertex_index, len(vertex_saliency) - 1)] if vertex_saliency else 0.0)

    baseline_waypoints = reorder_waypoints(waypoints_with_yaw, int(np.argmin([math.sqrt((x - robot_position[0]) ** 2 + (y - robot_position[1]) ** 2) for x, y, _ in waypoints_with_yaw])))
    baseline_saliency = reorder_waypoints(waypoint_saliency, int(np.argmin([math.sqrt((x - robot_position[0]) ** 2 + (y - robot_position[1]) ** 2) for x, y, _ in waypoints_with_yaw])))

    optimized_forward_waypoints = baseline_waypoints
    optimized_forward_saliency = baseline_saliency
    optimized_backward_waypoints = reverse_keep_start(optimized_forward_waypoints)
    optimized_backward_saliency = reverse_keep_start(optimized_forward_saliency)

    baseline_prefix_3 = calculate_prefix_cost(baseline_waypoints, robot_position, baseline_saliency, horizon=3)
    optimized_forward_prefix_3 = calculate_prefix_cost(optimized_forward_waypoints, robot_position, optimized_forward_saliency, horizon=3)
    optimized_backward_prefix_3 = calculate_prefix_cost(optimized_backward_waypoints, robot_position, optimized_backward_saliency, horizon=3)

    if optimized_backward_prefix_3 < optimized_forward_prefix_3:
        optimized_waypoints = optimized_backward_waypoints
        optimized_prefix_3 = optimized_backward_prefix_3
        optimized_direction = 'reverse'
    else:
        optimized_waypoints = optimized_forward_waypoints
        optimized_prefix_3 = optimized_forward_prefix_3
        optimized_direction = 'forward'

    baseline_full_cost = calculate_prefix_cost(baseline_waypoints, robot_position, baseline_saliency, horizon=len(baseline_waypoints))
    optimized_full_cost = calculate_prefix_cost(optimized_waypoints, robot_position, optimized_forward_saliency if optimized_direction == 'forward' else optimized_backward_saliency, horizon=len(optimized_waypoints))

    return {
        'building_name': building_way['tags'].get('name', f"building_{building_way['id']}"),
        'waypoint_count': len(waypoints_with_yaw),
        'baseline_prefix_3_cost': baseline_prefix_3,
        'optimized_prefix_3_cost': optimized_prefix_3,
        'prefix_3_improvement': baseline_prefix_3 - optimized_prefix_3,
        'baseline_full_cost': baseline_full_cost,
        'optimized_full_cost': optimized_full_cost,
        'direction': optimized_direction,
    }


def summarize(records):
    if not records:
        return {}

    baseline_values = [record['baseline_prefix_3_cost'] for record in records]
    optimized_values = [record['optimized_prefix_3_cost'] for record in records]
    improvements = [record['prefix_3_improvement'] for record in records]

    return {
        'count': len(records),
        'baseline_prefix_3_mean': float(np.mean(baseline_values)),
        'optimized_prefix_3_mean': float(np.mean(optimized_values)),
        'mean_improvement': float(np.mean(improvements)),
        'median_improvement': float(np.median(improvements)),
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
    })

    plot_eval1_summary(records, summary, resolve_path(args.plot_out))

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    print(f'Comparison CSV saved to: {output_csv}')
    print(f'Summary JSON saved to: {output_json}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

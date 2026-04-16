#!/usr/bin/env python3
"""Offline comparison of exploration strategies on a known OSM map.

The script does not run ROS navigation. It only reads the map geometry,
generates the same perimeter waypoints used by the exploration node, and
compares baseline vs adaptive ordering with map-only metrics.
"""

from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass, asdict
from math import atan2, cos, sin, sqrt, pi
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils_pkg import CoordinateTransformer, OSMHandler


Point = Tuple[float, float]
Waypoint = Tuple[float, float, float]


@dataclass
class StrategyMetrics:
    building: str
    strategy: str
    waypoint_count: int
    path_length: float
    start_distance: float
    turn_cost: float
    angular_span: float
    mean_radial_distance: float


@dataclass
class BuildingComparison:
    building: str
    baseline_path_length: float
    adaptive_path_length: float
    path_length_delta: float
    baseline_turn_cost: float
    adaptive_turn_cost: float
    turn_cost_delta: float
    baseline_waypoint_count: int
    adaptive_waypoint_count: int


def normalize_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def angular_distance(angle_a: float, angle_b: float) -> float:
    return abs(normalize_angle(angle_a - angle_b))


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    return sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def polygon_center(points: Sequence[Point]) -> Point:
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    count = len(points)
    return x_sum / count, y_sum / count


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def inflate_polygon(points: Sequence[Point], offset_distance: float) -> List[Point]:
    try:
        import pyclipper
    except Exception as exc:  # pragma: no cover - dependency error is runtime only
        raise RuntimeError('pyclipper is required for polygon inflation') from exc

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(list(points), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    inflated = pco.Execute(offset_distance)
    if not inflated:
        return list(points)
    return [(float(x), float(y)) for x, y in inflated[0]]


def sample_waypoints_evenly(coordinates: Sequence[Point], distance: float) -> List[Point]:
    if len(coordinates) < 2:
        return list(coordinates)

    sampled_waypoints: List[Point] = []
    accumulated_distance = 0.0
    target_distance = 0.0

    for index in range(len(coordinates)):
        start = np.array(coordinates[index], dtype=float)
        end = np.array(coordinates[(index + 1) % len(coordinates)], dtype=float)
        segment_length = float(np.linalg.norm(end - start))

        if segment_length <= 1e-9:
            accumulated_distance += segment_length
            continue

        while accumulated_distance + segment_length >= target_distance:
            ratio = (target_distance - accumulated_distance) / segment_length
            ratio = max(0.0, min(1.0, ratio))
            new_point = start + ratio * (end - start)
            sampled_waypoints.append((float(new_point[0]), float(new_point[1])))
            target_distance += distance

        accumulated_distance += segment_length

    return sampled_waypoints


def waypoint_yaw(waypoint: Point, center: Point) -> float:
    return atan2(waypoint[1] - center[1], waypoint[0] - center[0])


def generate_waypoints(
    building_points: Sequence[Point],
    robot_position: Point,
    exploration_radius: float,
    exploration_points: int,
) -> List[Waypoint]:
    if len(building_points) < 3:
        return []

    if building_points[0] != building_points[-1]:
        building_points = list(building_points) + [building_points[0]]

    inflated = inflate_polygon(building_points, exploration_radius)
    sampled = sample_waypoints_evenly(inflated, exploration_points)
    if not sampled:
        return []

    center = polygon_center(building_points)
    waypoints = [
        (point[0], point[1], waypoint_yaw(point, center))
        for point in sampled
    ]

    closest_index = min(
        range(len(waypoints)),
        key=lambda index: euclidean_distance(robot_position, (waypoints[index][0], waypoints[index][1]))
    )
    return waypoints[closest_index:] + waypoints[:closest_index]


def ordered_waypoints(
    waypoints: Sequence[Waypoint],
    strategy: str,
    robot_position: Point,
    center: Point,
    distance_weight: float = 1.0,
    diversity_weight: float = 2.0,
) -> List[Waypoint]:
    if strategy == 'baseline':
        return list(waypoints)

    remaining = list(range(len(waypoints)))
    visited: List[int] = []
    ordered: List[Waypoint] = []

    while remaining:
        if not visited:
            next_index = min(
                remaining,
                key=lambda index: euclidean_distance(robot_position, (waypoints[index][0], waypoints[index][1]))
            )
        else:
            def score(index: int) -> float:
                waypoint = waypoints[index]
                travel_cost = euclidean_distance(robot_position, (waypoint[0], waypoint[1]))
                candidate_heading = waypoint_yaw((waypoint[0], waypoint[1]), center)
                visited_headings = [waypoint_yaw((waypoints[i][0], waypoints[i][1]), center) for i in visited]
                coverage_gain = min(
                    angular_distance(candidate_heading, heading)
                    for heading in visited_headings
                )
                return distance_weight * travel_cost - diversity_weight * coverage_gain

            next_index = min(remaining, key=score)

        visited.append(next_index)
        remaining.remove(next_index)
        ordered.append(waypoints[next_index])
        robot_position = (waypoints[next_index][0], waypoints[next_index][1])

    return ordered


def measure_path(
    ordered: Sequence[Waypoint],
    start_position: Point,
    center: Point,
) -> StrategyMetrics:
    if not ordered:
        return StrategyMetrics('', '', 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    path_length = euclidean_distance(start_position, (ordered[0][0], ordered[0][1]))
    turn_cost = 0.0
    radial_distances: List[float] = []
    heading_values: List[float] = []

    for index, waypoint in enumerate(ordered):
        radial_distances.append(euclidean_distance((waypoint[0], waypoint[1]), center))
        heading_values.append(waypoint[2])
        if index > 0:
            prev = ordered[index - 1]
            path_length += euclidean_distance((prev[0], prev[1]), (waypoint[0], waypoint[1]))
            turn_cost += angular_distance(waypoint[2], prev[2])

    angular_span = 0.0
    sorted_headings = sorted(normalize_angle(value) for value in heading_values)
    for index in range(len(sorted_headings)):
        current_value = sorted_headings[index]
        next_value = sorted_headings[(index + 1) % len(sorted_headings)]
        diff = next_value - current_value if index + 1 < len(sorted_headings) else (sorted_headings[0] + 2.0 * pi) - current_value
        angular_span += abs(diff)

    return StrategyMetrics(
        building='',
        strategy='',
        waypoint_count=len(ordered),
        path_length=path_length,
        start_distance=euclidean_distance(start_position, (ordered[0][0], ordered[0][1])),
        turn_cost=turn_cost,
        angular_span=angular_span,
        mean_radial_distance=float(sum(radial_distances) / len(radial_distances)),
    )


def load_buildings(osm_file: Path, building_name: Optional[str], min_area: float) -> Dict[str, List[Point]]:
    handler = OSMHandler()
    handler.apply_file(str(osm_file))

    transformer = CoordinateTransformer()
    buildings: Dict[str, List[Point]] = {}

    for way in handler.ways_info:
        tags = way.get('tags', {})
        if tags.get('building') != 'yes':
            continue

        name = tags.get('name', f"building_{way['id']}")
        if building_name and name != building_name:
            continue

        coordinates = handler.get_way_nodes_locations(way['id'])
        if len(coordinates) < 3:
            continue

        projected = [transformer.wgs84_to_utm(lon, lat)[:2] for lon, lat in coordinates]
        if polygon_area(projected) < min_area:
            continue

        buildings[name] = projected

    return buildings


def evaluate_building(
    building_name: str,
    building_points: Sequence[Point],
    robot_position: Point,
    exploration_radius: float,
    exploration_points: int,
    distance_weight: float,
    diversity_weight: float,
) -> List[StrategyMetrics]:
    center = polygon_center(building_points)
    base_waypoints = generate_waypoints(building_points, robot_position, exploration_radius, exploration_points)

    results: List[StrategyMetrics] = []
    for strategy in ('baseline', 'adaptive'):
        ordered = ordered_waypoints(
            base_waypoints,
            strategy,
            robot_position,
            center,
            distance_weight=distance_weight,
            diversity_weight=diversity_weight,
        )
        metrics = measure_path(ordered, robot_position, center)
        metrics.building = building_name
        metrics.strategy = strategy
        results.append(metrics)

    return results


def summarize_building_results(results: Sequence[StrategyMetrics]) -> List[BuildingComparison]:
    grouped: Dict[str, Dict[str, StrategyMetrics]] = {}
    for result in results:
        grouped.setdefault(result.building, {})[result.strategy] = result

    summary: List[BuildingComparison] = []
    for building_name, strategies in grouped.items():
        baseline = strategies.get('baseline')
        adaptive = strategies.get('adaptive')
        if baseline is None or adaptive is None:
            continue

        summary.append(
            BuildingComparison(
                building=building_name,
                baseline_path_length=baseline.path_length,
                adaptive_path_length=adaptive.path_length,
                path_length_delta=adaptive.path_length - baseline.path_length,
                baseline_turn_cost=baseline.turn_cost,
                adaptive_turn_cost=adaptive.turn_cost,
                turn_cost_delta=adaptive.turn_cost - baseline.turn_cost,
                baseline_waypoint_count=baseline.waypoint_count,
                adaptive_waypoint_count=adaptive.waypoint_count,
            )
        )

    return sorted(summary, key=lambda item: item.building)


def write_csv(output_path: Path, results: Sequence[StrategyMetrics], summary: Sequence[BuildingComparison]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['type', 'building', 'strategy', 'waypoint_count', 'path_length', 'start_distance', 'turn_cost', 'angular_span', 'mean_radial_distance'])
        for result in results:
            writer.writerow([
                'raw',
                result.building,
                result.strategy,
                result.waypoint_count,
                f'{result.path_length:.6f}',
                f'{result.start_distance:.6f}',
                f'{result.turn_cost:.6f}',
                f'{result.angular_span:.6f}',
                f'{result.mean_radial_distance:.6f}',
            ])

        writer.writerow([])
        writer.writerow(['type', 'building', 'baseline_path_length', 'adaptive_path_length', 'path_length_delta', 'baseline_turn_cost', 'adaptive_turn_cost', 'turn_cost_delta', 'baseline_waypoint_count', 'adaptive_waypoint_count'])
        for item in summary:
            writer.writerow([
                'summary',
                item.building,
                f'{item.baseline_path_length:.6f}',
                f'{item.adaptive_path_length:.6f}',
                f'{item.path_length_delta:.6f}',
                f'{item.baseline_turn_cost:.6f}',
                f'{item.adaptive_turn_cost:.6f}',
                f'{item.turn_cost_delta:.6f}',
                item.baseline_waypoint_count,
                item.adaptive_waypoint_count,
            ])


def plot_summary(output_path: Path, summary: Sequence[BuildingComparison]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        raise RuntimeError('matplotlib is required for plotting') from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    buildings = [item.building for item in summary]
    deltas = [item.path_length_delta for item in summary]
    turn_deltas = [item.turn_cost_delta for item in summary]

    fig, (ax_path, ax_turn) = plt.subplots(2, 1, figsize=(max(10, len(buildings) * 0.7), 8), sharex=True)

    ax_path.bar(buildings, deltas, color=['#2E86AB' if value <= 0 else '#C0392B' for value in deltas])
    ax_path.axhline(0.0, color='black', linewidth=1)
    ax_path.set_ylabel('Path length delta\n(adaptive - baseline)')
    ax_path.set_title('Offline OSM Exploration Comparison')

    ax_turn.bar(buildings, turn_deltas, color=['#16A085' if value <= 0 else '#D35400' for value in turn_deltas])
    ax_turn.axhline(0.0, color='black', linewidth=1)
    ax_turn.set_ylabel('Turn cost delta\n(adaptive - baseline)')
    ax_turn.set_xlabel('Building')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline OSM-based exploration evaluation')
    parser.add_argument('--osm-file', required=True, help='Path to the OSM map file')
    parser.add_argument('--building', default=None, help='Optional building name filter')
    parser.add_argument('--robot-x', type=float, default=None, help='Robot start x in projected coordinates')
    parser.add_argument('--robot-y', type=float, default=None, help='Robot start y in projected coordinates')
    parser.add_argument('--exploration-radius', type=float, default=2.0)
    parser.add_argument('--exploration-points', type=int, default=5)
    parser.add_argument('--adaptive-distance-weight', type=float, default=1.0)
    parser.add_argument('--adaptive-diversity-weight', type=float, default=2.0)
    parser.add_argument('--min-area', type=float, default=10.0, help='Skip very small polygons')
    parser.add_argument('--csv-out', default=None, help='Optional CSV output path')
    parser.add_argument('--plot-out', default=None, help='Optional plot output path (png/pdf)')
    parser.add_argument('--json', action='store_true', help='Print JSON only')
    args = parser.parse_args()

    osm_file = Path(args.osm_file).expanduser().resolve()
    if not osm_file.exists():
        raise FileNotFoundError(f'OSM file not found: {osm_file}')

    buildings = load_buildings(osm_file, args.building, args.min_area)
    if not buildings:
        raise RuntimeError('No matching buildings found in the OSM file')

    all_results = []
    for building_name, building_points in buildings.items():
        center = polygon_center(building_points)
        robot_position = (
            args.robot_x if args.robot_x is not None else center[0] + 10.0,
            args.robot_y if args.robot_y is not None else center[1] - 10.0,
        )
        all_results.extend(
            evaluate_building(
                building_name,
                building_points,
                robot_position,
                args.exploration_radius,
                args.exploration_points,
                args.adaptive_distance_weight,
                args.adaptive_diversity_weight,
            )
        )

    summary = [asdict(result) for result in all_results]
    comparison_summary = summarize_building_results(all_results)

    if args.csv_out:
        write_csv(Path(args.csv_out).expanduser().resolve(), all_results, comparison_summary)

    if args.plot_out:
        plot_summary(Path(args.plot_out).expanduser().resolve(), comparison_summary)

    if args.json:
        print(json.dumps({
            'raw_results': summary,
            'comparison_summary': [asdict(item) for item in comparison_summary],
        }, ensure_ascii=False, indent=2))
        return

    if comparison_summary:
        baseline_path_mean = sum(item.baseline_path_length for item in comparison_summary) / len(comparison_summary)
        adaptive_path_mean = sum(item.adaptive_path_length for item in comparison_summary) / len(comparison_summary)
        baseline_turn_mean = sum(item.baseline_turn_cost for item in comparison_summary) / len(comparison_summary)
        adaptive_turn_mean = sum(item.adaptive_turn_cost for item in comparison_summary) / len(comparison_summary)

        print('Summary:')
        print(f'  Mean path length: baseline={baseline_path_mean:.2f}, adaptive={adaptive_path_mean:.2f}, delta={adaptive_path_mean - baseline_path_mean:.2f}')
        print(f'  Mean turn cost:   baseline={baseline_turn_mean:.2f}, adaptive={adaptive_turn_mean:.2f}, delta={adaptive_turn_mean - baseline_turn_mean:.2f}')
        print('')

    for result in all_results:
        print(
            f"{result.building:>12} | {result.strategy:>8} | waypoints={result.waypoint_count:2d} "
            f"| path={result.path_length:8.2f} | start={result.start_distance:7.2f} "
            f"| turn={result.turn_cost:7.2f} | span={result.angular_span:7.2f} "
            f"| radial={result.mean_radial_distance:7.2f}"
        )


if __name__ == '__main__':
    main()
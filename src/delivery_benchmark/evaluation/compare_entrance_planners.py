import math

import numpy as np
import pyclipper


def inflate_building(coordinates, offset_distance):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(coordinates, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    inflated_polygon = pco.Execute(offset_distance)
    return inflated_polygon[0] if inflated_polygon else coordinates


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
    if not order:
        return 0

    for index, candidate in enumerate(order):
        candidate_position = np.asarray(candidate['pose'][:2], dtype=float)
        if np.linalg.norm(candidate_position - np.asarray(target_pose[:2], dtype=float)) < coverage_radius * 0.45:
            return index

    distances = [np.linalg.norm(np.asarray(candidate['pose'][:2], dtype=float) - np.asarray(target_pose[:2], dtype=float)) for candidate in order]
    return int(np.argmin(distances))
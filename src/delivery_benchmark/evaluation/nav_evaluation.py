import pandas as pd
from math import sqrt


def load_navigation_csv(file_path):
    """读取导航CSV，并兼容有表头/无表头两种格式。"""
    df = pd.read_csv(file_path)

    if 'status' in df.columns and 'robot_x' in df.columns and 'robot_y' in df.columns:
        return df

    df = pd.read_csv(file_path, header=None)
    base_columns = ['timestamp', 'status', 'task', 'robot_x', 'robot_y', 'robot_yaw']

    if df.shape[1] <= len(base_columns):
        df.columns = base_columns[:df.shape[1]]
    else:
        extra_columns = [f'extra_{i}' for i in range(df.shape[1] - len(base_columns))]
        df.columns = base_columns + extra_columns

    return df


def get_task_series(df):
    """返回统一的任务列，兼容 task/address/缺失列。"""
    if 'task' in df.columns:
        return df['task'].astype(str)
    if 'address' in df.columns:
        return df['address'].astype(str)
    return pd.Series([''] * len(df), index=df.index, dtype=str)

def extract_end_records(file_path):
    """从CSV文件中提取结束记录，并计算每段轨迹的长度"""
    df = load_navigation_csv(file_path)

    if 'status' not in df.columns:
        raise KeyError('CSV文件中未找到 status 列')
    if 'robot_x' not in df.columns or 'robot_y' not in df.columns:
        raise KeyError('CSV文件中未找到 robot_x 或 robot_y 列')

    df = df.reset_index(drop=True)
    df['robot_x'] = pd.to_numeric(df['robot_x'], errors='coerce')
    df['robot_y'] = pd.to_numeric(df['robot_y'], errors='coerce')
    df = df.dropna(subset=['status', 'robot_x', 'robot_y']).reset_index(drop=True)

    end_records = []
    track_lengths = []

    task_series = get_task_series(df)
    navigation_mask = task_series.str.contains('NAVIGATION', na=False)
    if navigation_mask.any():
        mission_start_indices = df.index[(df['status'] == 'start') & navigation_mask].tolist()
    else:
        mission_start_indices = df.index[df['status'] == 'start'].tolist()

    for mission_position, start_index in enumerate(mission_start_indices):
        end_index = mission_start_indices[mission_position + 1] - 1 if mission_position + 1 < len(mission_start_indices) else len(df) - 1

        if end_index < start_index:
            continue

        mission_df = df.iloc[start_index:end_index + 1].reset_index(drop=True)
        if len(mission_df) == 0:
            continue

        current_track_length = 0.0
        for i in range(1, len(mission_df)):
            x1, y1 = mission_df.at[i - 1, 'robot_x'], mission_df.at[i - 1, 'robot_y']
            x2, y2 = mission_df.at[i, 'robot_x'], mission_df.at[i, 'robot_y']
            current_track_length += sqrt((x2 - x1)**2 + (y2 - y1)**2)

        end_records.append(mission_df.iloc[-1].to_dict())
        track_lengths.append(current_track_length)

    return end_records, track_lengths

def read_coordinates(file_path):
    """从TXT文件中读取真值坐标"""
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if ') (' in line:
                for coord in line.split(') ('):
                    coord = coord.strip('()')
                    x, y = map(float, coord.split(','))
                    coordinates.append((x, y))
            elif line.startswith('(') and line.endswith(')'):
                line = line[1:-1]
                x, y = map(float, line.split(','))
                coordinates.append((x, y))
    return coordinates

def calculate_metrics(predicted, ground_truth):
    """计算简单的误差指标，例如欧几里得距离"""
    if len(predicted) != len(ground_truth):
        raise ValueError("预测值和真值列表长度不一致！")

    errors = []
    for (pred, gt) in zip(predicted, ground_truth):
        error = sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
        errors.append(error)
    
    return errors

def calculate_success_rate(errors, threshold=10):
    """计算成功率，error <= threshold 为成功"""
    successes = sum(1 for error in errors if error <= threshold)
    total = len(errors)
    success_rate = successes / total if total > 0 else 0
    return success_rate, successes, total

def calculate_origin_distances(ground_truth):
    """计算从(0, 0)到每个真值坐标的距离"""
    origin_distances = []
    for gt in ground_truth:
        distance = sqrt(gt[0]**2 + gt[1]**2)
        origin_distances.append(distance)
    
    return origin_distances

def calculate_spl(track_lengths, origin_distances, errors, threshold=10):
    """计算SPL指标"""
    N = len(track_lengths)
    if N == 0:
        return 0  # No records to process

    spl_sum = 0.0

    for i in range(N):
        S_i = 1 if errors[i] <= threshold else 0
        p_i = track_lengths[i]
        l_i = origin_distances[i]
        ratio = l_i / max(p_i, l_i)
        print(ratio)
        spl_sum += S_i * ratio

    spl = spl_sum / N
    return spl

# 使用方法：将file_path替换为你的文件路径
csv_file_path = '/home/rm123/Desktop/ROS2_LightMap_Delivery/src/delivery_benchmark/result/test1.csv'
txt_file_path = '/home/rm123/Desktop/ROS2_LightMap_Delivery/src/delivery_benchmark/data/test1_gt.txt'

# 从CSV文件提取结束记录和轨迹长度
end_records, track_lengths = extract_end_records(csv_file_path)
print(len(end_records))

# 从结束记录中提取预测的坐标 (robot_x, robot_y)
predicted_coordinates = [(record['robot_x'], record['robot_y']) for record in end_records]

# 从TXT文件读取真值坐标
ground_truth_coordinates = read_coordinates(txt_file_path)

# 计算误差指标
errors = calculate_metrics(predicted_coordinates, ground_truth_coordinates)

# 计算从(0, 0)到每个真值坐标的距离
origin_distances = calculate_origin_distances(ground_truth_coordinates)

# 计算成功率
success_rate, successes, total = calculate_success_rate(errors, threshold=10)

# 计算spl指标
spl = calculate_spl(track_lengths, origin_distances, errors, threshold=10)

# 打印误差、轨迹长度、成功率、到(0, 0)的距离和spl
for i, (error, track_length, origin_distance) in enumerate(zip(errors, track_lengths, origin_distances)):
    print(f"Record {i + 1}: Error = {error:.4f}, Track Length = {track_length:.4f}, Distance to Origin = {origin_distance:.4f}")

print(f"\nTotal Records: {total}")
print(f"Successful Records: {successes}")
print(f"Success Rate: {success_rate:.2%}")
print(f"SPL: {spl:.4f}")
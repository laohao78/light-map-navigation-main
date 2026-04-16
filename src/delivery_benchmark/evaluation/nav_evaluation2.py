import pandas as pd
from math import sqrt


def load_navigation_csv(file_path):
    """读取导航评估CSV，并兼容有表头/无表头两种格式。"""
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
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        coordinates = []
        if line:
            coords = line.split(') (')
            for coord in coords:
                coord = coord.strip('()')
                x, y = map(float, coord.split(','))
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

def calculate_weighted_success_rate(errors, threshold=10, r=0.9):
    """计算带有权重的成功率，并进行归一化"""
    n = len(errors)
    weighted_success_rate = 0.0
    total_weight = 0.0

    for i in range(1, n + 1):
        S_i = 1 if errors[i-1] <= threshold else 0
        c_i = (r ** (i - 1) * (1 - r)) / (1 - r ** n)
        weighted_success_rate += c_i * S_i
        total_weight += c_i

    # 归一化处理
    if total_weight > 0:
        LSR = weighted_success_rate / total_weight
    else:
        LSR = 0  # 如果总权重为0，返回0

    return LSR

def calculate_segment_distances(ground_truth):
    """计算每个真值坐标段的距离"""
    segment_distances = []
    
    if not ground_truth:
        return segment_distances

    # 计算从原点到第一个真值坐标的距离
    x0, y0 = 0, 0
    x1, y1 = ground_truth[0]
    segment_distances.append(sqrt((x1 - x0)**2 + (y1 - y0)**2))

    # 计算每两个相邻真值坐标之间的距离
    for i in range(1, len(ground_truth)):
        x1, y1 = ground_truth[i-1]
        x2, y2 = ground_truth[i]
        segment_distances.append(sqrt((x2 - x1)**2 + (y2 - y1)**2))
    
    return segment_distances

def calculate_lspl(track_lengths, segment_distances, errors, threshold=10, r=0.9):
    """计算LSPL指标，并进行归一化"""
    n = len(track_lengths)
    lspl_sum = 0.0
    total_weight = 0.0

    for i in range(1, n + 1):
        S_i = 1 if errors[i-1] <= threshold else 0
        c_i = (r ** (i - 1) * (1 - r)) / (1 - r ** n)
        l_i = segment_distances[i-1]
        p_i = track_lengths[i-1]
        lspl_sum += c_i * S_i * (l_i / max(p_i, l_i))
        total_weight += c_i

    # 归一化处理
    if total_weight > 0:
        LSPL = lspl_sum / total_weight
    else:
        LSPL = 0  # 如果总权重为0，返回0

    return LSPL

# 使用方法：将file_path替换为你的文件路径
csv_file_path = '/home/rm123/Desktop/ROS2_LightMap_Delivery/src/delivery_benchmark/result/test2_ada.csv'
txt_file_path = '/home/rm123/Desktop/ROS2_LightMap_Delivery/src/delivery_benchmark/data/test2_gt.txt'

# 从CSV文件提取结束记录和轨迹长度
end_records, track_lengths = extract_end_records(csv_file_path)
print(len(end_records))

# 从结束记录中提取预测的坐标 (robot_x, robot_y)
predicted_coordinates = [(record['robot_x'], record['robot_y']) for record in end_records]

# 从TXT文件读取真值坐标
ground_truth_coordinates = read_coordinates(txt_file_path)

# 计算误差指标
errors = calculate_metrics(predicted_coordinates, ground_truth_coordinates)

# 计算每个真值坐标段的距离
segment_distances = calculate_segment_distances(ground_truth_coordinates)

# 计算带有权重的成功率并归一化
weighted_success_rate = calculate_weighted_success_rate(errors, threshold=10, r=0.9)

# 计算LSPL指标并归一化
lspl = calculate_lspl(track_lengths, segment_distances, errors, threshold=10, r=0.9)

# 打印误差、轨迹长度、带权重的成功率、坐标段距离
for i, (error, track_length, segment_distance) in enumerate(zip(errors, track_lengths, segment_distances)):
    print(f"Record {i + 1}: Error = {error:.4f}, Track Length = {track_length:.4f}, Segment Distance = {segment_distance:.4f}")

print(f"\nWeighted Success Rate (Normalized): {weighted_success_rate:.4f}")
print(f"LSPL (Normalized): {lspl:.4f}")
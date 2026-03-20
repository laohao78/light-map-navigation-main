import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import csv
import math
import numpy as np

# 地球半径 (米)
R = 6378137

def latlon_to_xy(lat, lon, lat0, lon0):
    """将经纬度转换为局部平面坐标"""
    # 转换为弧度后再计算，避免用度直接乘以半径导致尺度错误
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos(lat0_rad) * R
    y = (lat_rad - lat0_rad) * R
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    """将局部平面坐标反算回经纬度"""
    # 逆变换：注意 x,y/R 给出弧度差，需转回度
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R * math.cos(lat0_rad)))
    return lat, lon

def generate_ticks(min_val, max_val):
    """自动生成合适的刻度位置"""
    range_val = max_val - min_val
    if range_val == 0:
        return [min_val]
    
    # 目标大约 5-8 个刻度
    raw_step = range_val / 6
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    
    if normalized < 1.5: step = 1 * magnitude
    elif normalized < 3.5: step = 2 * magnitude
    elif normalized < 7.5: step = 5 * magnitude
    else: step = 10 * magnitude
    
    start = math.ceil(min_val / step) * step
    ticks = []
    curr = start
    while curr <= max_val + 1e-9:
        ticks.append(curr)
        curr += step
    return ticks

# =====================
# 1. 读取 OSM 数据
# =====================
# 请确保文件名与你实际的文件名一致
osm_file = 'medium.osm' 
try:
    tree = ET.parse(osm_file)
except FileNotFoundError:
    print(f"错误: 找不到文件 {osm_file}")
    exit(1)

root = tree.getroot()

bounds = root.find('bounds')
if bounds is None:
    print("错误: OSM 文件中未找到 <bounds> 标签")
    exit(1)

minlat = float(bounds.get('minlat'))
minlon = float(bounds.get('minlon'))
maxlat = float(bounds.get('maxlat'))
maxlon = float(bounds.get('maxlon'))

lat0 = minlat
lon0 = minlon

# 计算地图总宽高 (米)
# 注意：经度/纬度差需要先转为弧度
lat0_rad = math.radians(lat0)
width_m = math.radians(maxlon - minlon) * math.cos(lat0_rad) * R
height_m = math.radians(maxlat - minlat) * R

# 读取 node 并转换坐标
nodes = {}
for n in root.findall('node'):
    nid = n.get('id')
    lat = float(n.get('lat'))
    lon = float(n.get('lon'))
    nodes[nid] = latlon_to_xy(lat, lon, lat0, lon0)

# 保存节点经纬度与局部笛卡尔坐标到 CSV，方便后续使用
csv_file = "medium_coords.csv"
with open(csv_file, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['id', 'x_m', 'y_m', 'lat', 'lon'])
    for n in root.findall('node'):
        nid = n.get('id')
        lat = float(n.get('lat'))
        lon = float(n.get('lon'))
        x, y = nodes[nid]
        writer.writerow([nid, f"{x:.3f}", f"{y:.3f}", f"{lat:.7f}", f"{lon:.7f}"])
print(f"节点坐标已保存: {os.path.abspath(csv_file)}")

# =====================
# 2. 绘图设置
# =====================
fig, ax = plt.subplots(figsize=(10, 10))

# --- 绘制建筑 ---
for way in root.findall('way'):
    tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
    if tags.get('building') != 'yes':
        continue

    name = tags.get('name', '')
    pts = []
    for nd in way.findall('nd'):
        ref = nd.get('ref')
        if ref in nodes:
            pts.append(nodes[ref])

    if len(pts) < 3:
        continue

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    ax.fill(xs, ys, color='gray', alpha=0.3, edgecolor='black', linewidth=0.5)

    if name:
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        ax.text(cx, cy, name, fontsize=7, ha='center', va='center', color='darkblue')

# --- 绘制道路 ---
for way in root.findall('way'):
    tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
    if not tags.get('highway'):
        continue

    name = tags.get('name', '')
    pts = []
    for nd in way.findall('nd'):
        ref = nd.get('ref')
        if ref in nodes:
            pts.append(nodes[ref])

    if len(pts) < 2:
        continue

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    ax.plot(xs, ys, color='white', linewidth=3, zorder=10)
    ax.plot(xs, ys, color='orange', linewidth=1.5, zorder=11)

    if name:
        mid = len(xs) // 2
        if mid > 0:
            dx = xs[mid] - xs[mid - 1]
            dy = ys[mid] - ys[mid - 1]
            angle = math.degrees(math.atan2(dy, dx))
            if 90 < angle < 270:
                angle += 180
            ax.text(xs[mid], ys[mid], name, fontsize=8, rotation=angle, 
                    ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.7, pad=1))

# =====================
# 3. 设置双坐标轴 (修复版)
# =====================

# 主坐标轴设置
ax.set_xlabel("Distance X (meters)", fontsize=10)
ax.set_ylabel("Distance Y (meters)", fontsize=10)
ax.set_title("OSM Map with Lat/Lon and Local Coordinates", fontsize=12)

# 留一点边距
margin_x = width_m * 0.05 if width_m > 0 else 10
margin_y = height_m * 0.05 if height_m > 0 else 10

ax.set_xlim(-margin_x, width_m + margin_x)
ax.set_ylim(-margin_y, height_m + margin_y)

# 生成刻度
x_ticks_m = generate_ticks(0, width_m)
y_ticks_m = generate_ticks(0, height_m)

# 如果没有生成任何刻度（地图太小），至少放一个中心点
if not x_ticks_m: x_ticks_m = [width_m/2]
if not y_ticks_m: y_ticks_m = [height_m/2]

ax.set_xticks(x_ticks_m)
ax.set_yticks(y_ticks_m)

# --- 使用 secondary_xaxis/secondary_yaxis 添加经纬度刻度 (替代 twinx/twiny) ---
# 这样不会创建共享轴，能同时设置等比例显示

# 辅助转换函数：米 <-> 经度/纬度（基于 lat0/lon0）
def x_to_lon(x_m):
    # 支持标量或数组输入
    x_arr = np.asarray(x_m)
    return lon0 + np.degrees(x_arr / (R * math.cos(lat0_rad)))

def lon_to_x(lon):
    lon_arr = np.asarray(lon)
    return np.radians(lon_arr - lon0) * math.cos(lat0_rad) * R

def y_to_lat(y_m):
    y_arr = np.asarray(y_m)
    return lat0 + np.degrees(y_arr / R)

def lat_to_y(lat):
    lat_arr = np.asarray(lat)
    return np.radians(lat_arr - lat0) * R

# 创建 secondary axes（顶部显示经度，右侧显示纬度）
ax_sec_x = ax.secondary_xaxis('top', functions=(x_to_lon, lon_to_x))
ax_sec_y = ax.secondary_yaxis('right', functions=(y_to_lat, lat_to_y))

# 使用与主轴相同的位置刻度，但在副轴显示经纬度刻度线与标签
ax.set_xticks(x_ticks_m)
ax.set_yticks(y_ticks_m)

# 计算经纬度刻度值（secondary axis 使用经纬度值作为刻度位置）
lon_ticks = [x_to_lon(x) for x in x_ticks_m]
lat_ticks = [y_to_lat(y) for y in y_ticks_m]

ax_sec_x.set_xticks(lon_ticks)
ax_sec_x.set_xticklabels([f"{lon:.6f}°" for lon in lon_ticks], fontsize=9, color='blue')
ax_sec_x.set_xlabel("Longitude", fontsize=10, color='blue')

ax_sec_y.set_yticks(lat_ticks)
ax_sec_y.set_yticklabels([f"{lat:.6f}°" for lat in lat_ticks], fontsize=9, color='blue')
ax_sec_y.set_ylabel("Latitude", fontsize=10, color='blue')

# 设置等比例显示，使用 'box' 在无共享轴的情况下可用
ax.set_aspect('equal', adjustable='box')

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

# 在图上标注边界经纬度和中心经纬度，便于查看
center_x = width_m / 2.0
center_y = height_m / 2.0
center_lat, center_lon = xy_to_latlon(center_x, center_y, lat0, lon0)

bbox_text = (
    f"Bounds:\nminlat={minlat:.7f}\nminlon={minlon:.7f}\n"
    f"maxlat={maxlat:.7f}\nmaxlon={maxlon:.7f}\n"
    f"Center: {center_lat:.7f}, {center_lon:.7f}"
)
ax.text(0.02, 0.98, bbox_text, transform=ax.transAxes, fontsize=8,
        va='top', ha='left', color='blue', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# --- 在图上绘制两点（示例：来自 osrm_test.sh，格式为 (lon, lat)） ---
# 你可以修改下面的列表以绘制其它点
# [116.9998,40.497702],[116.999815,40.497702],[117.000158,40.497704],[117.000524,40.497709],[117.000524,40.4979]
points_lonlat = [ (116.9998, 40.4977), (117.0004, 40.4979) , 
(116.9998,40.497702),(116.999815,40.497702),(117.000158,40.497704),(117.000524,40.497709),(117.000524,40.4979) ]
point_labels = ['A', 'B', 
'C', 'D', 'E', 'F', 'G']  # 你可以根据需要修改标签
for (lon, lat), lab in zip(points_lonlat, point_labels):
    px, py = latlon_to_xy(lat, lon, lat0, lon0)
    ax.scatter(px, py, s=60, c='red', edgecolor='black', zorder=30)
    ax.text(px + 1.0, py + 1.0, f"{lab}", fontsize=9, fontweight='bold', color='black', zorder=31)

# 绘制以 A 为圆心、半径 82.5 米的圆
radius_m = 82.5
if len(points_lonlat) > 0:
    lonA, latA = points_lonlat[0]
    ax_cx, ax_cy = latlon_to_xy(latA, lonA, lat0, lon0)
    circ = Circle((ax_cx, ax_cy), radius_m, fill=False, edgecolor='red', linestyle='--', linewidth=1.5, zorder=100, clip_on=False)
    ax.add_patch(circ)
    ax.text(ax_cx + radius_m, ax_cy, f"R={radius_m}m", fontsize=8, color='red')

# 再保存一份带点标记的图片
points_output = "medium_with_points.png"
try:
    plt.savefig(points_output, dpi=300, bbox_inches='tight')
    print(f"带点的地图已保存至: {os.path.abspath(points_output)}")
except Exception as e:
    print(f"保存带点图片失败: {e}")

plt.tight_layout()

output_file = "medium_with_coords.png"
try:
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    abs_path = os.path.abspath(output_file)
    print(f"成功！地图已保存至: {abs_path}")
except Exception as e:
    print(f"保存失败: {e}")

plt.close(fig)
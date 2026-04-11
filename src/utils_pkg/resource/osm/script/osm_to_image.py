import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import numpy as np

# =====================
# 地球半径
# =====================
R = 6378137

# =====================
# 坐标转换函数
# =====================
def latlon_to_xy(lat, lon, lat0, lon0):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos(lat0_rad) * R
    y = (lat_rad - lat0_rad) * R
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    lat0_rad = math.radians(lat0)
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R * math.cos(lat0_rad)))
    return lat, lon

def generate_ticks(min_val, max_val):
    range_val = max_val - min_val
    if range_val == 0:
        return [min_val]
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
# 文件设置
# =====================
file_base = 'medium'
osm_file = f'../{file_base}.osm'  # 向上一级目录
output_file = f'{file_base}_units_centered.png'

# =====================
# 读取 OSM
# =====================
tree = ET.parse(osm_file)
root = tree.getroot()

bounds = root.find('bounds')
if bounds is None:
    raise ValueError("OSM 文件缺少 <bounds> 标签")

minlat = float(bounds.get('minlat'))
minlon = float(bounds.get('minlon'))
maxlat = float(bounds.get('maxlat'))
maxlon = float(bounds.get('maxlon'))

lat0, lon0 = minlat, minlon
lat0_rad = math.radians(lat0)
width_m = math.radians(maxlon - minlon) * math.cos(lat0_rad) * R
height_m = math.radians(maxlat - minlat) * R

center_x = width_m / 2
center_y = height_m / 2

# =====================
# 节点坐标
# =====================
nodes = {}
for n in root.findall('node'):
    nid = n.get('id')
    lat = float(n.get('lat'))
    lon = float(n.get('lon'))
    nodes[nid] = latlon_to_xy(lat, lon, lat0, lon0)

# =====================
# 建筑单元数据
# =====================
building_units = {
    "building1": ['unit1', 'unit2'],
    "building2": ['unit1'],
    "building3": ['unit1', 'unit2'],
    "building4": ['unit1', 'unit2'],
    "building5": ['unit1', 'unit2'],
    "building6": ['unit1', 'unit2'],
    "building7": ['unit1', 'unit2'],
    "building8": ['unit1', 'unit2', 'unit3'],
    "building9": ['unit1', 'unit2'],
    "building10": ['unit1', 'unit2', 'unit3'],
    "building11": ['unit1', 'unit2', 'unit3'],
    "building12": ['unit1', 'unit2', 'unit3'],
    "building13": ['unit1', 'unit2', 'unit3'],
    "building14": ['unit1', 'unit2'],
}

building_units_coords = {
    "building1": ['(-28.63, 29.34)', '(-24.94, 28.68)'],
    "building2": ['(-25.54, 9.27)'],
    "building3": ['(-34.89, -8.06)', '(-25.32, -7.53)'],
    "building4": ['(-34.69, -19.47)', '(-32.10, -20.92)'],
    "building5": ['(-24.44, -21.21)', '(-27.03, -19.76)'],
    "building6": ['(-29.96, -34.22)', '(-28.51, -31.63)'],
    "building7": ['(9.50, 25.45)', '(-6.49, 24.51)'],
    "building8": ['(-0.96, 7.53)', '(8.51, 7.97)', '(13.38, 7.57)'],
    "building9": ['(24.48, 8.08)', '(29.52, 8.06)'],
    "building10": ['(0.65, -15.58)', '(1.19, -12.14)', '(3.83, -16.40)'],
    "building11": ['(0.85, -27.57)', '(-2.59, -27.03)', '(1.66, -24.35)'],
    "building12": ['(13.18, -29.07)', '(13.47, -25.23)', '(16.08, -29.87)'],
    "building13": ['(27.28, -29.19)', '(27.64, -25.31)', '(30.52, -30.19)'],
    "building14": ['(28.67, -17.74)', '(28.69, -10.10)'],
}

# =====================
# 绘图
# =====================
fig, ax = plt.subplots(figsize=(10, 10))

# --- 绘制建筑 ---
buildings = []
for way in root.findall('way'):
    tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
    if tags.get('building') != 'yes':
        continue
    pts = [nodes[nd.get('ref')] for nd in way.findall('nd') if nd.get('ref') in nodes]
    if len(pts) < 3:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.fill(xs, ys, color='gray', alpha=0.3, edgecolor='black', linewidth=0.5)
    center_x_poly = sum(xs) / len(xs)
    center_y_poly = sum(ys) / len(ys)
    building_name = tags.get('name', '')
    buildings.append((building_name, center_y_poly, center_x_poly, center_y_poly))

buildings.sort(key=lambda item: item[0])

for building_name, _, center_x_poly, center_y_poly in buildings:
    ax.text(
        center_x_poly,
        center_y_poly,
        building_name,
        fontsize=8,
        color='black',
        ha='center',
        va='center',
        zorder=20,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0),
    )

# --- 绘制道路 ---
for way in root.findall('way'):
    tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
    if not tags.get('highway'):
        continue
    pts = [nodes[nd.get('ref')] for nd in way.findall('nd') if nd.get('ref') in nodes]
    if len(pts) < 2:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color='white', linewidth=3, zorder=10)
    ax.plot(xs, ys, color='orange', linewidth=1.5, zorder=11)

# --- 绘制建筑单元（中心对齐） ---
for bname, units in building_units.items():
    coords_list = building_units_coords.get(bname, [])
    for idx, coord_str in enumerate(coords_list):
        x_str, y_str = coord_str.strip('()').split(',')
        ux, uy = float(x_str), float(y_str)
        # 对齐到地图中心
        x = center_x + ux
        y = center_y + uy
        ax.scatter(x, y, s=30, c='green', edgecolor='black', zorder=50)
        ax.text(x + 0.5, y + 0.5, f"{idx+1}", fontsize=7, color='darkgreen', zorder=51)

# =====================
# 坐标轴设置
# =====================
margin_x = width_m * 0.05 if width_m > 0 else 10
margin_y = height_m * 0.05 if height_m > 0 else 10
ax.set_xlim(-margin_x, width_m + margin_x)
ax.set_ylim(-margin_y, height_m + margin_y)
ax.set_xlabel("Distance X (meters)")
ax.set_ylabel("Distance Y (meters)")
ax.set_title("OSM Map with Building Units (Green Dots Centered)")

x_ticks_m = generate_ticks(0, width_m)
y_ticks_m = generate_ticks(0, height_m)
ax.set_xticks(x_ticks_m)
ax.set_yticks(y_ticks_m)

# 辅助经纬度轴
def x_to_lon(x): return lon0 + np.degrees(x / (R * math.cos(lat0_rad)))
def lon_to_x(lon): return np.radians(lon - lon0) * R * math.cos(lat0_rad)
def y_to_lat(y): return lat0 + np.degrees(y / R)
def lat_to_y(lat): return np.radians(lat - lat0) * R

ax_sec_x = ax.secondary_xaxis('top', functions=(x_to_lon, lon_to_x))
ax_sec_y = ax.secondary_yaxis('right', functions=(y_to_lat, lat_to_y))
ax_sec_x.set_xticks([x_to_lon(x) for x in x_ticks_m])
ax_sec_x.set_xticklabels([f"{lon:.6f}°" for lon in [x_to_lon(x) for x in x_ticks_m]], fontsize=9, color='blue')
ax_sec_x.set_xlabel("Longitude", fontsize=10, color='blue')
ax_sec_y.set_yticks([y_to_lat(y) for y in y_ticks_m])
ax_sec_y.set_yticklabels([f"{lat:.6f}°" for lat in [y_to_lat(y) for y in y_ticks_m]], fontsize=9, color='blue')
ax_sec_y.set_ylabel("Latitude", fontsize=10, color='blue')

ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

# =====================
# 保存图片
# =====================
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 地图已保存: {output_file}")
plt.show()
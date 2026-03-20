import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
import numpy as np

R = 6378137

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * math.cos(math.radians(lat0)) * R
    y = (lat - lat0) * R
    return x, y

# 读取 OSM
tree = ET.parse('large.osm')
root = tree.getroot()

bounds = root.find('bounds')
lat0 = float(bounds.get('minlat'))
lon0 = float(bounds.get('minlon'))

# 读取 node
nodes = {}
for n in root.findall('node'):
    nid = n.get('id')
    lat = float(n.get('lat'))
    lon = float(n.get('lon'))
    nodes[nid] = latlon_to_xy(lat, lon, lat0, lon0)

plt.figure(figsize=(8, 8))

# =====================
# 🏢 画建筑（填充 + 居中标注）
# =====================
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

    # 填充建筑
    plt.fill(xs, ys, alpha=0.4)

    # 计算中心
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    if name:
        plt.text(cx, cy, name, fontsize=8, ha='center', va='center')

# =====================
# 🛣️ 画道路（加粗 + 旋转标注）
# =====================
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

    # 画道路（更粗）
    plt.plot(xs, ys, linewidth=2)

    # === 计算中点 + 方向角 ===
    mid = len(xs) // 2

    if mid > 0:
        dx = xs[mid] - xs[mid - 1]
        dy = ys[mid] - ys[mid - 1]
        angle = math.degrees(math.atan2(dy, dx))
    else:
        angle = 0

    if name:
        plt.text(xs[mid], ys[mid], name,
                 fontsize=8,
                 rotation=angle,
                 ha='center',
                 va='center')

# =====================
# 🎯 美化
# =====================
plt.axis('equal')
plt.title("OSM Map (Styled)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.tight_layout()
plt.savefig("large.png", dpi=300)
plt.show()
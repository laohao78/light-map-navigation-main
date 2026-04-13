cd ./src/utils_pkg/resource/osm

# 1. 设置文件名为 medium.osm
FILE="medium.osm"
BASE_NAME="${FILE%.osm}"

echo "正在处理 $FILE ..."

# 2. 提取 (Extract)
# 这次应该能解析出成千上万个节点和道路
docker run -t -v $(pwd):/data osrm/osrm-backend \
osrm-extract -p /opt/car.lua /data/$FILE

# 如果上面成功，继续执行下面两步：
# 3. 分区 (Partition)
docker run -t -v $(pwd):/data osrm/osrm-backend \
osrm-partition /data/$BASE_NAME.osrm

# 4. 定制 (Customize)
docker run -t -v $(pwd):/data osrm/osrm-backend \
osrm-customize /data/$BASE_NAME.osrm

echo "预处理完成！准备启动服务..."

# 5. 启动服务
docker run -t -i -p 5001:5000 -v $(pwd):/data osrm/osrm-backend \
osrm-routed --algorithm mld /data/medium.osrm
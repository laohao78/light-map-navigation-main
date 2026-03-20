cd ./src/utils_pkg/resource/osm


# 5. 启动服务
docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend \
osrm-routed --algorithm mld /data/medium.osrm
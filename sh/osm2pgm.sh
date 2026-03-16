SRC_PATH=$(ros2 pkg prefix utils_pkg)/../../src/utils_pkg

ros2 run utils_pkg osm2pgm --ros-args \
    -p osm_file:="${SRC_PATH}/resource/osm/small.osm" \
    -p pgm_file:="${SRC_PATH}/resource/pgm/SMALL_OSM.pgm" \
    -p yaml_file:="${SRC_PATH}/resource/pgm/SMALL_OSM.yaml" \
    -p padding_meters:=10.0

ros2 run utils_pkg osm2pgm --ros-args \
    -p osm_file:="${SRC_PATH}/resource/osm/medium.osm" \
    -p pgm_file:="${SRC_PATH}/resource/pgm/MEDIUM_OSM.pgm" \
    -p yaml_file:="${SRC_PATH}/resource/pgm/MEDIUM_OSM.yaml" \
    -p padding_meters:=10.0

ros2 run utils_pkg osm2pgm --ros-args \
    -p osm_file:="${SRC_PATH}/resource/osm/large.osm" \
    -p pgm_file:="${SRC_PATH}/resource/pgm/LARGE_OSM.pgm" \
    -p yaml_file:="${SRC_PATH}/resource/pgm/LARGE_OSM.yaml" \
    -p padding_meters:=10.0
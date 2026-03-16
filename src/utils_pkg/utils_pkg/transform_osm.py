from typing import Tuple
import contextlib
import osmium as osm
import pyproj
import numpy as np


class CoordinateTransformer:
    """Handles coordinate system transformations."""
    
    def __init__(self) -> None:
        self.wgs84_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:32633", always_xy=True
        )
        self.utm_to_wgs84 = pyproj.Transformer.from_crs(
            "EPSG:32650", "EPSG:4326", always_xy=True
        )

    def transform(self, lon: float, lat: float) -> Tuple[float, float]:
        """Transform coordinates with high precision.

        Args:
            lon: Longitude (经度)
            lat: Latitude (纬度)

        Returns:
            Tuple[float, float]: 转换后的 (longitude, latitude)
        """
        self.wgs84_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:32633", always_xy=True
        )
        
        # 第一次转换：WGS84 -> UTM
        utm_x, utm_y = self.wgs84_to_utm.transform(
            lon,
            lat,
        )
        print(utm_x, utm_y)
        
        # 如果需要偏移，使用 numpy 以保持精度
        utm_x_shifted = np.float64(utm_x) + np.float64(500000)
        utm_y_shifted = np.float64(utm_y) + np.float64(4483000)
        
        # 最终转换：UTM -> WGS84
        final_lon, final_lat = self.utm_to_wgs84.transform(
            utm_x_shifted,
            utm_y_shifted,
        )
        
        # 验证转换精度
        self.wgs84_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:32650", always_xy=True
        )
        check_x, check_y = self.wgs84_to_utm.transform(final_lon, final_lat)
        
        # 使用更严格的精度检查
        if not (np.isclose(check_x, utm_x_shifted, rtol=1e-3, atol=1e-6) and 
                np.isclose(check_y, utm_y_shifted, rtol=1e-3, atol=1e-6)):
            raise ValueError(
                f"Precision loss detected:\n"
                f"Original UTM: {utm_x_shifted}, {utm_y_shifted}\n"
                f"Check UTM: {check_x}, {check_y}\n"
                f"Difference: {utm_x_shifted - check_x}, {utm_y_shifted - check_y}"
            )
        
        return final_lon, final_lat


class OSMTranslator(osm.SimpleHandler):
    def __init__(self, transformer, writer):
        super().__init__()
        self.transformer = transformer
        self.writer = writer

    def node(self, n):
        # Transform the node coordinates
        new_lon, new_lat = self.transformer.transform(n.location.lon, n.location.lat)

        # Write the transformed node
        self.writer.add_node(
            osm.osm.mutable.Node(
                id=n.id,
                location=osm.osm.Location(new_lon, new_lat),
                tags=n.tags,
                version=n.version  # Keep the version attribute from the input
            )
        )

    def way(self, w):
        # Write the way without modifying it (nodes are written separately)
        self.writer.add_way(
            osm.osm.mutable.Way(
                id=w.id,
                nodes=w.nodes,
                tags=w.tags,
                version=w.version  # Keep the version attribute from the input
            )
        )

    def relation(self, r):
        # Write the relation without modifying it (nodes and ways are written separately)
        self.writer.add_relation(
            osm.osm.mutable.Relation(
                id=r.id,
                members=r.members,
                tags=r.tags,
                version=r.version  # Keep the version attribute from the input
            )
        )


def translate_osm(input_file: str, output_file: str) -> None:
    """Transform coordinates in OSM file.

    Args:
        input_file: Path to input OSM file
        output_file: Path to output OSM file
    """
    transformer = CoordinateTransformer()
    
    with contextlib.closing(osm.SimpleWriter(output_file)) as writer:
        handler = OSMTranslator(transformer, writer)
        handler.apply_file(input_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform OSM coordinates to Beijing UTM Zone 50N"
    )
    parser.add_argument("input_file", help="Input OSM file")
    parser.add_argument("output_file", help="Output OSM file")

    args = parser.parse_args()

    try:
        translate_osm(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        raise
import argparse
import os
from typing import List, Tuple
import cv2
import numpy as np
import yaml

def pixel_to_physical(x, y, resolution, origin):
    """
    Converts pixel coordinates to physical coordinates using resolution and origin.
    Adjusted to correct coordinate system alignment.
    """
    px = x * resolution + origin[0]
    py = (y * resolution) + origin[1]
    return np.array([px, py, 1])

def physical_to_pixel(px, py, resolution, origin):
    """
    Converts physical coordinates to pixel coordinates using resolution and origin.
    Adjusted to align with image coordinates.
    """
    x = int((px - origin[0]) / resolution)
    y = int((py - origin[1]) / resolution)
    return (x, y)

def transform_points(image, transform_matrix, resolution, origin, threshold=50):
    """
    Transforms only pixels that are close to black using the transformation matrix.
    Returns the transformed points as physical coordinates.
    """
    points = []
    h, w = image.shape
    
    print(f"Image shape: {h}x{w}")

    for y in range(h):
        for x in range(w):
            if image[y, x] < threshold: 
                phys_coords = pixel_to_physical(x, h - y, resolution, origin)
                transformed_phys_coords = np.dot(transform_matrix, phys_coords)
                points.append(transformed_phys_coords[:2])

    return np.array(points)

def generate_new_pgm(points, resolution, output_image_path, margins):
    """Generate new PGM image with controllable margins
    
    Args:
        points: Transformed point set
        resolution: Resolution in meters/pixel
        output_image_path: Output image path
        margins: Margin parameters [left, right, top, bottom] in meters
    """
    if len(points) == 0:
        raise ValueError("No points to transform. Image may not contain any black pixels.")

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate margins in pixels
    left_margin = int(margins[0] / resolution)
    right_margin = int(margins[1] / resolution)
    top_margin = int(margins[2] / resolution)
    bottom_margin = int(margins[3] / resolution)

    points_width = int((max_x - min_x) / resolution)
    points_height = int((max_y - min_y) / resolution)

    new_width = points_width + left_margin + right_margin
    new_height = points_height + top_margin + bottom_margin

    new_origin = [
        min_x - (left_margin * resolution),
        min_y - (bottom_margin * resolution)
    ]

    new_image = np.full((new_height, new_width), 255, dtype=np.uint8)

    for point in points:
        px, py = physical_to_pixel(point[0], point[1], resolution, new_origin)
        if 0 <= px < new_width and 0 <= py < new_height:
            new_image[new_height - 1 - py, px] = 0

    cv2.imwrite(output_image_path, new_image)
    return new_origin

def update_yaml(yaml_data, new_image_path, new_origin, transform_matrix):
    """Updates YAML data with new parameters
    
    Args:
        yaml_data: Original YAML data
        new_image_path: Path to the transformed image
        new_origin: New origin coordinates
        transform_matrix: Applied transformation matrix
    """
    yaml_data['image'] = new_image_path.split('/')[-1]
    yaml_data['origin'] = [float(new_origin[0]), float(new_origin[1]), yaml_data['origin'][2]]
    yaml_data['transform_matrix'] = {
        'row1': transform_matrix[0].tolist(),
        'row2': transform_matrix[1].tolist(),
        'row3': transform_matrix[2].tolist()
    }
    return yaml_data

class TransformConfig:
    """Configuration class for transformation parameters"""
    def __init__(self, transform_matrix: np.ndarray = None):
        self.transform_matrix = transform_matrix or np.array([
            [1.0, 0.0, -500000],
            [0.0, 1.0, -4483000],
            [0.0, 0.0, 1.0]
        ])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transform PGM maps with associated YAML files')
    parser.add_argument('--input-dir', required=True, help='Input directory containing PGM and YAML files')
    parser.add_argument('--output-dir', required=True, help='Output directory for transformed files')
    parser.add_argument('--prefixes', nargs='+', default=['SMALL', 'MEDIUM', 'LARGE'],
                      help='Prefixes for input/output files')
    parser.add_argument('--margins', nargs=4, type=float, default=[20, 20, 20, 20],
                      help='Margins in meters [left, right, top, bottom]')
    return parser.parse_args()

def get_file_pairs(base_path: str, prefixes: List[str]) -> List[Tuple[str, str]]:
    """Generate input/output file pairs based on prefixes"""
    return [
        [f'{base_path}/{prefix}_OSM.pgm', f'{base_path}/{prefix}_OSM.yaml']
        for prefix in prefixes
    ]

def process_image_batch(input_files, output_files, margins=[20, 20, 20, 20]):
    """Process multiple image files
    
    Args:
        input_files: List of input file information, each containing [pgm_file_path, yaml_file_path]
        output_files: List of output file information, each containing [pgm_file_path, yaml_file_path]
        margins: Margin parameters [left, right, top, bottom] in meters
    """
    for (input_pgm, input_yaml), (output_pgm, output_yaml) in zip(input_files, output_files):
        rigid_transform_from_matrix(
            image_path=input_pgm,
            output_image_path=output_pgm,
            yaml_path=input_yaml,
            output_yaml_path=output_yaml,
            margins=margins
        )

def rigid_transform_from_matrix(image_path, output_image_path, yaml_path, output_yaml_path, margins=[10, 10, 10, 10]):
    """Transform PGM image using transformation matrix"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    resolution = yaml_data['resolution']
    origin = yaml_data['origin']

    transformed_points = transform_points(image, TRANSFORM_MATRIX, resolution, origin)
    new_origin = generate_new_pgm(transformed_points, resolution, output_image_path, margins)

    updated_yaml_data = update_yaml(yaml_data, output_image_path, new_origin, TRANSFORM_MATRIX)
    with open(output_yaml_path, 'w') as file:
        yaml.dump(updated_yaml_data, file, default_flow_style=False, sort_keys=False)

def main():
    """Main entry point"""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate input/output file pairs
    input_files = get_file_pairs(args.input_dir, args.prefixes)
    output_files = get_file_pairs(args.output_dir, args.prefixes)
    
    # Initialize transformation configuration
    config = TransformConfig()
    
    process_image_batch(
        input_files=input_files,
        output_files=output_files,
        margins=args.margins
    )

if __name__ == "__main__":
    main()
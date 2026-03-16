import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import argparse
import sys
from pathlib import Path

def process_osm_file(input_file: str, output_file: str) -> None:
    """
    Process OSM file by adding bounds and changeset attributes.
    """
    try:
        with open(input_file, 'r', encoding='UTF-8') as f:
            content = f.read()
        
        if content.count('<?xml') > 1:
            lines = content.splitlines()
            content = lines[0] + '\n' + '\n'.join(
                line for line in lines[1:] if not line.strip().startswith('<?xml')
            )
        
        root = ET.fromstring(content)
        
        bounds = {
            'min_lat': float('inf'),
            'max_lat': float('-inf'),
            'min_lon': float('inf'),
            'max_lon': float('-inf')
        }
        
        for node in root.findall('node'):
            if 'changeset' not in node.attrib:
                node.set('changeset', '1')
                
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            
            bounds['min_lat'] = min(bounds['min_lat'], lat)
            bounds['max_lat'] = max(bounds['max_lat'], lat)
            bounds['min_lon'] = min(bounds['min_lon'], lon)
            bounds['max_lon'] = max(bounds['max_lon'], lon)
        
        for way in root.findall('way'):
            if 'changeset' not in way.attrib:
                way.set('changeset', '1')
        
        bounds_elem = ET.Element('bounds')
        for key, value in bounds.items():
            bounds_elem.set(key.replace('_', ''), str(value))
        
        first_node = root.find('node')
        if first_node is not None:
            root_list = list(root)
            insert_pos = root_list.index(first_node)
            root.insert(insert_pos, bounds_elem)
        
        xmlstr = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xmlstr)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        lines = [line for line in pretty_xml.splitlines() if line.strip()]
        
        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            f.write('\n'.join(lines[1:]))
            f.write('\n')
            
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Process OSM file to add bounds and changeset attributes')
    parser.add_argument('input_file', help='Input OSM file path')
    parser.add_argument('-o', '--output', help='Output file path (defaults to input file)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else input_path
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    process_osm_file(str(input_path), str(output_path))

if __name__ == '__main__':
    main()

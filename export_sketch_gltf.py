"""
Convert sketch (.obj) to GLTF.
"""


import argparse
from plot2gltf import GLTFGeometryExporter
from utility_io import load_sketch_polyline_data

def export_polylines_gltf(polylines, filename="sketch.gltf"):
    """
    Export a list of polylines to a GLTF file using cylinder strips.
    
    Args:
        polylines (list of list of [x, y, z]): List of polylines as sequences of 3D points.
        filename (str): Output GLTF filename.
    """
    exporter = GLTFGeometryExporter()
    for polyline in polylines:
        exporter.add_cylinder_strips(polyline, radius=0.002)
    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export 3D sketch polylines to GLTF format.')
    parser.add_argument('sketch_file', help='Path to input sketch .obj file')
    parser.add_argument('--output', '-o', default='sketch.gltf', 
                        help='Output GLTF filename (default: sketch.gltf)')
    args = parser.parse_args()

    V, E, P = load_sketch_polyline_data(args.sketch_file)
    polylines = [[V[index] for index in p] for p in P]
    export_polylines_gltf(polylines, args.output)

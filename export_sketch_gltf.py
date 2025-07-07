"""
Convert sketch (.obj) to GLTF.

This script loads a 3D sketch in .obj format, extracts polyline data,
and exports it as a GLTF file using cylinder strips for visualization.

Input:
  - A sketch .obj file with vertices and 'l' polyline lines.

Output:
  - A .gltf file where each polyline is rendered as a 3D cylinder strip.

Usage:
  python export_sketch_gltf.py input.obj -o output.gltf
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

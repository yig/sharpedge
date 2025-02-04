import numpy as np
import argparse

from utility_io import load_sketch_polyline_data, load_obj
from plot2gltf import GLTFGeometryExporter


def export_sketch_surface_gltf(polylines, SV, SF, filename):
    '''
    Export the sketch and the surface to file, as gltf.

    polylines: polyline of the sketch 
    SV: surface vertices
    SF: surface faces
    
    '''
    # Initialize exporter
    exporter = GLTFGeometryExporter()

    POLYLINE_RADIUS = 0.002

    for polyline in polylines:
        exporter.add_cylinder_strips(polyline, radius=POLYLINE_RADIUS,add_spheres=False)
    exporter.add_triangles(SV, SF, color=(0.5, 0.5, 0.5)) 

    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")

    

parser = argparse.ArgumentParser(description='Export sketch and surface to a gltf file')

# Add arguments
parser.add_argument('sketch_file', nargs='?',
                    help='Input file containing normal data (.obj)')
parser.add_argument('surface_file', nargs='?',
                    help='The surface file obj saved, if not provided, no surface_file will be generated.')
parser.add_argument('gltf_file', nargs='?',
                    help='The gltf file obj saved.')

args = parser.parse_args()

sketch_file = args.sketch_file
surface_file = args.surface_file
gltf_file = args.gltf_file


V, E, P  = load_sketch_polyline_data(sketch_file)
polylines = [[V[p] for p in polyline] for polyline in P]
SV, SF = load_obj(surface_file)

export_sketch_surface_gltf(polylines, SV, SF, gltf_file)
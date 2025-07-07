"""
Convert sketch (.obj) and surface SDF (.csv) into a combined GLTF.

This script loads a sketch from an .obj file (containing polylines),
and a signed distance field (SDF) from a .csv file.
It extracts an isosurface from the SDF using marching cubes,
then visualizes both the sketch and the surface together in a single GLTF file.

Inputs:
  - A sketch .obj file with 3D polylines (vertices + lines)
  - A CSV file with SDF values and coordinates (xCoord, yCoord, zCoord, SDF)

Output:
  - A single .gltf file visualizing both the sketch and surface mesh

Usage:
  python export_sketch_surface_gltf.py sketch.obj sdf.csv -o result.gltf
"""

import pandas as pd
import numpy as np
from skimage import measure

import argparse

from plot2gltf import GLTFGeometryExporter
from utility_io import load_sketch_polyline_data

def extract_mesh_from_sdf(csv_file, iso_value=0.0):
    """
    Extract mesh (vertices and faces) from SDF CSV data
    
    Args:
        csv_file: Path to CSV file with columns xCoord, yCoord, zCoord, SDF
        iso_value: Isosurface level to extract (default: 0.0)
    
    Returns:
        sv: numpy array of vertices (N x 3)
        sf: numpy array of faces (M x 3)
    """
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Extract coordinates and SDF values
    x_coords = df['xCoord'].values
    y_coords = df['yCoord'].values
    z_coords = df['zCoord'].values
    sdf_values = df['SDF'].values
    
    # Determine grid dimensions
    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    z_unique = np.sort(np.unique(z_coords))
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    
    # Reshape SDF values into 3D grid
    try:
        # Try standard ordering: for k in nz: for j in ny: for i in nx
        sdf_grid = sdf_values.reshape((nz, ny, nx))
    except ValueError:
        try:
            # Try alternative ordering: for i in nx: for j in ny: for k in nz
            sdf_grid = sdf_values.reshape((nx, ny, nz))
            sdf_grid = np.transpose(sdf_grid, (2, 1, 0))  # Reorder to (nz, ny, nx)
        except ValueError:
            raise ValueError("Cannot reshape data into regular grid")
    
    # Grid spacing
    dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if len(z_unique) > 1 else 1.0
    spacing = (dx, dy, dz)
    
    # Extract isosurface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(
        sdf_grid, 
        level=iso_value, 
        spacing=spacing
    )
    
    # Translate vertices to correct world coordinates
    vertices[:, 0] += x_unique[0]
    vertices[:, 1] += y_unique[0]
    vertices[:, 2] += z_unique[0]
    
    return vertices, faces

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export sketch with SDF surface as GLTF.')
    parser.add_argument('sketch_file', help='Sketch .obj file')
    parser.add_argument('sdf_file', help='CSV file containing SDF grid')
    parser.add_argument('--output', '-o', default='output.gltf', help='Output GLTF file')
    args = parser.parse_args()

    # Load sketch data
    V, E, P = load_sketch_polyline_data(args.sketch_file)

    polylines = [[V[index] for index in p] for p in P]

    # Extract surface mesh from SDF
    sv, sf = extract_mesh_from_sdf(args.sdf_file)

    # Export to GLTF
    export_sketch_surface_gltf(polylines, sv, sf, args.output)


"""
Convert sketch (.obj) and surface (.obj) into a combined GLTF.

This script loads a sketch from an .obj file (containing polylines),
and a surface mesh from another .obj file (containing vertices and faces).
It then visualizes both the sketch and the surface together in a single GLTF file.

Inputs:
  - A sketch .obj file with 3D polylines (vertices + lines)
  - A surface .obj file (vertices + faces)

Output:
  - A single .gltf file visualizing both the sketch and surface mesh

Usage:
  python export_sketch_surface_gltf.py sketch.obj surface.obj -o result.gltf
"""

import numpy as np
import argparse

from plot2gltf import GLTFGeometryExporter
from utility_io import load_sketch_polyline_data

def load_surface_mesh_from_obj(obj_file):
    """
    Load surface mesh (vertices and faces) from OBJ file
    
    Args:
        obj_file: Path to OBJ file with vertices and faces
    
    Returns:
        vertices: numpy array of vertices (N x 3)
        faces: numpy array of faces (M x 3)
    """
    
    vertices = []
    faces = []
    
    with open(obj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex
                # Parse vertex coordinates
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
                
            elif parts[0] == 'f':  # Face
                # Parse face indices (convert from 1-based to 0-based)
                face_indices = []
                for part in parts[1:]:
                    # Handle different face formats: v, v/vt, v/vt/vn, v//vn
                    vertex_index = int(part.split('/')[0]) - 1
                    face_indices.append(vertex_index)
                
                # Only handle triangular faces
                if len(face_indices) == 3:
                    faces.append(face_indices)
                else:
                    print(f"Warning: Skipping non-triangular face with {len(face_indices)} vertices")
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    print(f"Loaded surface mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    return vertices, faces

def export_sketch_surface_gltf(polylines, surface_vertices, surface_faces, filename):
    """
    Export the sketch and the surface to file, as gltf.

    Args:
        polylines: list of polylines from the sketch 
        surface_vertices: surface mesh vertices (N x 3)
        surface_faces: surface mesh faces (M x 3)
        filename: output GLTF filename
    """
    # Initialize exporter
    exporter = GLTFGeometryExporter()

    POLYLINE_RADIUS = 0.002

    # Add sketch polylines as cylinder strips
    for polyline in polylines:
        exporter.add_cylinder_strips(polyline, radius=POLYLINE_RADIUS, add_spheres=False)
    
    # Add surface mesh as triangles
    exporter.add_triangles(surface_vertices, surface_faces, color=(0.5, 0.5, 0.5)) 

    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export sketch with surface mesh as GLTF.')
    parser.add_argument('sketch_file', help='Sketch .obj file (with polylines)')
    parser.add_argument('surface_file', help='Surface .obj file (with mesh faces)')
    parser.add_argument('--output', '-o', default='output.gltf', help='Output GLTF file')
    args = parser.parse_args()

    # Load sketch data
    print(f"Loading sketch from: {args.sketch_file}")
    V, E, P = load_sketch_polyline_data(args.sketch_file)
    polylines = [[V[index] for index in p] for p in P]
    print(f"Loaded sketch: {len(polylines)} polylines")

    # Load surface mesh from OBJ
    print(f"Loading surface mesh from: {args.surface_file}")
    surface_vertices, surface_faces = load_surface_mesh_from_obj(args.surface_file)

    # Export to GLTF
    export_sketch_surface_gltf(polylines, surface_vertices, surface_faces, args.output)
import numpy as np
import argparse
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data

import numpy as np

def remove_duplicates_and_write(V, P, output_filename, tolerance=1e-10):
    """
    Remove duplicate vertices from V, update polylines P, and write to OBJ file.
    
    Args:
        V: nx3 array of vertex coordinates
        P: list of polylines (each polyline is array of vertex indices)
        output_filename: output file path
        tolerance: tolerance for considering vertices as duplicates
    """
    n = len(V)
    vertex_map = {}  # Maps old index to new index
    unique_vertices = []
    
    # Find unique vertices
    for i in range(n):
        found_match = False
        for j, unique_vertex in enumerate(unique_vertices):
            dist = np.linalg.norm(V[i] - unique_vertex)
            if dist < tolerance:
                vertex_map[i] = j
                found_match = True
                break
        
        if not found_match:
            vertex_map[i] = len(unique_vertices)
            unique_vertices.append(V[i])
    
    # Update polylines with new indices
    updated_polylines = []
    for polyline in P:
        new_polyline = [vertex_map[old_idx] for old_idx in polyline]
        
        # Remove consecutive duplicate indices
        filtered_polyline = []
        for idx in new_polyline:
            if not filtered_polyline or idx != filtered_polyline[-1]:
                filtered_polyline.append(idx)
        
        # Keep polylines with at least 2 unique vertices
        if len(filtered_polyline) > 1:
            updated_polylines.append(filtered_polyline)
    
    # Remove duplicate polylines
    unique_polylines = []
    seen_polylines = set()
    
    for polyline in updated_polylines:
        # Convert to tuple for hashing (both forward and reverse)
        poly_tuple = tuple(polyline)
        poly_reverse = tuple(reversed(polyline))
        
        # Check if we've seen this polyline or its reverse
        if poly_tuple not in seen_polylines and poly_reverse not in seen_polylines:
            unique_polylines.append(polyline)
            seen_polylines.add(poly_tuple)
    
    updated_polylines = unique_polylines
    
    # Write to OBJ file
    with open(output_filename, 'w') as f:
        # Write unique vertices
        for vertex in unique_vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write updated polylines
        for polyline in updated_polylines:
            # Convert to 1-indexed
            indices_1based = [str(i + 1) for i in polyline]
            f.write(f"l {' '.join(indices_1based)}\n")
    
    print(f"Removed {n - len(unique_vertices)} duplicate vertices")
    print(f"Removed {len(P) - len(updated_polylines)} duplicate polylines")
    print(f"Written {len(unique_vertices)} vertices and {len(updated_polylines)} polylines to {output_filename}")

# Example usage:
# remove_duplicates_and_write(V, P, "output_clean.obj")


parser = argparse.ArgumentParser(description='Optimize edges to get normals')
parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
parser.add_argument('output_file',nargs='?', help='The curve sketch to write.')

args = parser.parse_args()

curve_file = args.curve_file
output_file = args.output_file

V, E, P = load_sketch_polyline_data(curve_file)

remove_duplicates_and_write(V,P, output_file)
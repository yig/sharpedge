"""
rotate_by_index.py

Interactively rotate a 3D sketch so that a selected line aligns vertically.
"""


import argparse
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data
from utility_viewer_ps import plot_cdt_skecth_with_polylines

def rotate_vertices(V, rotation_matrix):
    assert rotation_matrix.shape == (3, 3), "rotation_matrix must be 3x3"
    return V @ rotation_matrix.T

def rotation_matrix_align_vectors(v0, v1, to_vec):
    from_vec = v1 - v0
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)
    
    if np.isclose(dot, 1.0):
        return np.eye(3)
    elif np.isclose(dot, -1.0):
        perp = np.array([1, 0, 0]) if abs(from_vec[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(from_vec, perp)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(dot)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def center_vertices(V):
    centroid = np.mean(V, axis=0)
    return V - centroid

def save_sketch_polyline_data(filename, V, E, P):
    with open(filename, 'w') as f:
        for vertex in V:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for polyline in P:
            polyline_str = " ".join(map(str, polyline + 1))
            f.write(f"l {polyline_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rotate sketch using two vertex indices.')
    parser.add_argument('curve_file', help='The curve sketch to load.')
    parser.add_argument('output_file', nargs='?', help='The curve sketch to write.')
    args = parser.parse_args()

    curve_file = args.curve_file
    output_file = args.output_file or curve_file  # overwrite if not provided

    V, E, P = load_sketch_polyline_data(curve_file)
    V_centered = center_vertices(V)
   
    print("Original orientation:")
    plot_sketch_data(V, P)

    plot_cdt_skecth_with_polylines(V, E)

    # Ask for indices
    try:
        idx0 = input("Enter first vertex index (or 'q' to quit): ")
        if idx0.strip().lower() == 'q':
            print("Quit without modifying.")
            exit(0)
        idx1 = input("Enter second vertex index (or 'q' to quit): ")
        if idx1.strip().lower() == 'q':
            print("Quit without modifying.")
            exit(0)

        idx0 = int(idx0.strip())
        idx1 = int(idx1.strip())

        v0 = V[idx0]
        v1 = V[idx1]

    except Exception as e:
        print("Invalid input, exiting.")
        exit(1)

    r = rotation_matrix_align_vectors(v0, v1, to_vec=(0, 1, 0))
    V_rotated = rotate_vertices(V, r)


    print("Rotated (vertical) orientation:")
    plot_sketch_data(V_rotated, P)

    save_sketch_polyline_data(output_file, V_rotated, E, P)
    print(f"Saved rotated sketch to {output_file}")

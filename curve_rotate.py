import argparse
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data

def rotate_vertices(V, rotation_matrix):
    """
    Rotate vertices using an arbitrary rotation matrix.

    Args:
        V (ndarray): (N, 3) array of vertex coordinates.
        rotation_matrix (ndarray): (3, 3) rotation matrix.

    Returns:
        ndarray: Rotated vertex coordinates.
    """
    assert rotation_matrix.shape == (3, 3), "rotation_matrix must be 3x3"
    return V @ rotation_matrix.T

def rotation_matrix_align_vectors(v0, v1, to_vec):
    """
    Construct a rotation matrix that rotates `from_vec` to align with `to_vec`.

    Args:
        from_vec (ndarray): 3D vector to rotate from.
        to_vec (ndarray): 3D target direction to rotate to.

    Returns:
        ndarray: (3, 3) rotation matrix.
    """

    from_vec = v1 - v0
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)
    
    if np.isclose(dot, 1.0):
        return np.eye(3)  # Already aligned
    elif np.isclose(dot, -1.0):
        # 180 degrees: rotate around any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(from_vec[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(from_vec, perp)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(dot)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def center_vertices(V):
    """Center vertices around origin"""
    centroid = np.mean(V, axis=0)
    return V - centroid

def save_sketch_polyline_data(filename, V, E, P):
    """
    Save sketch polyline data to file in simple v and ls format.
    
    Args:
        filename (str): Output file path
        V (ndarray): nx3 array of vertex coordinates
        E (list/array): Edge data (not used in this format)
        P (list): List of polyline indices
    """
    with open(filename, 'w') as f:
        # Write vertices
        for i, vertex in enumerate(V):
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        # Write polylines as line segments (ls)
        for i, polyline in enumerate(P):
            polyline_str = " ".join(map(str, polyline+1))
            f.write(f"l {polyline_str}\n")

parser = argparse.ArgumentParser(description='Optimize edges to get normals')
parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
parser.add_argument('output_file',nargs='?', help='The curve sketch to write.')

args = parser.parse_args()

curve_file = args.curve_file
output_file = args.output_file

V, E, P = load_sketch_polyline_data(curve_file)

# Center the vertices first
V_centered = center_vertices(V)


v0 = V[83]
v1 = V[84]
r = rotation_matrix_align_vectors(v0, v1, to_vec=(0, -1, 0))

V_rotated = rotate_vertices(V, r)
# Rotate to make it vertical - try different combinations to get the desired orientation
# Common rotations for making objects vertical:

# If the above doesn't give the right orientation, try these alternatives:
# V_rotated = rotate_vertices(V_centered, axis='x', angle_degrees=90)  # Rotate 90° around X-axis
# V_rotated = rotate_vertices(V_centered, axis='y', angle_degrees=90)  # Rotate 90° around Y-axis

# For compound rotations (if needed):
# V_rotated = rotate_vertices(V_centered, axis='x', angle_degrees=90)
# V_rotated = rotate_vertices(V_rotated, axis='z', angle_degrees=45)

# Plot both original and rotated versions for comparison
print("Original orientation:")
plot_sketch_data(V, P)

print("Rotated (vertical) orientation:")
plot_sketch_data(V_rotated, P)

# Update V with the rotated vertices
V = V_rotated


save_sketch_polyline_data(output_file, V_rotated, E, P)

V, E, P  = load_sketch_polyline_data(output_file)

plot_sketch_data(V, P)



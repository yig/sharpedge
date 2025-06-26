import argparse
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data

def rotate_vertices(V, axis='x', angle_degrees=90):
    """
    Rotate vertices around a specified axis.
    
    Args:
        V (ndarray): nx3 array of vertex coordinates
        axis (str): Rotation axis ('x', 'y', or 'z')
        angle_degrees (float): Rotation angle in degrees
    
    Returns:
        ndarray: Rotated vertex coordinates
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Create rotation matrices
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Apply rotation
    V_rotated = np.dot(V, rotation_matrix.T)
    return V_rotated

def center_vertices(V):
    """Center vertices around origin"""
    centroid = np.mean(V, axis=0)
    return V - centroid

parser = argparse.ArgumentParser(description='Optimize edges to get normals')
parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
args = parser.parse_args()

curve_file = args.curve_file
V, E, P = load_sketch_polyline_data(curve_file)

# Center the vertices first
V_centered = center_vertices(V)

# Rotate to make it vertical - try different combinations to get the desired orientation
# Common rotations for making objects vertical:
V_rotated = rotate_vertices(V_centered, axis='z', angle_degrees=30)  # Rotate 90° around Z-axis

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
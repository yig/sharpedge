import numpy as np
import argparse
import matplotlib.pyplot as plt
from utility_io import load_sketch_polyline_data
from t2f_remove_duplicates import remove_duplicates_and_write

def normalize_vertices(vertices):
    """
    Normalize vertices to:
    1. Center at origin (0,0,0)
    2. Scale so bounding box diagonal = 1
    """
    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    print(f"Original bounding box:")
    print(f" Min: [{min_coords[0]:.6f}, {min_coords[1]:.6f}, {min_coords[2]:.6f}]")
    print(f" Max: [{max_coords[0]:.6f}, {max_coords[1]:.6f}, {max_coords[2]:.6f}]")
    
    # Calculate center and size
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    diagonal = np.linalg.norm(size)
    print(f" Center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f" Size: [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
    print(f" Diagonal: {diagonal:.6f}")
    
    # Step 1: Translate to center at origin
    centered_vertices = vertices - center
    
    # Step 2: Scale so diagonal = 1
    if diagonal > 0:
        scale_factor = 1.0 / diagonal
        normalized_vertices = centered_vertices * scale_factor
    else:
        normalized_vertices = centered_vertices
        scale_factor = 1.0
    
    # Verify normalization
    new_min = np.min(normalized_vertices, axis=0)
    new_max = np.max(normalized_vertices, axis=0)
    new_center = (new_min + new_max) / 2
    new_size = new_max - new_min
    new_diagonal = np.linalg.norm(new_size)
    
    print(f"\nNormalized bounding box:")
    print(f" Min: [{new_min[0]:.6f}, {new_min[1]:.6f}, {new_min[2]:.6f}]")
    print(f" Max: [{new_max[0]:.6f}, {new_max[1]:.6f}, {new_max[2]:.6f}]")
    print(f" Center: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
    print(f" Size: [{new_size[0]:.6f}, {new_size[1]:.6f}, {new_size[2]:.6f}]")
    print(f" Diagonal: {new_diagonal:.6f}")
    print(f" Scale factor applied: {scale_factor:.6f}")
    
    return normalized_vertices, center, scale_factor

def plot_sketch_data(V, P, save_path=None):
    """
    Plot a list of 3D polylines and optionally save the figure.
    Args:
        V (ndarray): nx3 array of vertex coordinates.
        P (list): List of arrays containing vertex indices for each polyline.
        save_path (str or None): If given, save the figure to this path as PNG.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    V = np.asarray(V)
    for index, polyline in enumerate(P):
        points = [V[i] for i in polyline]
        pts = np.asarray(points)
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        ax.plot(xs, ys, zs)
        ax.scatter(xs, ys, zs, s=5)
    
    # Set equal aspect ratio for all axes
    plt.axis('off')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='png')
        print(f"Figure saved to: {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('output_file', nargs='?', help='The curve sketch to write.')
    parser.add_argument('--png', type=str, help='Path to save PNG visualization (optional)')
    
    args = parser.parse_args()
    curve_file = args.curve_file
    output_file = args.output_file
    png_path = args.png
    
    # Load the sketch data
    V, E, P = load_sketch_polyline_data(curve_file)
    
    # Normalize vertices
    normalized_vertices, original_center, scale_factor = normalize_vertices(V)
    
    # Create PNG visualization if requested
    if png_path:
        plot_sketch_data(normalized_vertices, P, save_path=png_path)
    
    # Save processed data if output file specified
    if output_file is not None:
        remove_duplicates_and_write(normalized_vertices, P, output_file)
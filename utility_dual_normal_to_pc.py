import numpy as np
import polyscope as ps 
from scipy.spatial import cKDTree
import argparse
from pathlib import Path

from utility_io import read_two_normal
from utility_viewer_ps import plot_two_normals

def resample_dual_normals(V, E, normals, sample_length=0.01, proximity_threshold=5e-3):
    '''
    Sample points along edges, keeping the two most different normals for each point.
    Filters out points that are too close to existing points.
    
    Parameters:
        V: vertices array
        E: edges array (pairs of vertex indices)
        normals: dictionary with keys (edge_idx, which_edge) and normal vectors as values
        sample_length: desired length between samples
        proximity_threshold: minimum allowed distance between points
        
    Returns:
        points: array of filtered sampled points
        normals0: array of first normal vectors (most different)
        normals1: array of second normal vectors (most different)
    '''
    # Dictionary to store point with its collected normals
    points_dict = {}
    
    def add_point_with_check(point, edge_idx):
        """Helper function to add point and its normals"""
        point_key = tuple(point)
        
        # Get normals for this edge
        normal0 = normals.get((edge_idx, 0), np.zeros(3))
        normal1 = normals.get((edge_idx, 1), np.zeros(3))
        
        # Skip if normals aren't valid
        if np.linalg.norm(normal0) == 0 or np.linalg.norm(normal1) == 0:
            return False
            
        # Normalize the normals
        normal0 = normal0 / np.linalg.norm(normal0)
        normal1 = normal1 / np.linalg.norm(normal1)
        
        if not points_dict:  # First point, add directly
            points_dict[point_key] = [normal0, normal1]
            return True
            
        # Create KD-tree from existing points
        existing_points = np.array(list(points_dict.keys()))
        tree = cKDTree(existing_points)
        
        # Check if point is too close to any existing point
        distances, indices = tree.query(point, k=1)
        
        if distances > proximity_threshold:
            # If point is far enough, add it with its normals
            points_dict[point_key] = [normal0, normal1]
            return True
        else:
            # If point is close, add its normals to the existing point's collection
            closest_point = tuple(existing_points[indices])
            existing_normals = points_dict[closest_point]
            
            # Add new normals to the collection if they're not duplicates
            for normal in [normal0, normal1]:
                # Check if this normal is significantly different from existing ones
                is_new = True
                for existing_normal in existing_normals:
                    if abs(np.dot(normal, existing_normal)) > 0.95:  # Similar if cos(angle) > 0.95
                        is_new = False
                        break
                
                if is_new:
                    existing_normals.append(normal)
            
            return False
    
    # Process each edge
    for edge_idx, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)
        
        # Calculate number of samples
        n = max(2, int(np.ceil(edge_length / sample_length)))
        
        # Add endpoints if they pass proximity check
        add_point_with_check(p0, edge_idx)
        add_point_with_check(p1, edge_idx)
        
        if n == 2:
            # Add midpoint if needed
            mid_point = (p0 + p1) / 2
            add_point_with_check(mid_point, edge_idx)
        else:
            # Generate sample points along the edge
            t = np.linspace(0, 1, n)[1:-1]  # Exclude endpoints
            for ti in t:
                point = p0 + ti * edge_vec
                add_point_with_check(point, edge_idx)
    
    # Find the two most different normals for each point
    points = []
    normals0 = []
    normals1 = []
    
    for point_key, normal_list in points_dict.items():
        points.append(list(point_key))
        
        if len(normal_list) <= 2:
            # If we only have two or fewer normals, use them
            while len(normal_list) < 2:
                # If we have fewer than 2 normals, duplicate the last one
                normal_list.append(normal_list[-1] if normal_list else np.array([0, 0, 1]))
                
            normals0.append(normal_list[0])
            normals1.append(normal_list[1])
        else:
            # Find the pair with the largest angle between them
            max_angle = -1
            best_pair = (0, 1)
            
            for i in range(len(normal_list)):
                for j in range(i+1, len(normal_list)):
                    # Compute the angle between normals (using dot product)
                    dot_product = np.dot(normal_list[i], normal_list[j])
                    angle = np.arccos(max(-1, min(1, dot_product)))  # Clamp to [-1, 1]
                    
                    if angle > max_angle:
                        max_angle = angle
                        best_pair = (i, j)
            
            normals0.append(normal_list[best_pair[0]])
            normals1.append(normal_list[best_pair[1]])
    
    return np.array(points), np.array(normals0), np.array(normals1)

# Convert the dictionary format normals to arrays for plotting
def convert_normals_for_plotting(E, normals):
    """
    Convert normals from dictionary format to two arrays for plotting.
    
    Args:
        E: (m,2) array of edge vertex pairs
        normals: Dictionary with keys (edge_idx, which_edge)
        
    Returns:
        N1: (m,3) array of first normal vectors
        N2: (m,3) array of second normal vectors
    """
    m = len(E)
    N1 = np.zeros((m, 3))
    N2 = np.zeros((m, 3))
    
    for i in range(m):
        if (i, 0) in normals:
            N1[i] = normals[(i, 0)]
        if (i, 1) in normals:
            N2[i] = normals[(i, 1)]
    
    return N1, N2

def visualize_resampled_points_with_dual_normals(points, normals0, normals1, point_radius=0.002, vector_length=0.10):
    """
    Visualize resampled points with their dual normal vectors using Polyscope.
    
    Args:
        points: array of point coordinates
        normals0: first set of normal vectors (same length as points)
        normals1: second set of normal vectors (same length as points)
        point_radius: radius for point visualization
        vector_length: length for normal vector visualization
    """
    ps.init()
    
    # Register the points
    ps_points = ps.register_point_cloud("resampled_points", points)
    ps_points.set_color((0.5, 0.5, 0.5))  # Gray color for points
    ps_points.set_radius(point_radius)
    
    # Add the first set of normals (green)
    ps_points.add_vector_quantity("normal_vectors1", normals0,
                                 enabled=True,
                                 color=(0.0, 1.0, 0.0),
                                 length=vector_length)
    
    # Add the second set of normals (red)
    ps_points.add_vector_quantity("normal_vectors2", normals1,
                                 enabled=True,
                                 color=(1.0, 0.0, 0.0),
                                 length=vector_length)
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # Show the window
    ps.show()

def write_pc_dual_normals(points, normals0, normals1, filename):
    """
    Write points and dual normals to a file in the format:
    v x y z
    vn nx0 ny0 nz0
    vn nx1 ny1 nz1 
    
    Parameters:
        points: Array of 3D points with shape (n, 3)
        normals0: Array of first normal vectors with shape (n, 3)
        normals1: Array of second normal vectors with shape (n, 3)
        filename: Output file name (string)
    """
    with open(filename, 'w') as f:
        # Write each point with its two normals
        for i in range(len(points)):
            # Write vertex (point)
            p = points[i]
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            # Write first normal (lowercase 'vn')
            n0 = normals0[i]
            f.write(f"vn {n0[0]:.6f} {n0[1]:.6f} {n0[2]:.6f}\n")
            
            # Write second normal (uppercase 'Vn')
            n1 = normals1[i]
            f.write(f"vn {n1[0]:.6f} {n1[1]:.6f} {n1[2]:.6f}\n")
    
    # Print summary
    print(f"Wrote to {filename}:")
    print(f"- {len(points)} points")
    print(f"- {len(normals0)} first normals (vn)")
    print(f"- {len(normals1)} second normals (vn)")



parser = argparse.ArgumentParser(description='Edge normal file to point normal file')

# Add arguments
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.normal)')

args = parser.parse_args()

normal_file = args.normal_file


V, E, N = read_two_normal(normal_file)

# print(V, E, N)


N0, N1 = convert_normals_for_plotting(E, N)
# print(points, normals0, normals1)

plot_two_normals(V, E, N0, N1)

points, normals0, normals1 = resample_dual_normals(V, E, N)

visualize_resampled_points_with_dual_normals(points, normals0, normals1)

curve_name = Path(normal_file).stem
filename = f'signed-heat-3d/data/{curve_name}.pc'
write_pc_dual_normals(points, normals0, normals1, filename)
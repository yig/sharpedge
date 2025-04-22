from utility_io import load_normal_data
from marching_cube import plot_points_normal, resample_for_points_normal

import argparse

from pathlib import Path
import numpy as np 
from scipy.spatial import cKDTree


def resample_for_points_normal_v2(V, E, N, sample_length=0.01, proximity_threshold=5e-3):
    '''
    Sample the edge normal on the points using the sample length.
    Filters out points that are too close to existing points while keeping exact duplicates.
    For points that are exact duplicates, maintains separate normals for each edge.
    
    Parameters:
        V: vertices array
        E: edges array (pairs of vertex indices)
        N: normals array
        sample_length: desired length between samples
        proximity_threshold: minimum allowed distance between points
    Returns:
        points: array of sampled points (may include duplicates at exact same position)
        normals: array of corresponding normal vectors (one for each point)
    '''
    points_dict = {}  # Dictionary to store point-normal pairs
    
    def add_point_with_check(point, normal):
        """Helper function to add point or check proximity
        - Keeps exact duplicate points with their separate normals
        - Filters out points that are too close (but not exact matches)
        """
        point_tuple = tuple(point)
        
        if not points_dict:  # First point, add directly
            points_dict[point_tuple] = [normal]
            return True
            
        # Check if point already exists exactly (common in mesh intersections)
        if point_tuple in points_dict:
            # For exact matches, keep the point but add the normal to list
            points_dict[point_tuple].append(normal)
            return True  # Consider this a "keep" scenario
            
        # Create KD-tree from existing points for proximity search
        existing_points = np.array(list(points_dict.keys()))
        tree = cKDTree(existing_points)
        
        # Check if point is too close to any existing point
        distances, indices = tree.query(point, k=1)
        if distances > proximity_threshold:
            # Point is far enough from existing points, add it
            points_dict[point_tuple] = [normal]
            return True
        else:
            # If point is close but not exact, don't keep it
            # No need to add the normal to the closest point
            return False
    
    # Process each edge
    for index, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)
        n = max(2, int(np.ceil(edge_length / sample_length)))
        normal = N[index]
        
        # Skip edges with zero normal
        if np.linalg.norm(normal) < 1e-10:
            continue
            
        # Add endpoints
        add_point_with_check(p0, normal)
        add_point_with_check(p1, normal)
        
        # Add interior points based on sampling density
        if n > 2:
            # Generate sample points along the edge
            t = np.linspace(0, 1, n)[1:-1]  # Exclude endpoints
            for ti in t:
                point = p0 + ti * edge_vec
                add_point_with_check(point, normal)
        elif n == 2 and edge_length > proximity_threshold * 2:
            # Add midpoint for short edges, if it's not too close to endpoints
            mid_point = (p0 + p1) / 2
            add_point_with_check(mid_point, normal)
    
    # Convert dictionary back to separate arrays
    # For points with multiple normals, create separate entries for each normal
    points = []
    normals = []
    
    for point_key, normal_list in points_dict.items():
        point = list(point_key)
        
        # For each normal associated with this point, create a separate entry
        for normal in normal_list:
            points.append(point)
            
            # Ensure the normal is normalized
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            normals.append(normal)
    
    return np.array(points), np.array(normals)


def resample_for_points_normal_v3(V, E, N, sample_length=0.01, proximity_threshold=5e-3):
    '''
    Sample the edge normal on the points using the sample length.
    Filters out points that are too close to existing points while maintaining corresponding normals.
    Takes only the first normal for each point instead of averaging multiple normals.
    
    Parameters:
    V: vertices array
    E: edges array (pairs of vertex indices)
    N: normals array
    sample_length: desired length between samples
    proximity_threshold: minimum allowed distance between points
    
    Returns:
    points: array of filtered sampled points
    normals: array of corresponding normal vectors
    '''
    points_dict = {}  # Dictionary to store point-normal pairs
    
    def add_point_with_check(point, normal):
        """Helper function to add point only if it's not too close to existing points"""
        point_tuple = tuple(point)
        
        if not points_dict:  # First point, add directly
            points_dict[point_tuple] = normal
            return True
            
        # Create KD-tree from existing points
        existing_points = np.array(list(points_dict.keys()))
        tree = cKDTree(existing_points)
        
        # Check if point is too close to any existing point
        distances, indices = tree.query(point, k=1)
        if distances > proximity_threshold:
            # Point is far enough from existing points, add it
            points_dict[point_tuple] = normal
            return True
        # Point is too close, but we already have a normal for the closest point
        # so we don't modify anything
        return False
    
    for index, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)
        n = max(2, int(np.ceil(edge_length / sample_length)))
        normal = N[index]
        
        # Only process edges with valid normals
        if np.linalg.norm(normal) > 0:
            # Add endpoints if they pass proximity check
            add_point_with_check(p0, normal)
            add_point_with_check(p1, normal)
            
            if n == 2:
                # Add midpoint if needed
                mid_point = (p0 + p1) / 2
                add_point_with_check(mid_point, normal)
            else:
                # Generate sample points along the edge
                t = np.linspace(0, 1, n)[1:-1]  # Exclude endpoints
                for ti in t:
                    point = p0 + ti * edge_vec
                    add_point_with_check(point, normal)
    
    # Convert dictionary back to separate arrays
    points = list(points_dict.keys())
    normals = list(points_dict.values())
    
    return np.array(points), np.array(normals)

def write_pc_data(points, normals, filename):
    """
    Write points and normals to a file in the format:
    v x y z
    vn nx ny nz

    :param points: List of 3D points [(x, y, z), ...]
    :param normals: List of normal vectors [(nx, ny, nz), ...]
    :param filename: Output file name (string)
    """
    assert len(points) == len(normals), "Points and normals must have the same length"
    
    with open(filename, 'w') as file:
        for (x, y, z), (nx, ny, nz) in zip(points, normals):
            file.write(f"v {x} {y} {z}\n")
            file.write(f"vn {nx} {ny} {nz}\n")



parser = argparse.ArgumentParser(description='Edge normal file to point normal file')

# Add arguments
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.normal)')

args = parser.parse_args()

normal_file = args.normal_file
# pc_file = args.pc_file

# normal_file = 'debug_normals/cylinder_c_1n.normal'

V, E, N = load_normal_data(normal_file)
print(V, E, N)
points, normals = resample_for_points_normal(V, E, N, 0.01)



print('len(points)', len(points))
print('len(normals)', len(normals))


plot_points_normal( points, normals )

curve_name = Path(normal_file).stem

# print(curve_file)

pc_file = 'signed-heat-3d/data/' + curve_name + '.pc'

write_pc_data(points, normals, pc_file)

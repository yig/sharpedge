from utility_io import load_normal_data, export_obj
from utility_viewer_ps import plot_normal_data

import numpy as np 
import polyscope as ps 
import igl
import argparse
from collections import defaultdict
import polyscope.imgui as psim
from scipy.spatial.distance import pdist
import gpytoolbox
import sys 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import scipy


def n_for_segment(segment, target_length):
    """Calculate number of sample points needed for a line segment.

    Args:
        segment: Tuple of two points (p0, p1) defining line segment
        target_length: Desired length between samples

    Returns:
        int: Number of sample points (minimum 1)
    """
    p0, p1 = np.asarray(segment)
    segment_length = np.linalg.norm(p0 - p1)

    if segment_length < target_length:
        return 1
        
    n = segment_length / target_length
    return int(np.ceil(n - 0.5))

def resample_for_points_normal(V, E, N, sample_length=0.01, proximity_threshold=5e-3):
    '''
    Sample the edge normal on the points using the sample length.
    Filters out points that are too close to existing points while maintaining corresponding normals.
    
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
        if not points_dict:  # First point, add directly
            points_dict[tuple(point)] = [normal]
            return True
            
        # Create KD-tree from existing points
        existing_points = np.array(list(points_dict.keys()))
        tree = cKDTree(existing_points)
        
        # Check if point is too close to any existing point
        distances, indices = tree.query(point, k=1)
        if distances > proximity_threshold:
            points_dict[tuple(point)] = [normal]
            return True
        else:
            # If point is close, add its normal to the existing point
            closest_point = tuple(existing_points[indices])
            points_dict[closest_point].append(normal)
            return False
    
    for index, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)
        n = max(2, int(np.ceil(edge_length / sample_length)))
        normal = N[index]
        
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
    
    # Convert dictionary back to separate arrays and average normals
    points = []
    normals = []
    for point_key, normal_list in points_dict.items():
        points.append(list(point_key))
        # Average the normals and normalize
        avg_normal = np.mean(normal_list, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 0:
            avg_normal = avg_normal / norm
        normals.append(avg_normal)
    
    return np.array(points), np.array(normals)


def generate_bounding_box_points(V, scale_factor=2):
    """Generate axis-aligned bounding box vertices around 3D points.
    Args:
        V: Nx3 array of 3D vertex coordinates
        scale_factor: Factor to scale bounding box dimensions (default: 1.5)
    Returns:
        8x3 array of bounding box vertex coordinates
    """
    min_coords = np.min(V, axis=0)
    max_coords = np.max(V, axis=0)
    center = (min_coords + max_coords) / 2
    dimensions = (max_coords - min_coords) * scale_factor / 2
    bbox_vertices = np.array([
        [center[0] - dimensions[0], center[1] - dimensions[1], center[2] - dimensions[2]],
        [center[0] - dimensions[0], center[1] - dimensions[1], center[2] + dimensions[2]],
        [center[0] - dimensions[0], center[1] + dimensions[1], center[2] - dimensions[2]],
        [center[0] - dimensions[0], center[1] + dimensions[1], center[2] + dimensions[2]],
        [center[0] + dimensions[0], center[1] - dimensions[1], center[2] - dimensions[2]],
        [center[0] + dimensions[0], center[1] - dimensions[1], center[2] + dimensions[2]], 
        [center[0] + dimensions[0], center[1] + dimensions[1], center[2] - dimensions[2]],
        [center[0] + dimensions[0], center[1] + dimensions[1], center[2] + dimensions[2]]
    ])
    return bbox_vertices

def generate_matched_cube_mesh(num_divisions, bbox_vertices):
    """
    Generate a regular cube mesh that matches the scale and position of given bounding box vertices.
    
    Args:
        num_divisions: Number of divisions for the cube mesh
        bbox_vertices: 8x3 array of bounding box vertex coordinates
        
    Returns:
        GV: Vertex coordinates matching bbox scale and position
        F: Face connectivity
    """
    # Get initial regular cube mesh (this gives points in [0,1])
    GV, F = gpytoolbox.regular_cube_mesh(num_divisions)
    
    # Calculate bounding box dimensions and center
    bbox_min = np.min(bbox_vertices, axis=0)
    bbox_max = np.max(bbox_vertices, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_dimensions = bbox_max - bbox_min
    
    # Transform the regular mesh to match the bounding box:
    # 1. First center the regular mesh at origin (it starts at [0,1])
    GV = GV - 0.5
    
    # 2. Scale to match bounding box dimensions
    GV = GV * bbox_dimensions
    
    # 3. Translate to bounding box center
    GV = GV + bbox_center
    
    return GV, F

def plot_points_normal(points, normals):
    """
    Plot points and their normal vectors using Polyscope
    
    Args:
        points: (n,3) array of point coordinates
        normals: (n,3) array of normal vectors
    """
    ps.init()

    # ps.remove_all_structures()
    
    # Register the points
    ps_points = ps.register_point_cloud("points", points)
    ps_points.set_color((0.0, 0.0, 1.0))  # Blue color for points
    ps_points.set_radius(0.002)  # Point size
    
    # Add normal vectors as a vector quantity
    ps_points.add_vector_quantity(
        "normal_vectors", 
        normals,
        enabled=True,
        color=(0.0, 1.0, 0.0),  # Green color for normals
        length=0.10             # Vector length scaling
    )
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # Show the visualization window
    ps.show()


def plot_points_wns(tet_vertices, wns):
    """
    Plot points with their wn values as text labels using Matplotlib
    
    Args:
        tet_vertices: (n,3) array of tet_vertices coordinates
        wns: (n,1) array of wn values
    """
    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot tet_vertices
    ax.scatter(tet_vertices[:, 0], tet_vertices[:, 1], tet_vertices[:, 2], 
              c='blue', s=20)
    
    # Add text labels for wn values
    for i, (point, wn) in enumerate(zip(tet_vertices, wns)):
        # Offset the text slightly from the point
        offset = 0.01  # Adjust this value based on your scale
        ax.text(point[0] + offset, point[1] + offset, point[2] + offset,
                f'{wn:.2f}', size=8)
     
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points with WN Values')
    
 
    # Show the plot
    plt.axis('off')
    plt.axis('equal')
    plt.show()

def plot_points_large_wns(tet_vertices, wns):
    """
    Plot points with different colored text labels for extreme positive and negative winding numbers
    
    Args:
        tet_vertices: (n,3) array of tet_vertices coordinates
        wns: (n,1) array of wn values
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot all vertices in light gray
    ax.scatter(tet_vertices[:, 0], tet_vertices[:, 1], tet_vertices[:, 2], 
              c='lightgray', s=5)
    
    # Calculate average absolute winding number
    avg_wn = np.mean(np.abs(wns))
    threshold = 20 * avg_wn
    
    # Add text labels for extreme winding numbers with different colors
    for i, (point, wn) in enumerate(zip(tet_vertices, wns)):
        if abs(wn) > threshold:
            # Offset the text slightly from the point
            offset = 0.01  # Adjust this value based on your scale
            
            # Use red for positive extreme values, blue for negative
            color = 'red' if wn > 0 else 'blue'
            
            # Plot the point in the same color as its label
            ax.scatter(point[0], point[1], point[2], 
                      c=color, s=20)
            
            # Add the text label
            ax.text(point[0] + offset, point[1] + offset, point[2] + offset,
                   f'{wn:.2f}', size=18, color=color)
     
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points with Extreme WN Values\n(Red: Positive, Blue: Negative)')
    
    # Show the plot
    plt.axis('off')
    plt.axis('equal')
    plt.show()


def find_optimal_scale(points, normals, target_mean=1.0, points_area = np.pi, tolerance=0.01, max_iterations=50):
    """
    Find the optimal scale for base_points_area to achieve a target mean winding number.
    
    Args:
        points (ndarray): Point coordinates
        normals (ndarray): Normal vectors
        target_mean (float): Target mean winding number (default: 1.0)
        tolerance (float): Acceptable difference from target mean (default: 0.01)
        max_iterations (int): Maximum number of iterations (default: 50)
        
    Returns:
        tuple: (optimal_scale, mean_wn, base_points_area)
            - optimal_scale: The scale that achieves the target mean
            - mean_wn: The achieved mean winding number
            - base_points_area: The scaled base points area
    """
    # Binary search parameters
    scale_min = 0.01
    scale_max = 10.0
    
    for i in range(max_iterations):
        # Try current scale
        scale = (scale_min + scale_max) / 2
        base_points_area = np.ones((points.shape[0],)) / points.shape[0] * points_area * scale
        
        # Calculate winding numbers
        wns = igl.fast_winding_number_for_points(points, normals, base_points_area, points)
        current_mean = np.mean(wns)
        
        # Check if we're close enough
        if abs(current_mean - target_mean) < tolerance:
            return scale, current_mean, base_points_area
        
        # Adjust search range
        if current_mean < target_mean:
            scale_min = scale
        else:
            scale_max = scale
            
        # Debug info
        print(f"Iteration {i+1}: scale = {scale:.4f}, mean = {current_mean:.4f}")
    
    # If we reach max iterations, return best attempt
    return scale, current_mean, base_points_area


def area_for_isovalue( points, normals, points_area, grid_vertices, box_division, isovalue ):
    wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, isovalue)
    area = igl.doublearea( SV, SF ) / 2.0 
    return sum( area )


def calculate_genus(vertices, faces):
    # Count vertices (V)
    V = len(vertices)
    
    # Count faces (F)
    F = len(faces)
    
    # Count edges (E)
    # For a triangle mesh, we need to account for shared edges
    # Create a set of edges (sorted vertex pairs)
    edges = set()
    for face in faces:
        # Add all three edges of the triangle
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    E = len(edges)
    
    # Calculate Euler characteristic
    euler_char = V - E + F
    
    # Calculate genus
    genus = (2 - euler_char) // 2
    
    return genus


def calculate_area_and_genus(points, normals, cube_vertices, box_division, isovalue_start, isovalue_end, isovalue_step = 0.1):
    '''
    '''

    points_area = np.ones((points.shape[0], )) / points.shape[0] * np.pi
    wns = igl.fast_winding_number_for_points(points, normals, points_area , cube_vertices)

    areas = []
    genera = []
    isovalues = []

    for isovalue in np.arange(isovalue_start, isovalue_end, isovalue_step):

        SV, SF = gpytoolbox.marching_cubes(wns, cube_vertices, box_division, box_division, box_division, isovalue)
        # print(SV, SF)

        if len(SV) > 0 and len(SF) > 0:
            area = sum ( igl.doublearea(SV, SF) / 2.0 )

        else:
            # Handle the empty case
            area = 0  # or whatever default value makes sense
            print("Warning: Empty mesh detected")
            return areas, genera, isovalues
    
        areas.append( area )
        genus = calculate_genus(SV, SF)
        genera.append( genus)
        isovalues.append( isovalue )

    return areas, genera, isovalues



def plot_area_genus_relationship(areas, genera, isovalues, figsize=(10, 6)):
    """
    Plot the relationship between area and genus with isovalue annotations.
    
    Parameters:
    -----------
    areas : array-like
        List of area values
    genera : array-like
        List of genus values
    isovalues : array-like
        List of isovalue values that generated each (area, genus) pair
    figsize : tuple, default=(10, 6)
        Figure size in inches (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot main data
    plt.plot(areas, genera, 'b-', label='Genus')
    plt.scatter(areas, genera, color='blue', s=30)
    
    # Add isovalue annotations to each point
    for i, (area, genus, iso) in enumerate(zip(areas, genera, isovalues)):
        plt.annotate(f'{iso:.1f}', 
                    (area, genus),
                    xytext=(5, 5),  # 5 points offset
                    textcoords='offset points',
                    fontsize=8)
    
    # Add labels and title
    plt.xlabel('Area')
    plt.ylabel('Genus')
    plt.title('Relationship between Area and Genus\n(numbers indicate isovalues)')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    plt.show()


def find_minimal_surface_genus_zero(areas, genera, isovalues):
    """
    Find the minimal surface area and corresponding isovalue among all cases where genus = 0
    
    Parameters:
    areas (list): List of surface areas
    genera (list): List of genus values
    isovalues (list): List of isovalue parameters
    
    Returns:
    tuple: (minimal_area, corresponding_isovalue)
            Returns (None, None) if no genus 0 cases are found
    """
    # Convert to numpy arrays for easier handling
    areas = np.array(areas)
    genera = np.array(genera)
    isovalues = np.array(isovalues)
    
    # Find indices where genus is 0
    genus_zero_mask = (genera == 0)
    
    # If no genus 0 cases found, return None
    if not np.any(genus_zero_mask):
        print("No cases with genus 0 found")
        return None, None
    
    # Get areas and isovalues for genus 0 cases
    genus_zero_areas = areas[genus_zero_mask]
    genus_zero_isovalues = isovalues[genus_zero_mask]
    
    # Find the minimum area and its index
    min_area_idx = np.argmin(genus_zero_areas)
    min_area = genus_zero_areas[min_area_idx]
    corresponding_isovalue = genus_zero_isovalues[min_area_idx]
    
    return min_area, corresponding_isovalue


def find_mesh_holes(vertices, faces):
    """
    Find holes in a mesh by detecting boundary edges and grouping them into loops.
    
    Args:
        vertices: np.array of shape (N, 3) containing vertex coordinates
        faces: np.array of shape (M, 3) containing vertex indices for triangles
    
    Returns:
        list of lists, where each inner list contains vertex indices forming a hole boundary
    """
    # Create edge to face mapping
    edge_to_face = {}
    for face_idx, face in enumerate(faces):
        # For each edge in the triangle
        for i in range(3):
            # Create edge (always store with smaller index first)
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge not in edge_to_face:
                edge_to_face[edge] = []
            edge_to_face[edge].append(face_idx)
    
    # Find boundary edges (edges with only one adjacent face)
    boundary_edges = [edge for edge, faces in edge_to_face.items() if len(faces) == 1]
    
    if not boundary_edges:
        return []  # No holes found
    
    # Group boundary edges into loops (holes)
    holes = []
    used_edges = set()
    
    while boundary_edges:
        current_hole = []
        start_edge = boundary_edges[0]
        current_vertex = start_edge[0]
        
        while True:
            # Find next edge in the boundary
            next_edge = None
            for edge in boundary_edges:
                if edge in used_edges:
                    continue
                if edge[0] == current_vertex:
                    next_edge = edge
                    current_vertex = edge[1]
                    break
                elif edge[1] == current_vertex:
                    next_edge = edge
                    current_vertex = edge[0]
                    break
            
            if next_edge is None or (current_hole and current_vertex == current_hole[0]):
                break
                
            current_hole.append(current_vertex)
            used_edges.add(next_edge)
            boundary_edges.remove(next_edge)
        
        if len(current_hole) >= 3:  # Only add holes with at least 3 vertices
            holes.append(current_hole)
    
    return holes

# Example usage:


parser = argparse.ArgumentParser(description='Marching cube using normal file')

# Add arguments
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.obj)')
parser.add_argument('surface_file', nargs='?',
                    help='The surface file obj saved, if not provided, no surface_file will be generated.')
parser.add_argument('--box_division', type=int, default=100,
                    help='Box division parameter (default: 100)')
parser.add_argument('--use_points_area', type=str, default='true')

args = parser.parse_args()

normal_file = args.normal_file
surface_file = args.surface_file
box_division = args.box_division
use_points_area = args.use_points_area
use_points_area = args.use_points_area.lower() == 'true'

V, E, N = load_normal_data(normal_file)

# print('np.mean(V)', np.mean(V))

# move it to later so that I can run the script to get all obj
# plot_normal_data(V, E, N)

points, normals = resample_for_points_normal(V, E, N, 0.01)

print(len(points))
print(len(normals))


# sorted_distances = np.sort(pdist(points))
# print('min distance weithin points',sorted_distances )
# print('np.linalg.norm(normals, axis =1)', np.linalg.norm(normals, axis = 1))
# plot_points_wns(points, wns)

# scale larger
bbox_vertices = generate_bounding_box_points(V, scale_factor = 2.0)
# print('scaled bounding box diagnoal', np.linalg.norm(bbox_vertices[0] - bbox_vertices[-1]))

grid_vertices, faces = generate_matched_cube_mesh(box_division, bbox_vertices)

# now I use 4 pi                                                     
points_area = np.ones((points.shape[0], )) / points.shape[0] * np.pi

wns = igl.fast_winding_number_for_points(points, normals, points_area, points)
sorted_wns = np.sort(wns)[::-1]
# print('sorted_wns on original points', sorted_wns)
print('mean_wns on original points:', np.mean(wns))  
print('wns on original points:', np.max(wns))




# scale, mean_wn, _ = find_optimal_scale(points, normals, target_mean=1.25)
# print('scale', scale)
# print('mean_wn after scale', mean_wn)

# scale for points_area
scale = 1.0
area_slider = scale  # default value for 'a'
last_area = scale


slider_value = 0.5
last_value = 0.5


# This is kind of problematic
# area_term = scipy.optimize.minimize_scalar(
#     lambda x: area_for_isovalue(points, normals, base_points_area, grid_vertices, box_division, x),
#     bounds=[0, 3]
# )



if use_points_area is True:
    scale, mean_wn, _ = find_optimal_scale(points, normals, target_mean=1.25)
    wns = igl.fast_winding_number_for_points(points, normals, points_area * scale, grid_vertices)
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)
    print('genus',calculate_genus(SV, SF))
    area_slider = scale 
    last_area  = scale


else:
    areas, genera, isovalues = calculate_area_and_genus(points, normals, grid_vertices, box_division, isovalue_start=0.1, isovalue_end= 15.0, isovalue_step=0.1)
    print(genera)

    # plot_area_genus_relationship(areas, genera, isovalues)


    min_area, corresponding_isovalue = find_minimal_surface_genus_zero(areas, genera, isovalues)

    print(min_area,corresponding_isovalue )
    if corresponding_isovalue is not None:
        wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, corresponding_isovalue)
        slider_value = corresponding_isovalue
        last_value = corresponding_isovalue
        print('genus', calculate_genus(SV, SF))
    else:
        wns = igl.fast_winding_number_for_points(points, normals, points_area , grid_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)
        print('genus',calculate_genus(SV, SF))






if surface_file:
    export_obj(SV, SF, surface_file)
    sys.exit()


# 
# 

plot_normal_data(V, E, N)
plot_points_normal(points, normals)
ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
ps.show()


# Display initial mesh
# ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)



def callback():
    global slider_value, last_value, area_slider, last_area, SV, SF
    
    # Add slider for 'a' (point areas)
    changed_area_slider, area_slider = psim.SliderFloat("Point area (a)", area_slider, v_min=0.0001, v_max=10)
    changed_area_input, area_input = psim.InputFloat("Area input", area_slider, step=0.1)
    
    # Synchronize area slider and input value
    if changed_area_input and area_input != area_slider:
        area_slider = area_input
    elif changed_area_slider and area_slider != area_input:
        area_input = area_slider

    # Recalculate winding numbers if area changed
    if last_area != area_slider:

        current_points_area = points_area * area_slider

        # Scale the base area values by the current area_slider value
        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, points)
        sorted_wns = np.sort(wns)[::-1]
        print('sorted_wns on original points', sorted_wns)
        print('new mean_wns on original points:', np.mean(wns))  
        print('new wns on original points:', np.max(wns))


        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, grid_vertices)
        # sorted_wns = np.sort(wns)[::-1]
        # mean_wns = np.mean(wns)
        print('new mean_wns on grid_vertices:', np.mean(wns))  
        print('new wns on grid_vertices:', np.max(wns))
        

        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, slider_value)

        print('len(sv)', len(SV))
        print('len(sf)', len(SF))
        ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
        last_area = area_slider

    # Original iso-value slider
    changed_slider, slider_value = psim.SliderFloat("Iso value", slider_value, v_min=-10.0, v_max=10.0)
    changed_input, input_value = psim.InputFloat("Input", slider_value, step=0.1)

    # Synchronize slider and input value
    if changed_input and input_value != slider_value:
        slider_value = input_value
    elif changed_slider and slider_value != input_value:
        input_value = slider_value

    if not psim.IsAnyItemActive() and (last_value != slider_value or last_area != area_slider):
        # Use the current area-scaled points
        current_points_area = points_area * area_slider

        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, grid_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, slider_value)
        
        area = igl.doublearea( SV, SF ) / 2.0
        area_sum = sum( area )
        # print('new mean_wns on grid_vertices:', np.mean(wns))  
        # print('new wns on grid_vertices:', np.max(wns))
        
        print('len(sv)', len(SV))
        print('len(sf)', len(SF))
        print('area_sum', area_sum)
        print('genus', calculate_genus(SV, SF))


    
        ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
        last_value = slider_value
    
    if psim.Button("Export button"):
        if surface_file:
            export_obj(SV, SF, surface_file)
        else:
            export_obj(SV, SF, 'output.obj')


ps.init()
ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.show()









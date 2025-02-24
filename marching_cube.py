from utility_io import load_normal_data, export_obj
from utility_viewer_ps import plot_normal_data

import numpy as np 
import polyscope as ps 
import igl
import argparse
from collections import defaultdict
import polyscope.imgui as psim
import gpytoolbox

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import trimesh

from pathlib import Path
from scipy.spatial import ConvexHull
from utility_voronoi_area import compute_geodesic_voronoi_areas
import sys 

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
    V = len(vertices)  # Number of vertices
    F = len(faces)  # Number of faces
    
    # Use a set to count unique edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    E = len(edges)  # Number of edges
    
    # Euler characteristic
    euler_char = V - E + F
    
    # Genus formula for closed orientable surfaces
    genus = (2 - euler_char) / 2  # Use float division to prevent errors

    return genus  # Ensure integer output


def calculate_area_and_genus(points, normals, cube_vertices, box_division, isovalue_start, isovalue_end, isovalue_step = 0.1):
    '''
    '''

    points_area = np.ones((points.shape[0], )) / points.shape[0] * np.pi
    wns = igl.fast_winding_number_for_points(points, normals, points_area , cube_vertices)

    areas = []
    genera = []
    isovalues = []
    n_components = []
    larget_component_genera = []
    largest_component_surface_areas = []


    for isovalue in np.arange(isovalue_start, isovalue_end, isovalue_step):

        SV, SF = gpytoolbox.marching_cubes(wns, cube_vertices, box_division, box_division, box_division, isovalue)
        # print(SV, SF)

        if len(SV) > 0 and len(SF) > 0:
            area = sum ( igl.doublearea(SV, SF) / 2.0 )
            components = get_mesh_components(SV, SF)

            n_component = len(components)
            n_components.append( n_component )

            largest_component = max(components, key=lambda comp: len(comp[1]))
            # print('largest_component', largest_component)
            large_sv, large_sf = largest_component

            max_com = count_mesh_components(large_sv, large_sf )
            # print('len(max_com)', max_com)
            # print('large_sv, large_sf', large_sv, large_sf)
            large_area = sum( igl.doublearea(large_sv, large_sf) / 2.0 )
            # print('large_area',large_area)
            large_genus = calculate_genus( large_sv, large_sf)
            # print('large_genus', large_genus)
            larget_component_genera.append( large_genus ) 
            largest_component_surface_areas.append( large_area )

        else:
            # Handle the empty case
            area = 0  # or whatever default value makes sense
            n_components.append( 0 )
            larget_component_genera.append( 0 ) 
            largest_component_surface_areas.append( 0 )
            print("Warning: Empty mesh detected")
    
        areas.append( area )
        genus = calculate_genus(SV, SF)
        genera.append( genus)
        isovalues.append( isovalue )

    return isovalues, genera, areas, n_components, larget_component_genera, largest_component_surface_areas

def plot_isovalue_genus_area_components(isovalues, genera, areas, n_components=None,
                                        largest_component_genus=None, largest_component_surface_areas=None, 
                                        figname='isovalue_plot.png'):
    '''
    Creates a multi y-axis plot showing the relationship between isovalues vs areas, genera, and optionally
    number of components, largest component genus, and largest component surface areas
    ç
    Parameters:
    -----------
    isovalues : list
        List of isovalues (any range)
    genera : list
        List of corresponding genera counts
    areas : list
        List of corresponding areas
    n_components : list, optional
        List of number of components (default: None)
    largest_component_genus : list, optional
        List of genera values for the largest component (default: None)
    largest_component_surface_areas : list, optional
        List of surface areas for the largest component (default: None)
    figname : str, optional
        Output filename for saving the plot (default: 'isovalue_plot.png')
        
    Returns:
    --------
    None : Displays the plot
    '''
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot genera on primary y-axis
    color1 = 'tab:orange'
    ax1.set_xlabel('Isovalue')
    ax1.set_ylabel('Number of Genera', color=color1)
    line1 = ax1.plot(isovalues, genera, color=color1, label='Total Genera')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis for areas
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Area', color=color2)
    line2 = ax2.plot(isovalues, areas, color=color2, label='Total Area')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Initialize lists for legend
    lines = line1 + line2
    labels = ['Total Genera', 'Total Area']
    
    # Create third y-axis for components only if n_components is provided
    if n_components is not None:
        ax3 = ax1.twinx()
        # Offset the third axis spine
        ax3.spines['right'].set_position(('outward', 60))
        color3 = 'tab:green'
        ax3.set_ylabel('Number of Components', color=color3)
        line3 = ax3.plot(isovalues, n_components, color=color3, label='Components')
        ax3.tick_params(axis='y', labelcolor=color3)
        
        # Add components to legend lists
        lines += line3
        labels.append('Components')
    
    # Plot largest component genus if provided
    if largest_component_genus is not None:
        color4 = 'tab:red'
        line4 = ax1.plot(isovalues, largest_component_genus, color=color4, 
                         linestyle='--', label='Largest Component Genus')
        
        # Add to legend lists
        lines += line4
        labels.append('Largest Component Genus')
    
    # Plot largest component surface area if provided
    if largest_component_surface_areas is not None:
        color5 = 'tab:purple'
        line5 = ax2.plot(isovalues, largest_component_surface_areas, color=color5, 
                         linestyle='--', label='Largest Component Area')
        
        # Add to legend lists
        lines += line5
        labels.append('Largest Component Area')
    
    # Add legend
    ax1.legend(lines, labels, loc='upper right')
    
    # Set x-axis limits
    ax1.set_xlim(0, int(max(isovalues)) + 1)
    ax1.axhline(y=0, color='black', linewidth=1, zorder=0, alpha=0.4)
    
    # Create fine grid with different styles for major and minor lines
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)  # Major grid lines
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)   # Minor grid lines
    
    # Set x-axis minor ticks at 0.1 intervals
    from matplotlib.ticker import MultipleLocator
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Title
    plt.title('Mesh Properties vs Isovalue')
    
    # Save figure if filename is provided
    if figname is not None:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def find_minimal_surface_genus_zero_segment(isovalues, genera, areas, n_components):
    '''
    Find the isovalue and area where:
    - genus = 0
    - number of components = 1
    - area > 0
    and has minimal surface area within that segment.
    
    Parameters:
    -----------
    isovalues : list
        List of isovalues
    genera : list
        List of corresponding genera counts
    areas : list
        List of corresponding areas
    n_components : list
        List of number of components
        
    Returns:
    --------
    tuple : (optimal_isovalue, minimal_area)
        The isovalue and area where conditions are met and area is minimal in that segment
    '''
    # Initialize variables
    found_valid = False
    min_area = float('inf')
    optimal_isovalue = None
    
    # Iterate through all values
    for i in range(len(genera)):
        # If we find invalid conditions after finding valid ones, break
        if found_valid and (genera[i] != 0 or areas[i] <= 0):
            break
            
        # Check all conditions:
        # - genus = 0
        # - exactly 1 component
        # - area > 0
        if genera[i] == 0 and areas[i] > 0:
            found_valid = True
            
            # If this area is smaller than our current minimum
            if areas[i] < min_area:
                min_area = areas[i]
                optimal_isovalue = isovalues[i]
    
    # If we never found valid conditions, return None
    if not found_valid:
        return None, None
        
    return optimal_isovalue, min_area

def count_mesh_components(V, F):
    """
    Count number of connected components in a mesh using iterative graph traversal
    
    Parameters:
    -----------
    V : array-like
        Vertex coordinates, shape (n_vertices, dimension)
    F : array-like
        Face indices, shape (n_faces, vertices_per_face)
        
    Returns:
    --------
    int : Number of connected components
    """

    # Build adjacency list representation
    adj = defaultdict(set)
    for face in F:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            adj[v1].add(v2)
            adj[v2].add(v1)
    
    # Iterative DFS using stack
    def iterative_dfs(start, visited):
        stack = [start]
        while stack:
            vertex = stack.pop()
            if not visited[vertex]:
                visited[vertex] = True
                stack.extend(neighbor for neighbor in adj[vertex] 
                           if not visited[neighbor])
    
    # Initialize visited array
    visited = defaultdict(bool)
    components = 0
    
    # Count components using iterative DFS
    for vertex in range(len(V)):
        if not visited[vertex]:
            iterative_dfs(vertex, visited)
            components += 1
            
    return components

def get_mesh_components(V, F):
    """
    Separate mesh into connected components and return vertices and faces for each
    
    Parameters:
    -----------
    V : array-like
        Vertex coordinates, shape (n_vertices, dimension)
    F : array-like
        Face indices, shape (n_faces, vertices_per_face)
        
    Returns:
    --------
    list of tuples : [(V1,F1), (V2,F2), ...] where each tuple contains:
        - Vi: vertices for component i
        - Fi: faces for component i with reindexed vertex indices
    """
    
    # Build adjacency list representation
    adj = defaultdict(set)
    for face in F:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            adj[v1].add(v2)
            adj[v2].add(v1)
    
    # Iterative DFS using stack to get vertex sets
    def get_component_vertices(start, visited):
        component_verts = set()
        stack = [start]
        while stack:
            vertex = stack.pop()
            if not visited[vertex]:
                visited[vertex] = True
                component_verts.add(vertex)
                stack.extend(neighbor for neighbor in adj[vertex] 
                           if not visited[neighbor])
        return component_verts
    
    # Initialize visited array and components list
    visited = defaultdict(bool)
    components = []
    
    # Find all components
    for vertex in range(len(V)):
        if not visited[vertex]:
            # Get vertices in this component
            component_verts = get_component_vertices(vertex, visited)
            
            # Create vertex index mapping for this component
            old_to_new = {old_idx: new_idx for new_idx, old_idx in 
                         enumerate(sorted(component_verts))}
            
            # Get vertices for this component
            component_V = V[sorted(component_verts)]
            
            # Get faces that belong to this component and reindex them
            component_faces = []
            for face in F:
                # Check if face belongs to this component
                if all(v in component_verts for v in face):
                    # Reindex face vertices
                    new_face = [old_to_new[v] for v in face]
                    component_faces.append(new_face)
            
            # Convert faces to array
            component_F = np.array(component_faces)
            
            # Add to components list if valid
            if len(component_faces) > 0:
                components.append((component_V, component_F))
    
    return components


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

points, normals = resample_for_points_normal(V, E, N, 0.01)

print(len(points))
print(len(normals))

# scale larger
# bbox_vertices = generate_bounding_box_points(V, scale_factor = 1.5)
bbox_vertices = generate_bounding_box_points(V, scale_factor = 2)

grid_vertices, faces = generate_matched_cube_mesh(box_division, bbox_vertices)

slider_value = 0.5
last_value = 0.5



ps.init()


if use_points_area is True:
    
    # start with estimate, sum(points_area) = convex_hull_area
    convex_hull_area = ConvexHull(points).area 

    points_area = np.ones((points.shape[0], )) / points.shape[0] * convex_hull_area
    wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
    
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)
    area = igl.doublearea( SV, SF ) / 2.0
    surface_area = sum( area )

    print('1st round')
    print('genus',calculate_genus(SV, SF))
    print('sum(points_area)', convex_hull_area)
    print('surface_area', surface_area)
    print('surface_area - sum(points_area)', surface_area - convex_hull_area)
    print('-------------------')



    ps_mesh = ps.register_surface_mesh("round 1", SV, SF)
    ps.set_ground_plane_mode("none")
    ps.show()

 
    # update the points_area base on the estimate surface
    points_area = compute_geodesic_voronoi_areas(SV,SF, points)

    # another round
    wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)

    area = igl.doublearea( SV, SF ) / 2.0
    surface_area = sum( area )
    print('2nd round')
    print('genus',calculate_genus(SV, SF))
    print('sum(points_area)',np.sum(points_area))
    print('surface area', surface_area)
    print('surface_area - sum(points_area)', surface_area - np.sum(points_area))
    print('-------------------')

     

    ps_mesh = ps.register_surface_mesh("round 2", SV, SF)
    ps.set_ground_plane_mode("none")
    ps.show()


    ### another round to try:
    points_area = compute_geodesic_voronoi_areas(SV,SF, points)

    # another round
    wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)

    area = igl.doublearea( SV, SF ) / 2.0
    surface_area = sum( area )
    print('3rd round')
    print('genus',calculate_genus(SV, SF))
    print('sum(points_area)',np.sum(points_area))
    print('surface area', surface_area)
    print('surface_area - sum(points_area)', surface_area - np.sum(points_area))
    print('-------------------')



    ps_mesh = ps.register_surface_mesh("round 3", SV, SF)
    ps.set_ground_plane_mode("none")
    ps.show()

    ### another round to try:
    points_area = compute_geodesic_voronoi_areas(SV,SF, points)

    # another round
    wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
    SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)

    area = igl.doublearea( SV, SF ) / 2.0
    surface_area = sum( area )
    print('4th round')
    print('genus',calculate_genus(SV, SF))
    print('sum(points_area)',np.sum(points_area))
    print('surface area', surface_area)
    print('surface_area - sum(points_area)', surface_area - np.sum(points_area))
    print('-------------------')

    ps_mesh = ps.register_surface_mesh("round 4", SV, SF)
    ps.set_ground_plane_mode("none")
    ps.show()


    area_slider = 1.0 
    last_area  = 1.0


else:
    isovalues, genera, areas, n_components, larget_component_genera, largest_component_surface_areas = calculate_area_and_genus(points, normals, grid_vertices, box_division, isovalue_start=0.5, isovalue_end=6.0, isovalue_step=0.1)
    print('isovalues, genera, areas, n_componentsisovalues, genera, areas, n_components, larget_component_genera, largest_component_surface_areas', isovalues, genera, areas, n_components, larget_component_genera, largest_component_surface_areas)
    curve_name = Path(normal_file).stem 

    # plot_isovalue_genus_area_components(isovalues, genera, areas, n_components = None, figname='iso_figs/' + curve_name + '.png')
    plot_isovalue_genus_area_components(isovalues, genera, areas, n_components, larget_component_genera, largest_component_surface_areas, figname='iso_figs/' + curve_name + '.png')


    # optimal_isovalue, min_area = find_minimal_surface_genus_zero_segment(isovalues, genera, areas, n_components)
    optimal_isovalue, min_area = find_minimal_surface_genus_zero_segment(isovalues, larget_component_genera, largest_component_surface_areas, n_components)

    print('min_area,corresponding_isovalue', min_area, optimal_isovalue )
    if optimal_isovalue is not None:

        # wns = igl.fast_winding_number_for_points(points, normals, points_area, points)
        # print('mean_wns on original points:', np.mean(wns))  
        # print('wns on original points:', np.max(wns))
    
        wns = igl.fast_winding_number_for_points(points, normals, points_area, grid_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, optimal_isovalue)
        slider_value = optimal_isovalue
        last_value = optimal_isovalue
        print('genus', calculate_genus(SV, SF))
    else:

        # wns = igl.fast_winding_number_for_points(points, normals, points_area, points)
        # print('mean_wns on original points:', np.mean(wns))  
        # print('wns on original points:', np.max(wns))

        print('should come here')
    
        wns = igl.fast_winding_number_for_points(points, normals, points_area , grid_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, grid_vertices, box_division, box_division, box_division, 0.5)
        slider_value = 0.5
        last_value = 0.5
        print('genus',calculate_genus(SV, SF))


ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
ps.set_ground_plane_mode("none")
ps.show()

print('area_sum', sum(igl.doublearea(SV,SF)/2))
print('genus', calculate_genus(SV, SF))
print('count_mesh_components', count_mesh_components(SV, SF))



if surface_file:
    export_obj(SV, SF, surface_file)
    # sys.exit()


# 
# 

plot_normal_data(V, E, N)
plot_points_normal(points, normals)



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
        print('genus', calculate_genus(SV, SF))

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

        # Remove all existing components
        try:
            # Remove all component meshes
            for i in range(10):  # Using a reasonable upper limit
                try:
                    ps.remove_surface_mesh(f'component {i}')
                except:
                    break
        except:
            pass  # If mesh doesn't exist, continu
        
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
        print('count_mesh_components', count_mesh_components(SV, SF))
        # print('curvature',  igl.principal_curvature(SV, SF) )

        components = get_mesh_components(SV, SF)
        # print('components', components)

        for i in range(len(components)):
            sv_i, sv_f = components[i]
            print(f'{i}th genus ', calculate_genus(sv_i, sv_f))
            mesh_i = trimesh.Trimesh(vertices=sv_i, faces=sv_f)
            print(f"Is watertight: {mesh_i.is_watertight}")





        for i in range(len(components)):
            sv_i, sf_j = components[i]
            mesh_i = ps.register_surface_mesh('component '  + str(i), sv_i, sf_j)
    
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









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


def fibonacci_sphere_with_normals(samples=1000, radius=1.0):
    """
    Generate points and normal vectors on a sphere using the Fibonacci spiral method.
    Points can be scaled to any radius while maintaining normalized normal vectors.
    
    Args:
        samples (int): Number of points to generate
        radius (float): Radius of the sphere (default=1.0)
        
    Returns:
        tuple: (points, normals)
            - points: ndarray of shape (samples, 3) containing (x, y, z) coordinates
            - normals: ndarray of shape (samples, 3) containing normalized normal vectors
    """
    # Create indices array
    indices = np.arange(samples)
    
    # Compute y coordinates (unscaled)
    y = 1 - (2 * indices / (samples - 1))
    
    # Compute radius at each y for unit sphere
    unit_radius = np.sqrt(1 - y * y)
    
    # Compute golden angle
    phi = np.pi * (np.sqrt(5.) - 1.)
    
    # Compute theta angles
    theta = phi * indices
    
    # Compute x and z coordinates for unit sphere
    x = np.cos(theta) * unit_radius
    z = np.sin(theta) * unit_radius
    
    # Stack coordinates into points array (unit sphere)
    unit_points = np.column_stack((x, y, z))
    
    # Normal vectors are the normalized unit points
    normals = unit_points / np.linalg.norm(unit_points, axis=1, keepdims=True)
    
    # Scale points to desired radius
    points = unit_points * radius
    
    return points, normals

def fibonacci_sphere_nonuniform(samples=1000, radius=1.0, first_quadrant_density=5.0):
    """
    Generate points and normal vectors on a sphere using the Fibonacci spiral method,
    with higher density sampling in the first quadrant (x>0, y>0, z>0).
    
    Args:
        samples (int): Base number of points to generate
        radius (float): Radius of the sphere
        first_quadrant_density (float): Relative density multiplier for first quadrant
        
    Returns:
        tuple: (points, normals)
            - points: ndarray containing (x, y, z) coordinates
            - normals: ndarray containing normalized normal vectors
    """
    # Calculate extra samples needed for first quadrant
    base_samples = samples
    extra_samples = int(base_samples * (first_quadrant_density - 1) / 8)  # divide by 8 as first quadrant is 1/8 of sphere
    total_samples = base_samples + extra_samples
    
    # Generate initial uniform distribution
    indices = np.arange(total_samples)
    
    # Compute y coordinates
    y = 1 - (2 * indices / (total_samples - 1))
    
    # Compute radius at each y for unit sphere
    unit_radius = np.sqrt(1 - y * y)
    
    # Compute golden angle
    phi = np.pi * (np.sqrt(5.) - 1.)
    
    # Compute theta angles
    theta = phi * indices
    
    # Compute x and z coordinates for unit sphere
    x = np.cos(theta) * unit_radius
    z = np.sin(theta) * unit_radius
    
    # Stack coordinates into points array
    unit_points = np.column_stack((x, y, z))
    
    # Find points in first quadrant
    first_quadrant_mask = (unit_points > 0).all(axis=1)
    other_quadrants_mask = ~first_quadrant_mask
    
    # Separate points
    first_quadrant_points = unit_points[first_quadrant_mask]
    other_points = unit_points[other_quadrants_mask]
    
    # Keep all points from other quadrants and a dense sampling from first quadrant
    points_to_keep = min(samples, len(other_points) + len(first_quadrant_points))
    final_points = np.vstack([
        other_points,
        first_quadrant_points[:points_to_keep - len(other_points)]
    ])
    
    # Normalize to get normal vectors
    normals = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
    
    # Scale points to desired radius
    points = final_points * radius
    
    return points, normals


def fibonacci_sphere_sparse_quadrant(samples=1000, radius=1.0, first_quadrant_sparsity=0.1):
    """
    Generate points and normal vectors on a sphere using the Fibonacci spiral method,
    with sparse sampling in the first quadrant (x>0, y>0, z>0).
    
    Args:
        samples (int): Base number of points to generate
        radius (float): Radius of the sphere
        first_quadrant_sparsity (float): Fraction of normal density for first quadrant (0 to 1)
        
    Returns:
        tuple: (points, normals)
            - points: ndarray containing (x, y, z) coordinates
            - normals: ndarray containing normalized normal vectors
    """
    # Generate more initial points to ensure good density in non-sparse regions
    total_samples = int(samples * 1.2)  # Generate extra points initially
    
    # Generate initial uniform distribution
    indices = np.arange(total_samples)
    
    # Compute y coordinates
    y = 1 - (2 * indices / (total_samples - 1))
    
    # Compute radius at each y for unit sphere
    unit_radius = np.sqrt(1 - y * y)
    
    # Compute golden angle
    phi = np.pi * (np.sqrt(5.) - 1.)
    
    # Compute theta angles
    theta = phi * indices
    
    # Compute x and z coordinates for unit sphere
    x = np.cos(theta) * unit_radius
    z = np.sin(theta) * unit_radius
    
    # Stack coordinates into points array
    unit_points = np.column_stack((x, y, z))
    
    # Find points in first quadrant
    first_quadrant_mask = (unit_points > 0).all(axis=1)
    other_quadrants_mask = ~first_quadrant_mask
    
    # Separate points
    first_quadrant_points = unit_points[first_quadrant_mask]
    other_points = unit_points[other_quadrants_mask]
    
    # Randomly select subset of first quadrant points based on sparsity
    if len(first_quadrant_points) > 0:
        sparse_count = max(1, int(len(first_quadrant_points) * first_quadrant_sparsity))
        sparse_indices = np.random.choice(
            len(first_quadrant_points), 
            sparse_count, 
            replace=False
        )
        first_quadrant_points = first_quadrant_points[sparse_indices]
    
    # Combine points
    final_points = np.vstack([
        other_points,
        first_quadrant_points
    ])
    
    # Normalize to get normal vectors
    normals = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
    
    # Scale points to desired radius
    points = final_points * radius
    
    return points, normals

def sphere_equator_poles(equator_samples=100, radius=1.0):
    """
    Generate points and normal vectors for a sphere, sampling only the equator
    and adding two pole vertices.
    
    Args:
        equator_samples (int): Number of points to generate along the equator
        radius (float): Radius of the sphere
        
    Returns:
        tuple: (points, normals)
            - points: ndarray containing (x, y, z) coordinates
            - normals: ndarray containing normalized normal vectors
    """
    # Generate equator points
    theta = np.linspace(0, 2*np.pi, equator_samples, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x)
    
    # Create equator points array
    equator_points = np.column_stack((x, y, z))
    
    # Add poles
    north_pole = np.array([[0, 0, 1]])
    south_pole = np.array([[0, 0, -1]])
    
    # Combine all points
    unit_points = np.vstack([equator_points, north_pole, south_pole])
    
    # For unit sphere, normals are same as points
    normals = unit_points.copy()
    
    # Scale points to desired radius
    points = unit_points * radius
    
    return points, normals

def calculate_sphere_areas_latitude(equator_samples=100, polar_angle=np.pi/3):
    """
    Calculate area weights using latitude bands.
    
    Args:
        equator_samples (int): Number of points along the equator
        polar_angle (float): Angle defining the polar regions (in radians)
        
    Returns:
        ndarray: Area weights for each point that sum to 4π
    """
    # Calculate areas using spherical caps
    polar_cap_area = 2 * np.pi * (1 - np.cos(polar_angle))
    equatorial_band_area = 4 * np.pi - 2 * polar_cap_area
    
    # Distribute areas
    equator_point_area = equatorial_band_area / equator_samples
    areas = np.ones(equator_samples) * equator_point_area
    pole_areas = np.array([polar_cap_area, polar_cap_area])
    
    return np.concatenate([areas, pole_areas])

def generate_bounding_box_points(V, scale_factor=1.5):
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

    return euler_char  # Ensure integer output




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_points_wns(tet_vertices, wns, ):
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


def find_optimal_scale(points, normals, target_mean=1.0, tolerance=0.01, max_iterations=50):
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
        base_points_area = np.ones((points.shape[0],)) / points.shape[0] * 4 * np.pi * scale
        
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


import argparse
parser = argparse.ArgumentParser(description='Optimize edges to get normals')
parser.add_argument('surface_file', nargs='?',  help='The curve sketch with optimized normal information.')

args = parser.parse_args()
surface_file = args.surface_file



points, normals = fibonacci_sphere_with_normals(samples=10000,radius=1)
# points, normals = fibonacci_sphere_with_normals(samples=1000,radius=1)
# points, normals = fibonacci_sphere_with_normals(samples=1000,radius=0.5)
# points, normals = fibonacci_sphere_with_normals(samples=1000,radius=0.25)

# points, normals = fibonacci_sphere_nonuniform(samples=3000)
# points, normals = fibonacci_sphere_sparse_quadrant()
points, normals = sphere_equator_poles(equator_samples=100,radius=1)
# points, normals = sphere_equator_poles(equator_samples=100,radius=0.5)

# points, normals = sphere_equator_poles(equator_samples=100,radius=1)
# points, normals = sphere_equator_poles(equator_samples=10, radius=0.5)

box_division = 100


# plot_points_wns(points, wns)

bbox_vertices = generate_bounding_box_points(points, scale_factor=1.5)

# box_division = 30
# box_division = 100
# Then create a matching cube mesh
matched_vertices, faces = generate_matched_cube_mesh(box_division, bbox_vertices)
                                                     




base_points_area = np.ones((points.shape[0], )) / points.shape[0]  * 4 * np.pi 
print('base_points_area', base_points_area)
print('np.sum(base_points_area)', np.sum(base_points_area))


# base_points_area = calculate_sphere_areas_latitude()
# print('base_points_area new ', base_points_area)
# print('np.sum(base_points_area)', np.sum(base_points_area))

wns = igl.fast_winding_number_for_points(points, normals, base_points_area, points)
sorted_wns = np.sort(wns)[::-1]
mean_wns = np.mean(wns)

print('wns on original points', wns)
print('sorted_wns on original points', sorted_wns)
print('mean_wns on original points:', mean_wns)  
print('wns on original points:', np.max(wns))


# # scale, mean_wn, _ = find_optimal_scale(points, normals)

# print('scale', scale)
# print('mean_wn', mean_wn)
# # print('base_points_area', base_points_area)
# print('np.sum(base_points_area)', np.sum(base_points_area) / 4 * np.pi)


# plot_points_wns(matched_vertices, wns)
slider_value = 0.5
last_value = 0.5

area_slider = 1.0  # default value for 'a'
last_area = 1.0


# area_slider = 1.0  # default value for 'a'
# last_area = 1.0

wns = igl.fast_winding_number_for_points(points, normals, base_points_area, matched_vertices)
SV, SF = gpytoolbox.marching_cubes(wns, matched_vertices, box_division, box_division, box_division, slider_value)

sorted_wns = np.sort(wns)[::-1]
mean_wns = np.mean(wns)
print('sorted_wns on matched_vertices', sorted_wns)
print('mean_wns on matched_vertices:', mean_wns)  
print('wns on matched_vertices:', np.max(wns))



plot_points_normal(points, normals)
# Display initial mesh
ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)






def callback():
    global slider_value, last_value, area_slider, last_area, SV, SF
    
    # Add slider for 'a' (point areas)
    changed_area_slider, area_slider = psim.SliderFloat("Point area (a)", area_slider, v_min=0.0001, v_max=100.0)
    changed_area_input, area_input = psim.InputFloat("Area input", area_slider, step=1)
    
    # Synchronize area slider and input value
    if changed_area_input and area_input != area_slider:
        area_slider = area_input
    elif changed_area_slider and area_slider != area_input:
        area_input = area_slider

    # Recalculate winding numbers if area changed
    if last_area != area_slider:
        # Scale the base area values by the current area_slider value
        current_points_area = base_points_area * area_slider

        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, points)
        # print('sorted_wns on original points', sorted_wns)
 
        sorted_wns = np.sort(wns)[::-1]
        mean_wns = np.mean(wns)
        print('sorted_wns on original points', sorted_wns)
        print('mean_wns on original points:', mean_wns)  
        print('wns on original points:', np.max(wns))


        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, matched_vertices)
        # sorted_wns = np.sort(wns)[::-1]
        # mean_wns = np.mean(wns)
        print('new mean_wns on matched_vertices:', np.mean(wns))  
        print('new wns on matched_vertices:', np.max(wns))


        

        SV, SF = gpytoolbox.marching_cubes(wns, matched_vertices, box_division, box_division, box_division, slider_value)

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
        current_points_area = base_points_area * area_slider


        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, points)
        # print('sorted_wns on original points', sorted_wns)
        print('new mean_wns on original points:', np.mean(wns))  
        print('new wns on original points:', np.max(wns))


        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, matched_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, matched_vertices, box_division, box_division, box_division, slider_value)
        
        # print('new mean_wns on matched_vertices:', mean_wns)  
        # print('new wns on matched_vertices:', np.max(wns))


        

        
        print('len(sv)', len(SV))
        print('len(sf)', len(SF))

        area = igl.doublearea( SV, SF ) / 2.0
        area_sum = sum( area )
        print('area_sum', area_sum)
        print('genus', calculate_genus(SV, SF))


    
        ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
        last_value = slider_value
    
    if psim.Button("Export button"):
        if surface_file is not None:
            export_obj(SV, SF, surface_file)
        else:
            export_obj(SV, SF, 'output.obj')





ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.show()









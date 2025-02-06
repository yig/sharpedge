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

def resample_for_points_normal_v1(V, E, N, sample_length=0.01):
    '''
    Sample the edge normal on the points using the sample length.
    Removes duplicate points while maintaining corresponding normals.
    Parameters:
        V: vertices array
        E: edges array (pairs of vertex indices)
        N: normals array
        sample_length: desired length between samples
    Returns:
        points: array of unique sampled points
        normals: array of corresponding normal vectors
    '''
    points = []
    normals = []

    for index, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        n = n_for_segment((p0, p1), sample_length)
        normal = N[index]
        if n == 1:
            point = (p0 + p1) / 2
            # Convert point to tuple for dictionary key
            points.append( point )
            normals.append( normal )
        else:
            # Generate sample points along the edge
            t = np.linspace(0, 1, n)
            # t = np.linspace(0, 1, n)
            for ti in t:
                point = p0 + ti * (p1 - p0)
                points.append ( point )
                normals.append( normal )
    
    points = np.asarray( points )
    normals = np.asarray( normals )

    return points, normals

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


def resample_for_points_normal(V, E, N, sample_length=0.01):
    '''
    Sample the edge normal on the points using the sample length.
    Removes duplicate points while maintaining corresponding normals.
    Parameters:
        V: vertices array
        E: edges array (pairs of vertex indices)
        N: normals array
        sample_length: desired length between samples
    Returns:
        points: array of unique sampled points
        normals: array of corresponding normal vectors
    '''
    points_dict = defaultdict(list)  # Dictionary to store point-normal pairs
    
    for index, edge in enumerate(E):
        e0, e1 = edge
        p0 = V[e0]
        p1 = V[e1]
        n = n_for_segment((p0, p1), sample_length)
        normal = N[index]
        
        # Add normals for the endpoints
        points_dict[tuple(p0)].append(normal)
        points_dict[tuple(p1)].append(normal)
        
        if n == 1:
            point = (p0 + p1) / 2
            point_key = tuple(point)
            points_dict[point_key].append(normal)
        else:
            # Generate sample points along the edge
            t = np.linspace(0, 1, n)[1:-1]  # Exclude endpoints as they're handled above
            for ti in t:
                point = p0 + ti * (p1 - p0)
                point_key = tuple(point)
                points_dict[point_key].append(normal)
    
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


parser = argparse.ArgumentParser(description='Marching cube using normal file')

# Add arguments
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.obj)')
parser.add_argument('surface_file', nargs='?',
                    help='The surface file obj saved, if not provided, no surface_file will be generated.')
parser.add_argument('--box_division', type=int, default=100,
                    help='Box division parameter (default: 100)')

args = parser.parse_args()

normal_file = args.normal_file
surface_file = args.surface_file
box_division = args.box_division


V, E, N = load_normal_data(normal_file)

print('np.mean(V)', np.mean(V))

# move it to later so that I can run the script to get all obj
# plot_normal_data(V, E, N)

points, normals = resample_for_points_normal(V, E, N, 0.01)

print(len(points))
print(len(normals))






# plot_points_wns(points, wns)

bbox_vertices = generate_bounding_box_points(V, scale_factor=1.5)

print('scaled bounding box diagnoal', np.linalg.norm(bbox_vertices[0] - bbox_vertices[-1]))
# box_division = 30
# box_division = 100
# Then create a matching cube mesh
matched_vertices, faces = generate_matched_cube_mesh(box_division, bbox_vertices)
                                                     


# plot_points_wns(matched_vertices, wns)
slider_value = 0.5
last_value = 0.5

area_slider = 1.0  # default value for 'a'
last_area = 1.0

base_points_area = np.ones((points.shape[0], )) / points.shape[0] * np.pi
print('base_points_area', base_points_area)

wns = igl.fast_winding_number_for_points(points, normals, base_points_area, points)
sorted_wns = np.sort(wns)[::-1]
# print('sorted_wns on original points', sorted_wns)
print('mean_wns on original points:', np.mean(wns))  
print('wns on original points:', np.max(wns))


wns = igl.fast_winding_number_for_points(points, normals, base_points_area, matched_vertices)
SV, SF = gpytoolbox.marching_cubes(wns, matched_vertices, box_division, box_division, box_division, slider_value)

sorted_wns = np.sort(wns)[::-1]
mean_wns = np.mean(wns)
# print('sorted_wns on matched_vertices', sorted_wns)
print('mean_wns on matched_vertices:', mean_wns)  
print('wns on matched_vertices:', np.max(wns))


if surface_file:
    export_obj(SV, SF, surface_file)
    sys.exit()



plot_normal_data(V, E, N)
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
        # sorted_wns = np.sort(wns)[::-1]
        # print('sorted_wns on original points', sorted_wns)
        print('new mean_wns on original points:', np.mean(wns))  
        print('new wns on original points:', np.max(wns))


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
        wns = igl.fast_winding_number_for_points(points, normals, current_points_area, matched_vertices)
        SV, SF = gpytoolbox.marching_cubes(wns, matched_vertices, box_division, box_division, box_division, slider_value)
        
        print('new mean_wns on matched_vertices:', mean_wns)  
        print('new wns on matched_vertices:', np.max(wns))
        
        print('len(sv)', len(SV))
        print('len(sf)', len(SF))

    
        ps_mesh = ps.register_surface_mesh("my mesh", SV, SF)
        last_value = slider_value
    
    if psim.Button("Export button"):
        if surface_file:
            export_obj(SV, SF, surface_file)
        else:
            export_obj(SV, SF, 'output.obj')



ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.show()









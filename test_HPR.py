import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import polyscope.imgui as psim


from utility_io import load_normal_data
from utility_viewer_ps import plot_normal_data

import argparse

import open3d as o3d
import polyscope as ps 

from sphere_marching_cube import fibonacci_sphere_with_normals

def HPR(pts, viewpoint, gamma, use_linear_kernel):
    """
    Hidden Point Removal (HPR) Algorithm for determining the direct visibility of point sets in 3D.
    Reference: Katz, S., Tal, A., & Basri, R. (2007). Direct visibility of point sets. ACM Transactions on Graphics (TOG), 26(3), 24-es.
    
    Parameters:
    - pts (numpy.ndarray): Array of shape (N, 3) representing N 3D points.
    - viewpoint (numpy.ndarray): Array of shape (3,) representing the viewpoint in 3D.
    - gamma (float): Kernel parameter to control the transformation.
    - use_linear_kernel (bool): Flag to determine whether to use a linear or power kernel.
    
    Returns:
    - visible_points (numpy.ndarray): The subset of input points that are directly visible.
    - visible_indices (numpy.ndarray): Indices of the visible points in the original input.
    """
    # Convert inputs to numpy arrays if they aren't already
    pts = np.asarray(pts)
    viewpoint = np.asarray(viewpoint)
    
    # Validate input dimensions
    if pts.shape[1] != 3:
        raise ValueError(f"Points must be 3D, got shape {pts.shape}")
    if viewpoint.shape != (3,):
        raise ValueError(f"Viewpoint must be 3D, got shape {viewpoint.shape}")
        
    # Center the points around the viewpoint
    centered_points = pts - viewpoint
    
    # Normalize directions
    norms = np.linalg.norm(centered_points, axis=1, keepdims=True)
    directions = centered_points / (norms + np.finfo(float).eps)  # Add eps to avoid division by zero
    
    # Transform the points using the chosen kernel
    if use_linear_kernel:
        trans_points = (gamma - norms) * directions
    else:
        trans_points = np.power(norms, gamma) * directions
        
    # Compute convex hull of the transformed points
    hull = ConvexHull(trans_points)
    
    # Extract visible points and their indices
    visible_points = pts[hull.vertices]
    visible_indices = hull.vertices
    
    return visible_points, visible_indices


def visualize_hpr_results(points, viewpoint_start, viewpoint_end, visible_indices, name="Point Cloud"):
    """
    Visualize HPR results using Polyscope
    
    Parameters:
    - points: numpy array of shape (N, 3) containing the point cloud
    - viewpoint_start: tuple (x0, y0, z0) starting point of view line
    - viewpoint_end: tuple (x1, y1, z1) ending point of view line
    - visible_indices: numpy array Indices of the visible points in the original input.
    - name: string identifier for the point cloud visualization
    """
    
    # Register the original point cloud
    ps_cloud = ps.register_point_cloud(name, points)
    
    # Get visible points using the provided indices
    visible_points = points[visible_indices]
    
    # Create a color array (red for all points initially)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = 1.0  # Set red channel to 1 for all points
    
    # Set visible points to green
    colors[visible_indices] = [0.0, 1.0, 0.0]
    
    # Add colors to the point cloud
    ps_cloud.add_color_quantity("visibility", colors, enabled=True)
    
    # Convert tuples to numpy arrays for the viewpoint line
    start_point = np.array(viewpoint_start).reshape(1, 3)
    end_point = np.array(viewpoint_end).reshape(1, 3)
    
    # Add the viewpoint markers at both ends
    ps.register_point_cloud(
        "viewpoint_start", 
        start_point,
        color=[0.0, 0.0, 1.0],  # Blue for viewpoint
        radius=0.02  # Make viewpoint larger
    )
    
    ps.register_point_cloud(
        "viewpoint_end", 
        end_point,
        color=[0.0, 0.0, 1.0],  # Blue for viewpoint
        radius=0.02  # Make viewpoint larger
    )
    
    # Draw the line between the two viewpoints
    line_points = np.vstack([start_point, end_point])
    edges = np.array([[0, 1]])  # Connect first and second points
    ps.register_curve_network(
            "view_direction",
            line_points,
            edges,
            color=[0.0, 0.0, 1.0]
    )
    
    # Print information
    print(f"\nViewing {name}")
    print(f"Total points: {len(points)}")
    print(f"Visible points: {len(visible_points)}")




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




parser = argparse.ArgumentParser(description='Marching cube using normal file')
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.obj)')

args = parser.parse_args()
normal_file = args.normal_file


V, E, N = load_normal_data(normal_file)

points, normals = resample_for_points_normal(V, E, N)

# points, normals = fibonacci_sphere_with_normals()


plot_points_normal(points, normals)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.normals = o3d.utility.Vector3dVector(normals)




# Initialize global variables
camera_offset = 0.1
radius = 0.1
last_camera_offset = camera_offset
last_radius = radius

# Set up random seed and index
np.random.seed(42)
i = np.random.randint(0, len(points))

def callback():
    global camera_offset, radius, last_camera_offset, last_radius, i

    # Add slider for camera offset
    changed_camera_slider, camera_offset = psim.SliderFloat("Camera offset", camera_offset, v_min=0.1, v_max=5.0)
    changed_camera_input, camera_input = psim.InputFloat("Camera offset input", camera_offset, step=0.1)
    
    # Synchronize camera offset slider and input value 
    if changed_camera_input and camera_input != camera_offset:
        camera_offset = camera_input
    elif changed_camera_slider and camera_offset != camera_input:
        camera_input = camera_offset

    # Add slider for radius
    changed_radius_slider, radius = psim.SliderFloat("Radius", radius, v_min=0.01, v_max=1.0)
    changed_radius_input, radius_input = psim.InputFloat("Radius input", radius, step=0.01)
    
    # Synchronize radius slider and input value
    if changed_radius_input and radius_input != radius:
        radius = radius_input
    elif changed_radius_slider and radius != radius_input:
        radius_input = radius

    # Update visualization if values changed
    if not psim.IsAnyItemActive() and (last_camera_offset != camera_offset or last_radius != radius):
        v_i = points[i]
        n_i = normals[i]
        camera = v_i + n_i * camera_offset
        
        # Create point cloud from vertices
        vertices, pt_map = pcd.hidden_point_removal(camera, radius)
        
        # Remove previous visualizations if they exist
        try:
            ps.remove_point_cloud("Point Cloud")
            ps.remove_point_cloud("viewpoint_start")
            ps.remove_point_cloud("viewpoint_end")
            ps.remove_curve_network("view_direction")
        except:
            pass

        # Register the original point cloud
        ps_cloud = ps.register_point_cloud("Point Cloud", points)
        
        # Get visible points using the provided indices
        visible_points = points[pt_map]
        
        # Create a color array (red for all points initially)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = 1.0  # Set red channel to 1 for all points
        
        # Set visible points to green
        colors[pt_map] = [0.0, 1.0, 0.0]
        
        # Add colors to the point cloud
        ps_cloud.add_color_quantity("visibility", colors, enabled=True)
        
        # Convert points to numpy arrays for the viewpoint line
        start_point = np.array(v_i).reshape(1, 3)
        end_point = np.array(camera).reshape(1, 3)
        
        # Add the viewpoint markers at both ends
        ps.register_point_cloud(
            "viewpoint_start",
            start_point,
            color=[0.0, 0.0, 1.0],  # Blue for viewpoint
            radius=0.02  # Make viewpoint larger
        )
        
        ps.register_point_cloud(
            "viewpoint_end",
            end_point,
            color=[0.0, 0.0, 1.0],  # Blue for viewpoint
            radius=0.02  # Make viewpoint larger
        )
        
        # Draw the line between the two viewpoints
        line_points = np.vstack([start_point, end_point])
        edges = np.array([[0, 1]])  # Connect first and second points
        ps.register_curve_network(
            "view_direction",
            line_points,
            edges,
            color=[0.0, 0.0, 1.0]
        )
        
        # Print information
        print(f"\nViewing Point Cloud")
        print(f"Total points: {len(points)}")
        print(f"Visible points: {len(visible_points)}")
        print(f"Camera offset: {camera_offset:.2f}, Radius: {radius:.2f}")
        
        last_camera_offset = camera_offset 
        last_radius = radius


ps.init()
ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.show()

import numpy as np
import polyscope as ps
from scipy.spatial import ConvexHull

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


def visualize_hpr_results(points, viewpoint, visible_indices, name="Point Cloud"):
    """
    Visualize HPR results using Polyscope
    
    Parameters:
    - points: numpy array of shape (N, 3) containing the point cloud
    - viewpoint: numpy array of shape (3,) containing the viewpoint coordinates
    - visible_indices: numpy array Indices of the visible points in the original input.
    - name: string identifier for the point cloud visualization
    """
    # Initialize polyscope
    ps.init()
    
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
    
    # Add the viewpoint as a point cloud with a different color
    ps.register_point_cloud(
        "viewpoint", 
        viewpoint.reshape(1, 3),
        color=[0.0, 0.0, 1.0],  # Blue for viewpoint
        radius=0.02  # Make viewpoint larger
    )
    
    # Draw a line from viewpoint to center of visible points
    center = visible_points.mean(axis=0)
    line_points = np.stack([viewpoint, center])
    edges = np.array([[0, 1]])  # Convert edges to numpy array
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
    
    ps.set_ground_plane_mode("none")
    # Show the polyscope GUI
    ps.show()

# Example usage
if __name__ == "__main__":
    # Create some example points (sphere)
    phi = np.random.uniform(0, 2*np.pi, 1000)
    cos_theta = np.random.uniform(-1, 1, 1000)
    theta = np.arccos(cos_theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.column_stack([x, y, z])
    
    # Add some noise
    points += np.random.normal(0, 0.05, points.shape)
    
    # Define viewpoint
    viewpoint = np.array([2.0, 0.0, 0.0])

    visible_points, visible_indices = HPR(points, viewpoint, 2, True)

    
    # Visualize
    visualize_hpr_results(points, viewpoint, visible_indices, "Sphere Example")
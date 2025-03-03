#!/usr/bin/env python3

import numpy as np
import open3d as o3d
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



def create_cube_points(size=1.0, points_per_face=100):
    """Create points on the surface of a cube"""
    points = []
    # Generate points for each face
    for direction in [-1, 1]:
        for axis in range(3):
            # Create grid points on the face
            xy = np.random.uniform(-size/2, size/2, (points_per_face, 2))
            face_points = np.zeros((points_per_face, 3))
            face_points[:, :2] = xy
            face_points[:, 2] = direction * size/2
            # Rotate points to correct face
            if axis == 1:
                face_points = face_points[:, [2, 0, 1]]
            elif axis == 2:
                face_points = face_points[:, [1, 2, 0]]
            points.append(face_points)
    return np.vstack(points)

def create_torus_points(R=1.0, r=0.3, n_points=1000):
    """Create points on a torus surface
    R: major radius
    r: minor radius
    """
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, 2*np.pi, n_points)
    
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    return np.column_stack([x, y, z])

def create_spiral_points(turns=3, points_per_turn=100, radius=1, height=2):
    """Create points along a spiral"""
    n_points = turns * points_per_turn
    t = np.linspace(0, turns*2*np.pi, n_points)
    
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, n_points)
    
    # Add some random variation
    noise = np.random.normal(0, 0.05, (n_points, 3))
    points = np.column_stack([x, y, z]) + noise
    
    return points

def create_two_spheres(r1=1.0, r2=0.5, distance=2.0, n_points=500):
    """Create points on two spheres of different sizes"""
    # First sphere at origin
    phi1 = np.random.uniform(0, 2*np.pi, n_points)
    cos_theta1 = np.random.uniform(-1, 1, n_points)
    theta1 = np.arccos(cos_theta1)
    
    x1 = r1 * np.sin(theta1) * np.cos(phi1)
    y1 = r1 * np.sin(theta1) * np.sin(phi1)
    z1 = r1 * np.cos(theta1)
    sphere1 = np.column_stack([x1, y1, z1])
    
    # Second sphere offset in x-direction
    phi2 = np.random.uniform(0, 2*np.pi, n_points)
    cos_theta2 = np.random.uniform(-1, 1, n_points)
    theta2 = np.arccos(cos_theta2)
    
    x2 = r2 * np.sin(theta2) * np.cos(phi2) + distance
    y2 = r2 * np.sin(theta2) * np.sin(phi2)
    z2 = r2 * np.cos(theta2)
    sphere2 = np.column_stack([x2, y2, z2])
    
    return np.vstack([sphere1, sphere2])

def visualize_hpr_results(points, viewpoint, name="Point Cloud"):
    """Visualize original and visible points using HPR"""
    # Run HPR
    visible_points, visible_indices = HPR(points, viewpoint, gamma=2.0, use_linear_kernel=True)
    
    # Create visualization
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    pcd_original.paint_uniform_color([1, 0, 0])  # Red for original
    
    pcd_visible = o3d.geometry.PointCloud()
    pcd_visible.points = o3d.utility.Vector3dVector(visible_points)
    pcd_visible.paint_uniform_color([0, 1, 0])  # Green for visible
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    print(f"\nViewing {name}")
    print(f"Total points: {len(points)}")
    print(f"Visible points: {len(visible_points)}")
    o3d.visualization.draw_geometries([pcd_original, pcd_visible, coord_frame])
    
def main():
    # Create various example point clouds
    examples = {
        "Cube": create_cube_points(size=1.0, points_per_face=100),
        "Torus": create_torus_points(R=1.0, r=0.3, n_points=1000),
        "Spiral": create_spiral_points(turns=3, points_per_turn=100),
        "Two Spheres": create_two_spheres(r1=1.0, r2=0.5, distance=2.0)
    }
    
    # View each example from different viewpoints
    viewpoints = [
        np.array([2.0, 0.0, 0.0]),  # View from +x axis
        np.array([0.0, 2.0, 2.0]),  # View from diagonal
        np.array([0.0, 0.0, 3.0])   # View from above
    ]
    
    for name, points in examples.items():
        for i, viewpoint in enumerate(viewpoints):
            visualize_hpr_results(points, viewpoint, f"{name} (Viewpoint {i+1})")
            
if __name__ == "__main__":
    main()
    
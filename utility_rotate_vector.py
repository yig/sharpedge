import numpy as np
import polyscope as ps

def rotation_matrix_from(v1, v2):
    """
    Calculate the rotation matrix that rotates v1 to align with v2.
    
    Parameters:
    v1, v2: numpy arrays representing 3D vectors
    
    Returns:
    3x3 rotation matrix as numpy array
    """
    # Normalize the input vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    

    # Compute the dot product and cross product
    dot_product = np.dot(v1_normalized, v2_normalized)
    cross_product = np.cross(v1_normalized, v2_normalized)
    
    # Handle the case where vectors are nearly parallel
    if np.allclose(np.linalg.norm(cross_product), 0, atol=1e-10):
        if dot_product > 0:
            # Vectors are almost identical
            return np.eye(3)
        else:
            # Vectors are almost anti-parallel
            # Find a perpendicular vector to rotate around
            perpendicular = np.array([1, 0, 0]) 
            if np.allclose(np.abs(np.dot(perpendicular, v1_normalized)), 1):
                perpendicular = np.array([0, 1, 0])
            perpendicular = perpendicular - np.dot(perpendicular, v1_normalized) * v1_normalized
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            # Build rotation matrix for 180-degree rotation
            K = np.zeros((3, 3))
            K[0, 1] = -perpendicular[2]
            K[0, 2] = perpendicular[1]
            K[1, 0] = perpendicular[2]
            K[1, 2] = -perpendicular[0]
            K[2, 0] = -perpendicular[1]
            K[2, 1] = perpendicular[0]
            
            return np.eye(3) + 2 * np.matmul(K, K)
    
    # For non-parallel vectors, use the formula directly
    v = cross_product
    s = np.linalg.norm(v)  # sine of the angle
    c = dot_product       # cosine of the angle
    
    # Skew-symmetric cross-product matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Rodrigues' rotation formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    return R


def main():
    # Initialize polyscope
    ps.init()
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate two random vectors
    v1 = np.random.uniform(-2, 2, 3)
    v2 = np.random.uniform(-2, 2, 3)
    
    print("Original v1:", v1)
    print("Target v2:", v2)
    
    # Compute rotation matrix
    R = rotation_matrix_from(v1, v2)
    print("\nRotation matrix:\n", R)
    
    # Apply rotation to v1
    rotated_v1 = R @ v1
    print("\nRotated v1:", rotated_v1)
    
    # Compare normalized vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    rotated_v1_norm = rotated_v1 / np.linalg.norm(rotated_v1)
    
    print("\nNormalized vectors:")
    print("Normalized v1:", v1_norm)
    print("Normalized v2:", v2_norm)
    print("Normalized rotated_v1:", rotated_v1_norm)
    
    aligned = np.allclose(v2_norm, rotated_v1_norm, atol=1e-10)
    print("\nAre rotated_v1 and v2 aligned?", aligned)
    
    # Create point cloud for origin
    origin_point = np.array([[0, 0, 0]])
    origin_cloud = ps.register_point_cloud("origin", origin_point)
    origin_cloud.set_color((0.5, 0.5, 0.5))
    origin_cloud.set_radius(0.05)
    
    # Scale vectors for better visualization
    scale = 1.0
    
    # Add vectors as point clouds with lines
    def add_vector(name, vector, color):
        points = np.vstack([np.zeros(3), vector * scale])
        cloud = ps.register_point_cloud(name, points)
        cloud.set_color(color)
        cloud.set_radius(0.03)
        
        # Add line
        edges = np.array([[0, 1]])
        curve = ps.register_curve_network(f"{name}_line", points, edges)
        curve.set_color(color)
        curve.set_radius(0.01)
        
        return cloud, curve
    
    # Add vectors
    add_vector("v1", v1, (1.0, 0.0, 0.0))  # Red
    add_vector("v2", v2, (0.0, 0.0, 1.0))  # Blue
    add_vector("rotated_v1", rotated_v1, (1.0, 0.0, 1.0))  # Purple
    
    # Add coordinate axes
    axis_length = 3.0
    axes_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    axes_edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3]
    ])
    axes = ps.register_curve_network("axes", axes_points, axes_edges)
    
    # Set colors for axes (X:red, Y:green, Z:blue)
    axes_colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    axes.add_color_quantity("axes_colors", axes_colors, defined_on='edges')
    
    # Add vector info panel
    info_text = f"v1: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]\n"
    info_text += f"v2: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}]\n"
    info_text += f"rotated v1: [{rotated_v1[0]:.2f}, {rotated_v1[1]:.2f}, {rotated_v1[2]:.2f}]\n"
    info_text += f"Vectors aligned: {aligned}"
    
    # Show the viewer
    ps.set_ground_plane_mode("none")
    ps.show()


if __name__ == "__main__":
    main()
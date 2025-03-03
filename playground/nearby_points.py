import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sample 3D points
def generate_random_points_3d(n_points, seed=42):
    """Generate random points in 3D space"""
    np.random.seed(seed)
    return np.random.rand(n_points, 3)

# Method 1: Using BallTree (most efficient for many points)
def count_nearby_points_balltree_3d(points, query_point, distance):
    """
    Count 3D points within distance of query_point using BallTree
    
    Parameters:
    - points: array of shape (n_points, 3)
    - query_point: array of shape (3,)
    - distance: maximum distance to consider a point "nearby"
    
    Returns:
    - count: number of points within distance (excluding query point if it's in the dataset)
    - indices: indices of nearby points
    """
    # Create a BallTree for efficient nearest neighbor search
    tree = BallTree(points)
    
    # Find all points within distance
    indices = tree.query_radius([query_point], r=distance)[0]
    
    # If query point is in the dataset, it will be counted, so we subtract 1
    count = len(indices)
    if np.any(np.all(points[indices] == query_point, axis=1)):
        count -= 1
        
    return count, indices

# Method 2: Using Euclidean distance calculation (simple for small datasets)
def count_nearby_points_euclidean_3d(points, query_point, distance):
    """
    Count 3D points within distance of query_point using Euclidean distance
    
    Parameters:
    - points: array of shape (n_points, 3)
    - query_point: array of shape (3,)
    - distance: maximum distance to consider a point "nearby"
    
    Returns:
    - count: number of points within distance (excluding query point if it's in the dataset)
    - indices: indices of nearby points
    """
    # Calculate Euclidean distances from query_point to all points
    distances = np.sqrt(np.sum((points - query_point)**2, axis=1))
    
    # Find indices where distance is less than or equal to the threshold
    indices = np.where(distances <= distance)[0]
    
    # If query point is in the dataset, it will be counted, so we subtract 1
    count = len(indices)
    if np.any(distances == 0):
        count -= 1
        
    return count, indices

# Visualize results for 3D points
def visualize_nearby_points_3d(points, query_point, nearby_indices, distance):
    """Visualize 3D points and highlight nearby points"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='blue', alpha=0.3, s=30, label='All points')
    
    # Highlight nearby points
    ax.scatter(points[nearby_indices, 0], points[nearby_indices, 1], points[nearby_indices, 2], 
               c='red', alpha=0.7, s=50, label='Nearby points')
    
    # Plot query point
    ax.scatter(query_point[0], query_point[1], query_point[2], 
               c='green', s=200, label='Query point', marker='X')
    
    # Add a simple wireframe sphere to represent the distance
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = query_point[0] + distance * np.outer(np.cos(u), np.sin(v))
    y = query_point[1] + distance * np.outer(np.sin(u), np.sin(v))
    z = query_point[2] + distance * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='green', alpha=0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Points within distance {distance:.2f}: {len(nearby_indices)}')
    ax.legend()
    
    # Set consistent scale
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# Example usage for 3D points
if __name__ == "__main__":
    # Generate 1000 random 3D points
    n_points = 1000
    points_3d = generate_random_points_3d(n_points)
    
    # Choose a query point (could be any point)
    query_point = np.array([0.5, 0.5, 0.5])
    
    # Set a distance threshold
    distance = 0.25
    
    # Method 1: Using BallTree (more efficient)
    count_bt, indices_bt = count_nearby_points_balltree_3d(points_3d, query_point, distance)
    print(f"Using BallTree: {count_bt} points within distance {distance}")
    
    # Method 2: Using Euclidean distance calculation
    count_eucl, indices_eucl = count_nearby_points_euclidean_3d(points_3d, query_point, distance)
    print(f"Using Euclidean: {count_eucl} points within distance {distance}")
    
    # Visualize the results
    visualize_nearby_points_3d(points_3d, query_point, indices_bt, distance)
    
    # Performance comparison for larger datasets
    print("\nPerformance comparison:")
    import time
    
    # Generate larger dataset for timing comparison
    large_points = generate_random_points_3d(10000, seed=123)
    
    # Time BallTree method
    start = time.time()
    count_bt_large, _ = count_nearby_points_balltree_3d(large_points, query_point, distance)
    bt_time = time.time() - start
    print(f"BallTree method: {bt_time:.5f} seconds for 10,000 points")
    
    # Time Euclidean method
    start = time.time()
    count_eucl_large, _ = count_nearby_points_euclidean_3d(large_points, query_point, distance)
    eucl_time = time.time() - start
    print(f"Euclidean method: {eucl_time:.5f} seconds for 10,000 points")
    print(f"BallTree is {eucl_time/bt_time:.1f}x faster")
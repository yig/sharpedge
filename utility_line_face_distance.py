import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def line_triangle_distance(line_start, line_end, triangle, epsilon = 1e-6):
    """
    Calculate minimum distance between a line segment and a triangle.
    
    Parameters:
    line_start: [x, y, z] - Start point of line segment
    line_end: [x, y, z] - End point of line segment
    triangle: list of three [x, y, z] - Triangle vertices
    
    Returns:
    float: minimum distance between line and triangle
    tuple: closest points on line and triangle

    There could be 3 cases of line segment triangle distance 
    1. Line on or very close to triangle plane (d1 ≈ 0 and d2 ≈ 0)
    2. Line segment intersects triangle plane (d1 * d2 < 0)
    3. Line endpoints on same side of triangle plane


    If they're on the same side, could be calculated using the function same_side_line_triangle_distance
    Otherwise break the line to 2 segments and calculate minimum from them.
    """
    line_start = np.asarray(line_start)
    line_end = np.asarray(line_end)
    triangle = np.asarray( triangle )

    # Calculate triangle normal
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distances from line endpoints to triangle plane
    d1 = np.dot(normal, line_start - triangle[0])
    d2 = np.dot(normal, line_end - triangle[0])


    # Case 1: Line is on or very close to the triangle plane
    if abs(d1) < epsilon and abs(d2) < epsilon:
        # Return the distance to the nearest point on the triangle
        min_dist, (point_a, point_b) = same_side_line_triangle_distance(
            line_start, line_end, triangle)
        return min_dist, (point_a, point_b)


    # Case 2: Line segment intersects triangle plane
    if d1 * d2 <= 0:
        # Safe calculation of intersection point
        if abs(d1 - d2) < epsilon:
            # Line is almost parallel to plane, use midpoint
            intersection = (line_start + line_end) / 2
        else:
            t = d1 / (d1 - d2)
            intersection = line_start + t * (line_end - line_start)
        
        # Calculate distances for both segments
        min_distance_0, (point_a_0, point_b_0) = same_side_line_triangle_distance(
            line_start, intersection, triangle)
        min_distance_1, (point_a_1, point_b_1) = same_side_line_triangle_distance(
            intersection, line_end, triangle)
        
        # Return the smaller distance and its corresponding points
        if min_distance_0 < min_distance_1:
            return min_distance_0, (point_a_0, point_b_0)
        else:
            return min_distance_1, (point_a_1, point_b_1)
    
    # Case 3: Line endpoints on same side of triangle plane
    return same_side_line_triangle_distance(line_start, line_end, triangle)


def same_side_line_triangle_distance(line_start, line_end, triangle_points):
    """
    1. Calculate distances from line endpoints to triangle
    2. Calculate distances from triangle vertices to line
    3. Return minimum of all these distances
    """
    def point_line_distance(point, line_start, line_end):
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        point = np.array(point)
        
        line_dir = line_end - line_start
        line_length = np.linalg.norm(line_dir)
        
        if line_length == 0:
            return np.linalg.norm(point - line_start), line_start
        
        line_dir_normalized = line_dir / line_length
        point_vector = point - line_start
        projection_length = np.dot(point_vector, line_dir_normalized)
        
        if projection_length < 0:
            return np.linalg.norm(point - line_start), line_start
        elif projection_length > line_length:
            return np.linalg.norm(point - line_end), line_end
        
        projection_point = line_start + projection_length * line_dir_normalized
        return np.linalg.norm(point - projection_point), projection_point
    
    def point_triangle_distance(point, triangle_points):
        edge1 = triangle_points[1] - triangle_points[0]
        edge2 = triangle_points[2] - triangle_points[0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        plane_dist = abs(np.dot(normal, point - triangle_points[0]))
        projected_point = point - np.dot(point - triangle_points[0], normal) * normal
        
        def is_point_in_triangle(point, triangle):
            v0 = triangle[1] - triangle[0]
            v1 = triangle[2] - triangle[0]
            v2 = point - triangle[0]
            
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            
            return u >= 0 and v >= 0 and u + v <= 1
        
        if is_point_in_triangle(projected_point, triangle_points):
            return plane_dist, projected_point
        
        edge_distances = []
        for i in range(3):
            dist, proj_point = point_line_distance(point, 
                                                 triangle_points[i], 
                                                 triangle_points[(i+1)%3])
            edge_distances.append((dist, proj_point))
        
        min_edge_dist, closest_edge_point = min(edge_distances, key=lambda x: x[0])
        return min_edge_dist, closest_edge_point

    # Calculate distances and closest points
    dist1, point1 = point_triangle_distance(line_start, triangle_points)
    dist2, point2 = point_triangle_distance(line_end, triangle_points)
    
    vertex_distances = []
    for vertex in triangle_points:
        dist, proj_point = point_line_distance(vertex, line_start, line_end)
        vertex_distances.append((dist, vertex, proj_point))
    
    all_distances = [(dist1, line_start, point1), 
                    (dist2, line_end, point2)]
    all_distances.extend([(d, v, p) for d, v, p in vertex_distances])
    
    min_distance, point_a, point_b = min(all_distances, key=lambda x: x[0])
    
    return min_distance, (point_a, point_b)

def visualize_line_triangle_distance(line_start, line_end, triangle_points, closest_points=None):
    """
    Visualize the line segment, triangle, and their closest points in 3D space.
    
    Parameters:
    line_start, line_end: numpy arrays of shape (3,) representing line endpoints
    triangle_points: numpy array of shape (3,3) representing triangle vertices
    closest_points: tuple of two points (closest point on line, closest point on triangle)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot triangle
    triangle_x = triangle_points[:, 0]
    triangle_y = triangle_points[:, 1]
    triangle_z = triangle_points[:, 2]
    triangle_x = np.append(triangle_x, triangle_x[0])
    triangle_y = np.append(triangle_y, triangle_y[0])
    triangle_z = np.append(triangle_z, triangle_z[0])
    
    # Plot triangle surface
    ax.plot_trisurf(triangle_x[:3], triangle_y[:3], triangle_z[:3], 
                    alpha=0.3, color='blue')
    
    # Plot triangle edges
    ax.plot(triangle_x, triangle_y, triangle_z, 
            'b-', linewidth=2, label='Triangle')
    
    # Plot line segment
    ax.plot([line_start[0], line_end[0]], 
            [line_start[1], line_end[1]], 
            [line_start[2], line_end[2]], 
            'r-', linewidth=2, label='Line')
    
    # Plot endpoints
    ax.scatter(*line_start, color='red', s=100)
    ax.scatter(*line_end, color='red', s=100)
    
    if closest_points is not None:
        point_a, point_b = closest_points
        # Plot closest points
        ax.scatter(*point_a, color='green', s=100, label='Closest Point on Line')
        ax.scatter(*point_b, color='green', s=100, label='Closest Point on Triangle')
        
        # Plot distance line
        ax.plot([point_a[0], point_b[0]], 
                [point_a[1], point_b[1]], 
                [point_a[2], point_b[2]], 
                'g--', linewidth=2, label='Minimum Distance')
        
        # Add distance text
        distance = np.linalg.norm(point_a - point_b)
        plt.title(f'Minimum Distance: {distance:.3f}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.show()

def generate_random_triangle(bounds=(-5, 5)):
    """
    Generate a random triangle in 3D space.
    
    Parameters:
    bounds: tuple of (min, max) coordinates for each dimension
    
    Returns:
    numpy array of shape (3,3) representing triangle vertices
    """
    min_bound, max_bound = bounds
    # Generate three random points
    triangle = np.random.uniform(min_bound, max_bound, (3, 3))
    
    # Ensure the triangle isn't degenerate (points aren't collinear)
    while True:
        # Calculate triangle area using cross product
        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        area = np.linalg.norm(np.cross(edge1, edge2)) / 2
        
        # If area is too small, generate new triangle
        if area > 0.1:  # Minimum area threshold
            break
        triangle = np.random.uniform(min_bound, max_bound, (3, 3))
    
    return triangle

def generate_random_line(bounds=(-5, 5), min_length=1):
    """
    Generate a random line segment in 3D space.
    
    Parameters:
    bounds: tuple of (min, max) coordinates for each dimension
    min_length: minimum length of the line segment
    
    Returns:
    tuple of (start_point, end_point) as numpy arrays
    """
    min_bound, max_bound = bounds
    
    while True:
        # Generate two random points
        start = np.random.uniform(min_bound, max_bound, 3)
        end = np.random.uniform(min_bound, max_bound, 3)
        
        # Check if line length meets minimum requirement
        length = np.linalg.norm(end - start)
        if length >= min_length:
            break
    
    return start, end


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random geometry
    triangle = generate_random_triangle(bounds=(-5, 5))
    line_start, line_end = generate_random_line(bounds=(-5, 5), min_length=2)
    
    # Calculate distance and closest points
    distance, closest_points = line_triangle_distance(line_start, line_end, triangle)

    print(f"Random triangle vertices:\n{triangle}")
    print(f"Random line: start={line_start}, end={line_end}")
    print(f"Minimum distance between line and triangle: {distance:.3f}")
    print(f"Closest point on line: {closest_points[0]}")
    print(f"Closest point on triangle: {closest_points[1]}")
    
    
    for i in range(10):
        triangle = generate_random_triangle()
        line_start, line_end = generate_random_line()
        distance, closest_points = line_triangle_distance(line_start, line_end, triangle)
        
        
        visualize_line_triangle_distance(line_start, line_end, triangle, closest_points)

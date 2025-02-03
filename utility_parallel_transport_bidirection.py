import numpy as np 

def parallel_transport(tangent_vectors, U0, tol=1e-10):
    """
    Following section 3: https://legacy.cs.indiana.edu/ftp/techreports/TR425.pdf

    Given a list of points and an initial normal vector U0, 
    this function computes the parallel transported normal vectors.
    
    Parameters:
    - tangent_vectors: a list of 3D tangents, shape (N, 3)
    - V0: initial normal vector, orthogonal to the first tangent vector, shape (3,)
    - tol: tolerance for floating point comparisons (default 1e-10)
    
    Returns:
    - Us: a list of parallel transported normal vectors {Ui}, shape (N, 3)
    """
    
    N = len(tangent_vectors) - 1
    U0 = U0 / np.linalg.norm(U0)
    Us = [U0]  # List to store parallel-transported normal vectors
    
    for i in range(N):
        Ti = tangent_vectors[i]
        Ti_next = tangent_vectors[i + 1]
        
        # Compute B = Ti × Ti+1
        B = np.cross(Ti, Ti_next)
        
        if np.linalg.norm(B) < tol:
            # If B is almost zero, parallel transport directly
            Us.append(Us[i])
        else:
            # Normalize B
            B_hat = B / np.linalg.norm(B)
            
            # Compute the angle θ = arccos(Ti • Ti+1)
            theta = np.arccos(np.clip(np.dot(Ti, Ti_next), -1.0, 1.0))
            
            # Rotate Vi by θ about B_hat to get Vi+1
            Ui_next = rotate_vector(Us[i], B_hat, theta)
            Us.append(Ui_next)
            
    return Us

def rotate_vector(V, axis, theta):
    """
    Rotate vector V by angle theta around axis using Rodrigues' rotation formula.
    
    Parameters:
    - V: vector to rotate, shape (3,)
    - axis: rotation axis, must be a unit vector, shape (3,)
    - theta: rotation angle in radians
    
    Returns:
    - rotated_vector: the rotated vector, shape (3,)
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated_vector = (V * cos_theta + 
                                        np.cross(axis, V) * sin_theta + 
                                        axis * np.dot(axis, V) * (1 - cos_theta))

    return rotated_vector

def compute_tangent_vectors(points):
    """
    Compute the tangent vectors of the input points and ensure the dimension of the tangent vectors 
    matches the number of points by appending the last tangent vector.

    Parameters:
    points : np.array
        An array of points where each row represents a point in space (n x d).
        
    Returns:
    tangent_vectors : np.array
        Tangent vectors with the same number of rows as points.
    """
    # Compute the differences between consecutive points
    tangent_vectors = np.diff(points, axis=0)
    
    # Normalize each tangent vector
    tangent_vectors = tangent_vectors / np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    
    return tangent_vectors

def parallel_transport_bi_direction(points, edge_normal_constraint):
    '''
    Computes normal vectors along a polyline using bidirectional parallel transport 
    from a given constraint point.
    
    Parameters:
    -----------
    points : array-like, shape (N+1, 3)
        Ordered sequence of 3D points defining the polyline vertices.
        N is the number of edges (segments) in the polyline.
        
    edge_normal_constraint : tuple (int, array-like)
        Contains (point_index, normal_vector) where:
        - point_index: Index of the constraint point in the polyline (0 to N)
        - normal_vector: shape (3,) normalized vector perpendicular to edge at constraint point
        
    Returns:
    --------
    normal_vectors : array-like, shape (N, 3)
        Normal vectors for each edge of the polyline, computed by parallel transport
        from the constraint point in both directions.
        
    Notes:
    ------
    The function handles three cases:
    1. Constraint at start point (index 0):
       - Transport normal forward along entire curve
       
    2. Constraint at end point (index N):
       - Transport normal backward along entire curve
       
    3. Constraint at intermediate point:
       - Split curve at constraint point
       - Transport normal backward to start
       - Transport normal forward to end
       - Combine the results
    '''
    # Unpack constraint
    constraint_index, normal = edge_normal_constraint
    
    # Case 1: Constraint at start point
    if constraint_index == 0:
        # Compute edge tangents and transport normal forward
        tangent_vectors = compute_tangent_vectors(points)
        normals = parallel_transport(tangent_vectors, normal)
        return normals
    
    # Case 2: Constraint at end point
    if constraint_index == len(points) - 1:
        # Reverse points, compute transport, then reverse result
        points_reversed = points[::-1]
        tangent_vectors_reversed = compute_tangent_vectors(points_reversed)
        normals_reversed = parallel_transport(tangent_vectors_reversed, normal)
        return normals_reversed[::-1]
    
    # Case 3: Constraint at intermediate point
    else:
        # Split polyline at constraint point
        points_before = points[:constraint_index+1][::-1]  # Reverse for backward transport
        points_after = points[constraint_index:]           # Forward transport
        
        # Compute tangents for both parts
        tangents_before = compute_tangent_vectors(points_before)
        tangents_after = compute_tangent_vectors(points_after)
        
        # Transport normal in both directions
        normals_before = parallel_transport(tangents_before, normal)[::-1]  # Reverse back to original order
        normals_after = parallel_transport(tangents_after, normal)
        
        # Combine normal vectors from both directions
        normal_vectors = normals_before + normals_after
        return normal_vectors

## test and plot function
def generate_3d_curve(num_points, amplitude=1, freq=1):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = np.cos(freq * t)
    y = np.sin(freq * t)
    z = amplitude * np.sin(2 * freq * t)
    return np.column_stack((x, y, z))

def generate_helix(radius, pitch, num_points, num_turns):
    '''
    Generates points along a 3D helix.

    Parameters:
        radius: Radius of the helix
        pitch: Distance between successive turns of the helix along the z-axis
        num_points: Total number of points to generate
        num_turns: Number of full turns of the helix

    Returns:
        points: (num_points, 3) array of points along the helix
    '''
    theta_max = 2 * np.pi * num_turns  # Total angle covered by the helix
    theta = np.linspace(0, theta_max, num_points)  # Angles for each point
    
    x = radius * np.cos(theta)  # X coordinates
    y = radius * np.sin(theta)  # Y coordinates
    z = pitch * theta / (2 * np.pi)  # Z coordinates, linear with theta

    points = np.vstack((x, y, z)).T
    return points

def generate_random_line_points(num_points):
    '''
    Generates evenly spaced points along a line defined by two random points in 3D space.

    Parameters:
        num_points: The number of points to generate along the line, including the start and end points.

    Returns:
        points: Array of generated points along the line, dimension (num_points, 3).
    '''
    # Generate two random points in 3D space
    start = np.random.rand(3)  # Random point 1
    end = np.random.rand(3)    # Random point 2

    # Use linspace to generate the points between the start and end
    points = np.linspace(start, end, num_points)

    return points, start, end  # Return the points and the two random points


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_parallel_transport(points,  normal_vectors, binormal_vectors = None, tangent_vectors = None):
    """
    Plots the points, tangent vectors, and parallel-transported normal vectors in 3D.

    Parameters:
    - points: List of 3D points, shape (N+1, 3)
    - normal_vectors: List of parallel-transported normal vectors, shape (N, 3)
    - binormal_vectors: List of parallel-transported binormal vectors, shape (N, 3)
    - tangent_vectors: (Optional) List of tangent vectors, shape (N, 3)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points (the curve)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Curve', color='b')
    
    # Generate color gradient based on the index of points
    num_points = len(points)
    
    # Scatter the points with gradient colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s =10)
    
    # Calculate midpoints between consecutive points for normal vector plotting
    midpoints = (points[:-1] + points[1:]) / 2
    
 
        
    # Plot the normal vectors at midpoints
    for i in range(0, len(midpoints)): 
        ax.quiver(midpoints[i, 0], midpoints[i, 1], midpoints[i, 2], 
                            normal_vectors[i][0], normal_vectors[i][1], normal_vectors[i][2], 
                            color='r', length=0.2, normalize=True)
        
    # Plot the binormal vectors at midpoints
    if binormal_vectors is not None:
        for i in range(0, len(midpoints)): 
            ax.quiver(midpoints[i, 0], midpoints[i, 1], midpoints[i, 2], 
                                binormal_vectors[i][0], binormal_vectors[i][1], binormal_vectors[i][2], 
                                color='b', length=0.2, normalize=True)

   # Plot the tangent vectors 
    if tangent_vectors is not None:
        for i in range(0, len(tangent_vectors)):  
            ax.quiver(points[i, 0], points[i, 1], points[i, 2], 
                    tangent_vectors[i][0], tangent_vectors[i][1], tangent_vectors[i][2], 
                    color='g', length=0.2, normalize=True)
                    
    ax.set_axis_off()
    plt.show()



# Example usage
if __name__ == "__main__":
    points = generate_3d_curve(50)
    points = generate_helix(radius=1, pitch=0.5, num_points=100, num_turns=3)
#   points, _, _  = generate_random_line_points(50)

    # Initial normal vector, orthogonal to the first tangent
    V0 = np.array([0.0, 0.0, 1.0])
    index = np.random.randint(0, len(points)-1)

    # index = 0
    print('index', index)

    # Call the parallel transport function
    normal_vectors = parallel_transport_bi_direction(points, (index, np.array([0,0, 1])))
    
    
    print('len(points)', len(points))
    print('len(normal_vectors)',len(normal_vectors))
    assert len(points) == len(normal_vectors) + 1
    
    
    # Plot the result
    plot_parallel_transport(points, normal_vectors)
    
import numpy as np 


def compute_parallel_transport_frames( points ):
    '''
    Computes parallel transport frame vectors along a curve using moving frame method.
    
    Parameters:
    -----------
    points : array-like, shape (N, 3)
        Ordered sequence of 3D points defining the curve.
        
    Returns:
    --------
    Us : array-like, shape (N-1, 3)
        First set of normal vectors (U) along the curve.
        These vectors are perpendicular to the curve tangent.
        
    Vs : array-like, shape (N-1, 3)
        Second set of normal vectors (V) along the curve.
        These vectors complete the orthonormal frame with U and tangent.
        
    Notes:
    ------
    - Creates an orthonormal frame (tangent, U, V) at each curve point
    - Uses parallel transport to maintain minimal rotation of the frame
    - First frame is initialized using an arbitrary perpendicular vector
    - Frame vectors are normalized
    '''
    tangent_vectors = np.diff(points, axis=0)
    tangent_vectors = tangent_vectors / np.linalg.norm(tangent_vectors, axis=1, keepdims=True)  # Normalize
    
    V0 = perpendicular_normal(tangent_vectors[0])  # Initial normal vector
    Us, Vs = parallel_transport_frames(tangent_vectors, V0)  # Compute frames
    
    return Us, Vs

def perpendicular_normal(tangent):
    """
    Given a tangent vector, compute a normal vector perpendicular to it.
    
    Parameters:
    - tangent: A 3D tangent vector, shape (3,)
    
    Returns:
    - normal: A normal vector perpendicular to the tangent, shape (3,)
    """
    # Normalize the tangent vector
    tangent = tangent / np.linalg.norm(tangent)
    
    # Choose an arbitrary reference vector for cross product
    # If the tangent is close to the z-axis, use x-axis as the reference, otherwise use z-axis.
    reference = np.array([0, 0, 1]) if np.abs(tangent[2]) < 0.9 else np.array([1, 0, 0])
    
    # Compute the cross product to get a perpendicular vector
    normal = np.cross(tangent, reference)
    
    # Normalize the normal vector to unit length
    normal = normal / np.linalg.norm(normal)
    
    return normal

def parallel_transport_frames(tangent_vectors, V0, tol=1e-10):
    """
    Following section 3: https://legacy.cs.indiana.edu/ftp/techreports/TR425.pdf

    Given a list of tangents and an initial normal vector V0, 
    this function computes the parallel transported normal vectors.
    
    Parameters:
    - tangent_vectors: a list of 3D Tangents, shape (N, 3)
    - V0: initial normal vector, orthogonal to the first tangent vector, shape (3,)
    - tol: tolerance for floating point comparisons (default 1e-10)
    
    Returns:
    - frames: a list of parallel transported normal vectors {Vi}, shape (N, 3)
    """
    
#   print('len(points)', len(points))
#   print('len(tangent_vectors)', len(tangent_vectors))
    
    N = len(tangent_vectors) - 1
    normal_vectors = [V0]  # List to store parallel-transported normal vectors
    
    for i in range(N):
        Ti = tangent_vectors[i]
        Ti_next = tangent_vectors[i + 1]
        
        # Compute B = Ti × Ti+1
        B = np.cross(Ti, Ti_next)
        
        if np.linalg.norm(B) < tol:
            # If B is almost zero, parallel transport directly
            normal_vectors.append(normal_vectors[i])
        else:
            # Normalize B
            B_hat = B / np.linalg.norm(B)
            
            # Compute the angle θ = arccos(Ti • Ti+1)
            theta = np.arccos(np.clip(np.dot(Ti, Ti_next), -1.0, 1.0))
            
            # Rotate Vi by θ about B_hat to get Vi+1
            Vi_next = rotate_vector(normal_vectors[i], B_hat, theta)
            normal_vectors.append(Vi_next)
            
    binormal_vectors = []
    for i in range(len(normal_vectors)):
        binormal = np.cross(tangent_vectors[i], normal_vectors[i])
        binormal_vectors.append( binormal / np.linalg.norm(binormal) )
        
    return normal_vectors, binormal_vectors

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


### test function and view functions

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_curve_frame(points,  normal_vectors, binormal_vectors = None):
    """
    Plots the points, curve frames in 3D.

    Parameters:
    - points: List of 3D points, shape (N+1, 3)
    - normal_vectors: List of parallel-transported normal vectors, shape (N, 3)
    - binormal_vectors: List of parallel-transported binormal vectors, shape (N, 3)
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

                    
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":

    points = generate_helix(radius=1, pitch=0.5, num_points=100, num_turns=3)
    normal_vectors, binormal_vectors = compute_parallel_transport_frames( points )
    
    print('len(points)', len(points))
    print('len(normal_vectors)',len(normal_vectors))
    print('len(tangent_vectors)', len(binormal_vectors))
    
    # Plot the result
    plot_curve_frame(points, normal_vectors, binormal_vectors)
    
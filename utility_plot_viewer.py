import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_sketch_data(V, P):
    """
    Plot a list of 3D polylines.
    
    Args:
        V (ndarray): nx3 array of vertex coordinates where n is number of vertices
        P (list): List of arrays containing vertex indices for each polyline
                 Each array in P contains indices that reference vertices in V
    """
    fig = plt.figure(figsize=(8,8))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    V = np.asarray(V)
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s = 6, color = 'k')


    for polyline in P:
        points = [V[index] for index in polyline]
        pts = np.asarray( points )
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        ax.plot(xs, ys, zs)
        ax.scatter(xs, ys, zs, s = 5)
       
 
    
    plt.axis('off')
    plt.axis('equal')
    plt.show()


def plot_convex_hull_with_normals(points, faces, normals, scale=0.03):
    """
    Plots the convex hull and the normal vectors at each point.
    
    Args:
    - points: np.array of shape (n_points, 3), the 3D coordinates of the convex hull points.
    - faces: np.array of shape (n_faces, 3), the indices of the points that form the triangular faces.
    - normals: np.array of shape (n_points, 3), the normal vectors at each point.
    - scale: float, scale factor for the length of the normal vectors.
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot the triangular faces of the convex hull
    poly3d = [[points[face] for face in triangle] for triangle in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', alpha=.1, edgecolors=(0, 0, 0, 0.1)))
    
    # Plot the points of the convex hull
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha = 0.1, color="blue")
    
    # Plot the normal vectors at each point
    for i in range(len(points)):
        point = points[i]
        normal = normals[i]
        ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], 
                  length=scale, color='g')
    

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_edge_constraints(V, E, P, constraints, scale=0.03, filename = None):
    """
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        P: list of lists, where each inner list contains vertex indices for a polyline with its color
        constraints: dictionary mapping edge indices to normal vectors
        scale: scaling factor for normal vectors (default: 0.03)
        filename: if provided, save plot to this filename (default: None)
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot polylines
    for polyline in P:
        polyline_points = [V[index] for index in polyline]
        polyline_points = np.array(polyline_points)
        ax.plot(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2])

    # Plot normal vectors for constrained edges
    for edge_idx, normal in constraints:
        e = E[edge_idx]
        start = V[e[0]]
        end = V[e[1]]
        
        # Calculate midpoint of edge
        mid = (start + end) / 2
        
        # Plot normal vector
        ax.quiver(mid[0], mid[1], mid[2],
                normal[0], normal[1], normal[2],
                color='green', length=scale, normalize=True,
                arrow_length_ratio=0.2)

    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')

    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, 
                    dpi=300,           # High resolution
                    bbox_inches='tight',# Trim white space
                    pad_inches=0.1)     # Small padding
        plt.close()  # Close the figure to free memory
    else:
        plt.show()



def plot_edge_frames(V, E, Us, Vs, scale=0.03):
    """
    Plot polylines and their frame vectors with consistent styling.
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        Us, Vs: lists of frame vectors for each edge (guaranteed to exist)
        scale: scaling factor for frame vectors (default: 0.03)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot vertices
    ax.scatter(V[:, 0], V[:, 1], V[:, 2],
              c='blue',            
              s=3,                
              alpha=0.6,           
              marker='o',          
              label='Vertices')    

    # Plot edges and frames
    for e, u, v in zip(E, Us, Vs):
        start = V[e[0]]
        end = V[e[1]]
        
        # Plot edge
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='gray', linewidth=2)
        
        # Plot frame vectors at edge midpoint
        mid = (start + end) / 2
        ax.quiver(mid[0], mid[1], mid[2],
                 u[0], u[1], u[2],
                 color='blue', length=scale, normalize=True)
        ax.quiver(mid[0], mid[1], mid[2],
                 v[0], v[1], v[2],
                 color='green', length=scale, normalize=True)
    
    plt.axis('off')
    plt.axis('equal')
    plt.show()


def plot_normal_data(V, E, N, scale=0.03):
    """
    Plot edges and their normal vectors:
    - Black edges with red normal vectors
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        N: (m,3) array of normal vectors for edges
        scale: scaling factor for normal vectors (default: 0.03)
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot vertices
    ax.scatter(V[:, 0], V[:, 1], V[:, 2],
              c='blue',          # Color of points
              s=3,               # Size of points
              alpha=0.6,         # Transparency
              marker='o',        # Point style
              label='Vertices')  # Label for legend
    
    # Plot edges and normals
    for i, e in enumerate(E):
        start = V[e[0]]
        end = V[e[1]]
        
        # Plot edge
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='black', linewidth=2)
        
        # Calculate midpoint and plot normal vector
        mid = (start + end) / 2
        ax.quiver(mid[0], mid[1], mid[2],
                 N[i,0], N[i,1], N[i,2],
                 color='green', length=scale, normalize=True,
                 arrow_length_ratio=0.2)
    
    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')
    plt.show()



def plot_polyline_best_constraints(V, E, P, polyline_normal, scale=0.03, str=None, filename=None):
    """
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        P: list of lists, where each inner list contains vertex indices for a polyline with its color
        polyline_normal: dictionary, key - polyline_index, value : (edge_pos_in_polyline, best_normal_vector)
        scale: scaling factor for normal vectors (default: 0.03)
        str: optional string label for the plot (default: None)
        filename: if provided, save plot to this filename (default: None)
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot polylines
    for polyline_idx, polyline in enumerate(P):
        polyline_points = np.array([V[index] for index in polyline])
        ax.plot(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2])

    # Plot normal vectors for constrained edges
    for polyline_idx, (edge_pos, normal) in polyline_normal.items():
        polyline = P[polyline_idx]
        # Get the edge vertices
        start = V[polyline[edge_pos]]
        end = V[polyline[edge_pos + 1]]
        
        # Calculate midpoint of edge
        mid = (start + end) / 2
        
        # Plot normal vector
        ax.quiver(mid[0], mid[1], mid[2],
                 normal[0], normal[1], normal[2],
                 color='green', length=scale, normalize=True,
                 arrow_length_ratio=0.2)

    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')
    
    # Add title if str is provided
    if str is not None:
        ax.set_title(str)

    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, 
                    dpi=300,           # High resolution
                    bbox_inches='tight',# Trim white space
                    pad_inches=0.1)     # Small padding
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_polyline_normals(V, E, P, polyline_normals, scale=0.03, str=None, filename=None):
    """
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        P: list of lists, where each inner list contains vertex indices for a polyline with its color
        polyline_normals: dictionary, key - polyline_index, value: a list of normals corresponding to each segment of polyline
        scale: scaling factor for normal vectors (default: 0.03)
        str: optional string label for the plot (default: None)
        filename: if provided, save plot to this filename (default: None)
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot polylines
    for polyline_idx, polyline in enumerate(P):
        polyline_points = np.array([V[index] for index in polyline])
        ax.plot(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2])

        # Plot normal vectors for each segment of the polyline
        if polyline_idx in polyline_normals:
            normals = polyline_normals[polyline_idx]
            # For each segment in the polyline
            for i in range(len(polyline) - 1):
                # Get segment endpoints
                start = V[polyline[i]]
                end = V[polyline[i + 1]]
                
                # Calculate midpoint of segment
                mid = (start + end) / 2
                
                # Get corresponding normal vector
                normal = normals[i]
                
                # Plot normal vector
                ax.quiver(mid[0], mid[1], mid[2],
                         normal[0], normal[1], normal[2],
                         color='green', length=scale, normalize=True,
                         arrow_length_ratio=0.2)

    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')
    
    # Add title if str is provided
    if str is not None:
        ax.set_title(str)

    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, 
                    dpi=300,           # High resolution
                    bbox_inches='tight',# Trim white space
                    pad_inches=0.1)     # Small padding
        plt.close()  # Close the figure to free memory
    else:
        plt.show()
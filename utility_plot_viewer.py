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

def plot_edge_info(V, E):
    '''
    Given:
        - V: nx3 array of vertex coordinates 
        - E: mx2 array of edge vertex indices (no duplicates)
    '''

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    V = np.asarray(V)
    for edge_index, edge in enumerate(E):
        e0, e1 = edge
        pts = np.asarray([V[e0], V[e1]])
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        
        # Calculate midpoint for edge label
        midx = (xs[0] + xs[1]) / 2
        midy = (ys[0] + ys[1]) / 2
        midz = (zs[0] + zs[1]) / 2
        
        ax.plot(xs, ys, zs)
        ax.scatter(xs, ys, zs, s=5)
        ax.text(midx, midy, midz, str(edge_index))
    
    plt.axis('off')
    plt.axis('equal')
    plt.show()


def plot_convex_hull_with_normals(points, faces, normals, scale=0.08):
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


def plot_edge_frames(V, E, P, Us, Vs, scale=0.08):
    """
    Plot polylines with different colors and their frame vectors.
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        P: a list of arrays containing vertex indices for each polyline
        Us, Vs: lists of frame vectors for each edge (guaranteed to exist)
        scale: scaling factor for frame vectors (default: 0.03)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot vertices
    ax.scatter(V[:, 0], V[:, 1], V[:, 2],
              c='blue', s=3, alpha=0.6, marker='o', label='Vertices')
    
    # Generate colors for each polyline
    num_polylines = len(P)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_polylines))
    
    # Map from edge index to polyline color
    edge_to_color = {}
    
    # First identify which edge belongs to which polyline
    for poly_idx, polyline in enumerate(P):
        # Extract edges from polyline
        for i in range(len(polyline) - 1):
            v1, v2 = polyline[i], polyline[i+1]
            # Find this edge in the edge list
            for e_idx, (e1, e2) in enumerate(E):
                if (e1 == v1 and e2 == v2) or (e1 == v2 and e2 == v1):
                    edge_to_color[e_idx] = colors[poly_idx]
    
    # Plot edges and frames with polyline colors
    for e_idx, (e, u, v) in enumerate(zip(E, Us, Vs)):
        start = V[e[0]]
        end = V[e[1]]
        
        # Get color for this edge
        color = edge_to_color.get(e_idx, 'gray')  # Default to gray if not in a polyline
        
        # Plot edge
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color, linewidth=2)
        
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
    ax.set_title('U V Frame')
    plt.show()


def plot_normal_data(V, E, N, scale=0.08):
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
        # ax.text(mid[0], mid[1], mid[2], i)

    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')
    plt.show()


def plot_polyline_best_constraints(V, E, P, polyline_normal, scale=0.08, str=None, filename=None):
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


def plot_polyline_normals(V, E, P, polyline_normals, scale=0.08, str=None, filename=None):
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


def plot_edge_constraints(V, E, P, constraints, unconstrained_polylines_indices=None, scale=0.08, str=None, filename=None, block=True):
    """
    Plot 3D visualization of polylines with edge normal constraints.
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        P: list of lists, where each inner list contains vertex indices for a polyline
        constraints: Either:
                     1. [(index, normal)] list of tuples of index and normal constraint, or
                     2. {index: normal} dictionary mapping edge indices to normal vectors
        unconstrained_polylines_indices : a set/list of polyline indices
        scale: scaling factor for normal vectors (default: 0.03)
        str: optional title string for the plot (default: None)
        filename: if provided, save plot to this filename (default: None)
    Plot 3D visualization of polylines with edge normal constraints.
    Added 'block' parameter to control whether plot blocks execution.
    """
    # Store the current interactive state
    was_interactive = plt.isinteractive()
    
    # Set interactive mode according to blocking preference
    if block:
        plt.ioff()  # Turn off interactive mode for blocking display
    else:
        plt.ion()   # Turn on interactive mode for non-blocking display
    
    # Create figure or reuse existing one
    if hasattr(plot_edge_constraints, 'fig') and plt.fignum_exists(plot_edge_constraints.fig.number):
        # Clear existing figure
        plt.figure(plot_edge_constraints.fig.number)
        plt.clf()
        fig = plot_edge_constraints.fig
        ax = fig.add_subplot(111, projection='3d')
    else:
        # Create new figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_edge_constraints.fig = fig  # Store figure for reuse
    
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Rest of your plotting code remains the same...
    # [Your existing plotting code here]
    
    # Plot polylines
    if P is not None:
        for index, polyline in enumerate(P):
            polyline_points = np.array([V[idx] for idx in polyline])
            # Default style for constrained polylines
            style = {}
            scatter_style = {'s': 5}
            # Check if we need to use the unconstrained style
            if unconstrained_polylines_indices is not None and index in unconstrained_polylines_indices:
                style = {'linestyle': '--', 'color': 'r'}
                scatter_style['color'] = 'r'
            # Plot the polyline and points
            ax.plot(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2], **style)
            ax.scatter(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2], **scatter_style)
    elif P is None:
        V = np.asarray(V)
        for edge_index, edge in enumerate(E):
            e0, e1 = edge
            pts = np.asarray([V[e0], V[e1]])
            xs = pts[:, 0]
            ys = pts[:, 1]
            zs = pts[:, 2]
            ax.plot(xs, ys, zs)
            ax.scatter(xs, ys, zs, s=5)

            
        
    # Convert constraints to list of (edge_idx, normal) pairs if it's a dictionary
    if isinstance(constraints, dict):
        constraint_pairs = list(constraints.items())
    else:
        constraint_pairs = constraints
    
    # Plot normal vectors for constrained edges
    for edge_idx, normal in constraint_pairs:
        e = E[edge_idx]
        e0, e1 = e
        start = V[e0]
        end = V[e1]
        # Calculate midpoint of edge
        mid = (start + end) / 2

        color = 'green'

        if unconstrained_polylines_indices is not None:
            for index in unconstrained_polylines_indices:
                polyline_vertex_indices = P[index]

                if e0 in polyline_vertex_indices and e1 in polyline_vertex_indices:
                    color = 'red'
                    break

        # Plot normal vector
        ax.quiver(mid[0], mid[1], mid[2],
                 normal[0], normal[1], normal[2],
                 color=color , length=scale, normalize=True,
                 arrow_length_ratio=0.2)
        # ax.text(mid[0], mid[1], mid[2], edge_idx)
    
    # Make axes equal and set labels
    plt.axis('off')
    plt.axis('equal')
    
    # Add title if str is provided
    if str is not None:
        ax.set_title(str)
    
    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        # Draw the plot
        fig.canvas.draw()
        
        if block:
            # Use the correct blocking behavior
            plt.show()  # Default is blocking when interactive mode is off
        else:
            # For non-blocking, explicitly set block=False and add a pause
            plt.show(block=False)
            plt.pause(0.001)  # Small pause to ensure the plot displays
    
    # Restore previous interactive state
    if was_interactive:
        plt.ion()
    else:
        plt.ioff()
    
    return fig, ax


def plot_edge_constraints_two_normals(V, E, P, constraints, unconstrained_polylines_indices=None, scale=0.08, str=None, filename=None, block=False):
    """
    Plot 3D visualization of polylines with edge normal constraints.
    Handles two normals per edge using tuple keys (edge_idx, which_normal).
    """
    # Setup figure (same as before)
    plt.ion()
    
    if hasattr(plot_edge_constraints_two_normals, 'fig') and plt.fignum_exists(plot_edge_constraints_two_normals.fig.number):
        plt.figure(plot_edge_constraints_two_normals.fig.number)
        plt.clf()
        fig = plot_edge_constraints_two_normals.fig
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_edge_constraints_two_normals.fig = fig
    
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot polylines (same as before)
    for index, polyline in enumerate(P):
        polyline_points = np.array([V[idx] for idx in polyline])
        style = {}
        scatter_style = {'s': 5}
        if unconstrained_polylines_indices is not None and index in unconstrained_polylines_indices:
            style = {'linestyle': '--', 'color': 'r'}
            scatter_style['color'] = 'r'
        ax.plot(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2], **style)
        ax.scatter(polyline_points[:,0], polyline_points[:,1], polyline_points[:,2], **scatter_style)
    
    # Convert constraints to list format if it's a dictionary
    if isinstance(constraints, dict):
        constraint_pairs = list(constraints.items())
    else:
        constraint_pairs = constraints
    
    # Plot normal vectors
    for constraint_item in constraint_pairs:
        # Parse the edge info - could be different formats
        if isinstance(constraint_item[0], tuple):
            # Format: ((edge_idx, which_edge), normal)
            edge_key, normal = constraint_item
            edge_idx, which_edge = edge_key
        else:
            # Handle other possible formats if needed
            edge_idx, normal = constraint_item
            which_edge = 0
        
        # Get the edge vertices
        if edge_idx < len(E):  # Make sure edge_idx is valid
            e = E[edge_idx]
            e0, e1 = e
            start = V[e0]
            end = V[e1]
            
            # Calculate midpoint with offset
            mid = (start + end) / 2
            
            # Add a small offset to separate the two normal vectors visually
            offset = 0.01 * (which_edge - 0.5)
            edge_dir = end - start
            edge_dir_norm = edge_dir / np.linalg.norm(edge_dir)
            # Offset perpendicular to both edge and normal
            offset_dir = np.cross(edge_dir_norm, normal)
            if np.linalg.norm(offset_dir) > 1e-6:  # Check if not zero
                offset_dir = offset_dir / np.linalg.norm(offset_dir)
                mid = mid + offset * offset_dir
            
            # Choose color based on which edge (0 or 1)
            color = 'green' if which_edge == 0 else 'red'
            
            # Plot the normal vector
            ax.quiver(mid[0], mid[1], mid[2],
                     normal[0], normal[1], normal[2],
                     color=color, length=scale, normalize=True,
                     arrow_length_ratio=0.2)
    
    # Finalize plot (same as before)
    plt.axis('off')
    plt.axis('equal')
    
    if str is not None:
        ax.set_title(str)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        fig.canvas.draw()
        # Fix: We should respect the block parameter here
        if block:
            plt.ioff()  # Turn off interactive mode when blocking
            plt.show(block=True)  # Use plt.show with block=True
        else:
            plt.show(block=False)
            plt.pause(0.001)  # Only pause if not blocking
    
    return fig, ax

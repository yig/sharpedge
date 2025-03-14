import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from plot2gltf import GLTFGeometryExporter

from utility_line_face_distance import line_triangle_distance

from pathlib import Path

from utility_io import load_sketch_polyline_data

def compute_3d_convex_hull(points):
    '''
    Compute the convex hull for 3D points.
    
    Args:
        points: numpy.ndarray (Nx3)
            Input points in 3D space
    Returns:
        tuple containing:
        - vertices: numpy.ndarray
            Points that form the convex hull (could be same as input points)
        - edges: numpy.ndarray
            Edges of convex hull as pairs of vertex indices
        - triangles: numpy.ndarray
            Triangular faces of convex hull as triplets of vertex indices, 
            in counterclockwise order relative to outward normal
    '''
    # Compute convex hull
    hull = ConvexHull(points)
    
    # Get vertices (same as input points since ConvexHull keeps original points)
    vertices = points
    
    # Get triangular faces and ensure they're counterclockwise
    triangles = orient_triangle_normals_outward(points, hull.simplices)
    
    # Extract edges from triangular faces (without duplicates)
    edges = get_unique_edges_from_triangles(triangles)
    
    return vertices, edges, triangles

def orient_triangle_normals_outward(vertices, triangles):
    """
    Orient each triangle's vertices in counterclockwise order relative to its outward normal.
    
    Args:
        vertices: numpy.ndarray
            Nx3 array of 3D vertex coordinates
        triangles: numpy.ndarray
            Array of triangle vertex indices
    Returns:
        numpy.ndarray: Array of triangles with consistent counterclockwise ordering
    """    
    oriented_triangles = []
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        normal = np.cross(v1 - v0, v2 - v0)
        # Check if normal points outward (using hull center as reference)
        center = np.mean(vertices, axis=0)
        if np.dot(normal, v0 - center) < 0:  # Normal points inward
            triangle = triangle[::-1]  # Reverse order for outward normal
        oriented_triangles.append(triangle)
    return np.array(oriented_triangles)

def get_unique_edges_from_triangles(triangles):
    """
    Extract unique edges from triangular faces.
    
    Args:
        triangles: numpy.ndarray
            Array of triangular faces as vertex index triplets
    Returns:
        numpy.ndarray: Array of unique edges as vertex index pairs
    """
    # Create edges from triangles
    edges = []
    for triangle in triangles:
        # For each triangle, create three edges
        for i in range(3):
            edge = sorted([triangle[i], triangle[(i + 1) % 3]])  # Sort for consistent ordering
            edges.append(edge)
    
    # Convert to numpy array and remove duplicates
    edges = np.array(edges)
    edges = np.unique(edges, axis=0)
    
    return edges

def compute_hull_normals(vertices, edges, faces):
    """
    Compute normals for vertices, edges, and faces of a 3D hull.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Array of shape (N, 3) containing vertex coordinates
    edges : np.ndarray
        Array of shape (M, 2) containing vertex indices that form edges
    faces : np.ndarray
        Array of shape (K, 3) containing vertex indices that form triangular faces
    
    Returns:
    --------
    vertex_normals : np.ndarray
        Array of shape (N, 3) containing normalized vertex normals
    edge_normals : np.ndarray
        Array of shape (M, 3) containing normalized edge normals
    face_normals : np.ndarray
        Array of shape (K, 3) containing normalized face normals
    """
    
    # Initialize output arrays
    num_vertices = len(vertices)
    num_edges = len(edges)
    num_faces = len(faces)
    
    vertex_normals = np.zeros((num_vertices, 3))
    edge_normals = np.zeros((num_edges, 3))
    face_normals = np.zeros((num_faces, 3))
    
    # 1. Compute face normals first (we'll need these for vertices and edges)
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Compute vectors along two edges of the face
        vec1 = v1 - v0
        vec2 = v2 - v0
        # Compute normal using cross product
        normal = np.cross(vec1, vec2)
        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm > 0:
            face_normals[i] = normal / norm
    
    # 2. Compute vertex normals (average of adjacent face normals)
    vertex_face_map = [[] for _ in range(num_vertices)]
    for i, face in enumerate(faces):
        for vertex_idx in face:
            vertex_face_map[vertex_idx].append(i)
    
    for i in range(num_vertices):
        if vertex_face_map[i]:
            # Average the normals of all adjacent faces
            avg_normal = np.mean(face_normals[vertex_face_map[i]], axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                vertex_normals[i] = avg_normal / norm
    
    # 3. Compute edge normals (average of adjacent face normals)
    edge_face_map = [[] for _ in range(num_edges)]
    for i, edge in enumerate(edges):
        for j, face in enumerate(faces):
            # Check if edge vertices are in the face
            if (edge[0] in face and edge[1] in face):
                edge_face_map[i].append(j)
    
    for i in range(num_edges):
        if edge_face_map[i]:
            # Average the normals of all adjacent faces
            avg_normal = np.mean(face_normals[edge_face_map[i]], axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                edge_normals[i] = avg_normal / norm
    
    return vertex_normals, edge_normals, face_normals

def calculate_sketch_edge_normals(V, E, epsilon):
    """
    Calculate edge normals for sketch edges based on convex hull geometry.
    The function uses a hierarchical approach to determine edge normals:
    1. If the edge is part of the convex hull, use the hull edge normal
    2. If both vertices have normals from the hull, use averaged vertex normals
    3. If the edge is close to hull faces, use averaged face normals
    
    Parameters
    ----------
    V : np.ndarray, shape (N, 3)
        Array of 3D vertex coordinates where N is the number of vertices.
        Each row contains [x, y, z] coordinates.
    
    E : np.ndarray or list, shape (M, 2)
        List of M edges, where each edge is defined by two vertex indices [i, j]
        referencing vertices in V. Edge direction matters for normal orientation.
    
    epsilon : float
        Distance threshold for determining if an edge is close to a hull face.
        Smaller values require edges to be closer to faces for normal assignment.
    
    Returns
    -------
    edge_normals : np.ndarray, shape (M, 3)
        Array of normalized edge normals. Each row contains a [nx, ny, nz] normal vector.
        Zero vectors [0, 0, 0] indicate edges where no normal could be determined.
    """
    # Step 1: Compute convex hull and get vertices, edges, and faces
    # Using the provided compute_3d_convex_hull function
    hull_vertices, hull_edges, hull_faces = compute_3d_convex_hull(V)
    
    # Step 2: Compute hull normals using the provided compute_hull_normals function
    vertex_normals, hull_edge_normals, face_normals = compute_hull_normals(
        hull_vertices, hull_edges, hull_faces
    )
    # Step 3: Initialize output array for sketch edge normals
    edge_normals = np.zeros((len(E), 3))

    # Initialize lists to store edge indices for each method
    hull_edges_list = []
    vertex_normals_list = []
    nearby_faces_list = []
    
    # Process each sketch edge to determine its normal
    for i, (v1_idx, v2_idx) in enumerate(E):
        p1 = V[v1_idx]
        p2 = V[v2_idx]
        
        # Case 1: Check if the sketch edge is a convex hull edge
        hull_edge_found = False
        for j, hull_edge in enumerate(hull_edges):
            if {v1_idx, v2_idx} == set(hull_edge):
                edge_normals[i] = hull_edge_normals[j]
                hull_edges_list.append(i)  # Store edge index
                hull_edge_found = True
                break
                
        if hull_edge_found:
            continue
        
        # Case 2: If vertices have normals from the hull, average them
        v1_normal = vertex_normals[v1_idx]
        v2_normal = vertex_normals[v2_idx]
        if np.any(v1_normal != 0) or np.any(v2_normal != 0):
            avg_normal = (v1_normal + v2_normal) / 2
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                edge_normals[i] = avg_normal / norm
                vertex_normals_list.append(i)  # Store edge index
                continue
            
        # Case 3: If edge is close to hull faces, average their normals
        nearby_face_normals = []
        for j, face in enumerate(hull_faces):
            triangle_points = hull_vertices[face]
            dist, _ = line_triangle_distance(p1, p2, triangle_points)
            if dist < epsilon:
                nearby_face_normals.append(face_normals[j])
                
        if nearby_face_normals:
            avg_normal = np.mean(nearby_face_normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                edge_normals[i] = avg_normal / norm
                nearby_faces_list.append(i)  # Store edge index
    
    # Print summary
    print(f"# Normals from hull edges: {len(hull_edges_list)}")
    print(f"# Normals from vertex averaging: {len(vertex_normals_list)}")
    print(f"# Normals from nearby faces: {len(nearby_faces_list)}")
    print(f"Total edges processed: {len(E)}")
    print(f"Edges without normals: {len(E) - (len(hull_edges_list) + len(vertex_normals_list) + len(nearby_faces_list))}")
    print()
    # print("Normals from hull edges: ", hull_edges_list)
    # print("Normals from vertex averaging", vertex_normals_list)
    # print("Normals from nearby faces", nearby_faces_list)
          
    return edge_normals

def line_nearby_hull_faces(line, hull_vertices, hull_faces, epsilon):
    '''
    Finds all faces of a convex hull that are within a specified distance from a line segment.
    
    Args:
        line: Tuple of two points (p1, p2) defining the line segment
        hull_vertices: Array of vertices that make up the convex hull
        hull_faces: Array of triangular faces defined by indices into hull_vertices 
        epsilon: Maximum distance threshold - faces closer than this to the line are considered "nearby"
    
    Returns:
        nearby_faces: List of indices of faces that are within epsilon distance from the line
    '''
    nearby_faces = []
    p1, p2 = line
    for j, face in enumerate(hull_faces):
        # Get triangle vertices for distance calculation
        triangle_points = hull_vertices[face]
        dist, _ = line_triangle_distance(p1, p2, triangle_points)
        if dist < epsilon:
            nearby_faces.append(j)
    return nearby_faces

def filter_nonparallel_edge_normals(V, E, edge_normals, tol=1e-2):
    '''
    Given:
    V : np.ndarray, shape (N, 3)
        Array of 3D vertex coordinates where N is the number of vertices.
        Each row contains [x, y, z] coordinates.
    E : np.ndarray or list, shape (M, 2)
        List of M edges, where each edge is defined by two vertex indices [i, j]
        referencing vertices in V. Edge direction matters for normal orientation.
    edge_normals : np.ndarray, shape (M, 3)
        Array of normalized edge normals. Each row contains a [nx, ny, nz] normal vector.
        Zero vectors [0, 0, 0] indicate edges where no normal could be determined.
    tol: A tolerance value for determining if the normal is parallel to the tangent vector.
    
    Return:
        filtered_edge_constraints: list of tuple containing (edge_idx, normal)
    '''
    # Convert inputs to numpy arrays if they aren't already
    V = np.asarray(V)
    E = np.asarray(E)
    edge_normals = np.asarray(edge_normals)
    
    # Count edges that have non-zero normals
    valid_normals_mask = np.any(np.abs(edge_normals) > 1e-15, axis=1)
    edges_with_normals = np.sum(valid_normals_mask)
    
    filtered_edge_constraints = []
    total_edges = len(E)
    
    for edge_idx, (v1_idx, v2_idx) in enumerate(E):
        # Get edge normal
        normal = edge_normals[edge_idx]
        
        # Check if normal is zero vector (no normal determined)
        if np.allclose(normal, 0, atol=1e-15):
            continue
            
        # Calculate edge direction (tangent vector)
        edge_vector = V[v2_idx] - V[v1_idx]
        edge_length = np.linalg.norm(edge_vector)
        
        # Normalize edge vector
        edge_vector = edge_vector / edge_length
        
        # Check if normal is nearly parallel to edge direction
        alignment = abs(np.dot(normal, edge_vector))
        
        # If normal is not nearly parallel to edge (dot product not close to 1)
        if alignment < (1.0 - tol):
            filtered_edge_constraints.append((edge_idx, normal))
    
    print(f"Starting with {edges_with_normals} edges with valid normals out of {total_edges} total edges")
    print(f"After filtering: {len(filtered_edge_constraints)} edges remained")
    print()
    return filtered_edge_constraints

def get_sketch_edge_constraints(V, E, tol=2e-2, epsilon=5e-3):
    '''
    Computes and filters edge constraints for a sketch by calculating edge normals
    and filtering out those that are parallel to their edges.
    
    Parameters
    ----------
    V : np.ndarray, shape (N, 3)
        Array of 3D vertex coordinates where N is the number of vertices.
        Each row contains [x, y, z] coordinates.
    E : np.ndarray or list, shape (M, 2)
        List of M edges, where each edge is defined by two vertex indices [i, j]
        referencing vertices in V.
    tol : float, optional (default=1e-2)
        Tolerance value for filtering parallel normals.
        Higher values will filter out more edges that are nearly parallel.
    epsilon : float, optional (default=5e-3)
        Tolerance value for calculating edge normals.
        Controls the sensitivity of normal calculations.
    
    Returns
    -------
    list of tuple
        Each tuple contains (edge_idx, normal) where:
        - edge_idx (int): Index of the edge in E
        - normal (np.ndarray): (3,) normalized direction vector
        Only includes edges that have:
        1. Valid non-zero normals
        2. Normals not parallel to their edge direction
    
    Notes
    -----
    The function first calculates edge normals using calculate_sketch_edge_normals()
    and then filters out edges whose normals are nearly parallel using 
    filter_nonparallel_edge_normals().
    '''
    edge_normals = calculate_sketch_edge_normals(V, E, epsilon=epsilon)
    edge_constraints = filter_nonparallel_edge_normals(V, E, edge_normals, tol=tol)
    return edge_constraints


### helper -- plot and export function
def plot_3d_geometry(vertices, edges, triangles, polylines, 
                       vertex_color='red', 
                       edge_color=(0, 0, 0, 0.3),  # transparent black
                       face_color='blue', face_alpha=0.2,
                       polyline_colors=None):
    """
    Visualize the 3D convex hull using matplotlib.
    
    Args:
        vertices: numpy.ndarray (Nx3)
            Vertex coordinates
        edges: numpy.ndarray (Mx2)
            Edge vertex indices
        triangles: numpy.ndarray (Kx3)
            Triangle vertex indices
        polylines: list of numpy.ndarray
            List of polylines, where each polyline is an array of vertex indices
        vertex_color: str or tuple
            Color for vertices
        edge_color: str or tuple
            Color for hull edges (default: transparent black)
        face_color: str or tuple
            Color for hull faces
        face_alpha: float
            Transparency for faces
        polyline_colors: list of str or tuple, optional
            Colors for each polyline. If None, cycles through default colors
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              color=vertex_color, s=50, label='Vertices')
    
    # Plot hull edges with transparency
    for edge in edges:
        edge_vertices = vertices[edge]
        ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], edge_vertices[:, 2],
               color=edge_color, linewidth=1, alpha=edge_color[3] if len(edge_color) == 4 else 1)
    
    # Plot hull faces
    triangles_vertices = vertices[triangles]
    hull_triangles = Poly3DCollection(triangles_vertices)
    hull_triangles.set_facecolor(face_color)
    hull_triangles.set_alpha(face_alpha)
    ax.add_collection3d(hull_triangles)
    
    # Plot polylines with different colors
    if polyline_colors is None:
        # Default color cycle
        polyline_colors = plt.cm.tab10(np.linspace(0, 1, len(polylines)))
    
    for polyline, color in zip(polylines, polyline_colors):
        # Get vertices for this polyline
        line_vertices = vertices[polyline]
        # Plot the polyline
        ax.plot(line_vertices[:, 0], line_vertices[:, 1], line_vertices[:, 2],
               color=color, linewidth=2, label=f'Polyline {len(polyline)} points')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# add color for each polyline
def plot_3d_sketch_with_normals(vertices, edges, polylines, edge_normals, normal_scale=0.1):
    """
    Plot 3D sketch with edges, normals, and polylines using matplotlib.
    
    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Array of vertex coordinates
    edges : np.ndarray, shape (M, 2)
        Array of edge vertex indices
    polylines : list of np.ndarray
        List of polylines, where each polyline is an array of 3D points
    edge_normals : np.ndarray, shape (M, 3)
        Array of edge normal vectors
    normal_scale : float, optional
        Scale factor for normal vector visualization length
    show_indices : bool, optional
        If True, show vertex and edge indices
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot edge normal
    for i, (v1_idx, v2_idx) in enumerate(edges):
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        
 
        # Plot edge normal at midpoint
        if np.any(edge_normals[i] != 0):  # Only plot non-zero normals
            midpoint = (v1 + v2) / 2
            normal = edge_normals[i] * normal_scale
            
            # Plot normal vector as an arrow
            ax.quiver(midpoint[0], midpoint[1], midpoint[2],
                     normal[0], normal[1], normal[2],
                     color='k', alpha=0.6, label='Normals' if i == 0 else "")
            

    # Plot polylines
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polylines)))  # Generate distinct colors
    for polyline, color in zip(polylines, colors):
        points = np.array(polyline)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 
               color=color, linewidth=2, alpha=0.8)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               color=color, s = 20)
    

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def export_sketch_normal_gltf(vertices, edges, P, edge_normals, filename="sketch_with_normal.gltf"):
    """
    Export 3D sketch with edges, normals, and polylines as a GLTF file.
    
    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Array of vertex coordinates
    edges : np.ndarray, shape (M, 2)
        Array of edge vertex indices
    P : list of np.ndarray
        List of polylines, where each polyline is vertex index
    edge_normals : array-like
        Array/list of edge normal vectors. Can be:
        - np.ndarray of shape (M, 3)
        - list of np.ndarray
        - list of lists
    filename : str
        Output GLTF filename
    """


    polylines = [[vertices[i] for i in line] for line in P]

    # to make the old code works, so I don't need to convert between formats
    normals = np.zeros((len(edges), 3))
    if isinstance(edge_normals, list) and all(isinstance(item, tuple) and len(item) == 2 for item in edge_normals):
        for edge_index, normal in edge_normals:
            normals[edge_index] = normal
    elif isinstance(edge_normals, np.ndarray):
        normals = edge_normals
    elif isinstance(edge_normals, list):  # Changed from "else isinstance()" to "elif isinstance()"
        normals = edge_normals
        

     
    # Initialize exporter
    exporter = GLTFGeometryExporter()
    
    # Constants for visualization
    NORMAL_SHAFT_RADIUS = 0.002
    NORMAL_HEAD_RADIUS = 0.004
    VERTEX_RADIUS = 0.005
    POLYLINE_RADIUS = 0.002
    NORMAL_SCALE = 0.1
    
    # Ensure inputs are numpy arrays
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    
    # Add edge normals (green arrows)
    edge_normal_points = []
    edge_normal_directions = []
    
    for (v1_idx, v2_idx), normal in zip(edges, normals):
        # Convert normal to numpy array if it isn't already
        normal = np.asarray(normal, dtype=float)
        
        if np.any(np.abs(normal) > 1e-15):  # More robust zero check
            # Calculate midpoint of edge for normal placement
            midpoint = ((vertices[v1_idx] + vertices[v2_idx]) / 2).tolist()
            edge_normal_points.append(midpoint)
            edge_normal_directions.append((normal * NORMAL_SCALE).tolist())
    
    if edge_normal_points:
        exporter.add_normal_arrows(edge_normal_points, edge_normal_directions,
                                 color=(0, 1, 0),  # Green normals
                                 shaft_radius=NORMAL_SHAFT_RADIUS,
                                 head_radius=NORMAL_HEAD_RADIUS)
    
    # Generate colors for polylines using the same rainbow colormap
    num_polylines = len(polylines)
    colors = []
    for i in range(num_polylines):
        # Convert matplotlib's rainbow colors to RGB
        rgba = plt.cm.rainbow(i / max(1, num_polylines - 1))
        colors.append(rgba[:3])  # Take only RGB values, ignore alpha
    
    # Add polylines and their vertices
    for polyline, color in zip(polylines, colors):
        # Add the polyline as cylinder strips
        exporter.add_cylinder_strips(polyline, 
                                   color=color,
                                   radius=POLYLINE_RADIUS,
                                   add_spheres=False)
        
        # Add vertices as spheres with matching color
        exporter.add_spheres(polyline,
                           color=color,
                           radius=VERTEX_RADIUS)
    
    # Save the GLTF file
    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")

 
def export_sketch_polyline_normal_gltf(vertices, edges, polylines, polyline_normals, filename="sketch_with_normal.gltf", save_debug_gltf=True):
    """
    Export 3D sketch with edges, normals, and polylines as a GLTF file.
    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Array of vertex coordinates
    edges : np.ndarray, shape (M, 2)
        Array of edge vertex indices
    polylines : list of lists
        List of polylines, where each polyline is a list of vertex indices
    polyline_normals : dict
        Dictionary in one of two formats:
        1. {polyline_idx: (position_in_polyline, normal_vector)}
           - polyline_idx: index of the polyline
           - position_in_polyline: position of the edge in the polyline
           - normal_vector: reference normal vector
        2. {polyline_idx: [normal_vectors]}
           - polyline_idx: index of the polyline
           - normal_vectors: list of normal vectors corresponding to each edge in the polyline
    filename : str
        Output GLTF filename
    save_debug_gltf : bool, default=True
        Whether to save the GLTF file
    """
    # Skip if saving is disabled
    if not save_debug_gltf:
        print(f"GLTF export skipped for: {filename}")
        return
        
    # Initialize exporter
    exporter = GLTFGeometryExporter()
    # Constants for visualization
    NORMAL_SHAFT_RADIUS = 0.002
    NORMAL_HEAD_RADIUS = 0.004
    VERTEX_RADIUS = 0.005
    POLYLINE_RADIUS = 0.002
    NORMAL_SCALE = 0.1
    # Ensure vertices is a numpy array
    vertices = np.asarray(vertices)
    # Add edge normals (green arrows)
    edge_normal_points = []
    edge_normal_directions = []
    
    # Process normals from the polyline_normal dictionary
    for polyline_idx, normal_data in polyline_normals.items():
        # Get the polyline
        polyline = polylines[polyline_idx]
        
        # Determine format: single normal with position or list of normals
        if isinstance(normal_data, tuple):
            # Format 1: (position_in_polyline, normal_vector)
            edge_pos, normal = normal_data
            
            # Get the edge vertices
            if edge_pos + 1 < len(polyline):
                v1_idx = polyline[edge_pos]
                v2_idx = polyline[edge_pos + 1]
                
                # Convert normal to numpy array if it isn't already
                normal = np.asarray(normal, dtype=float)
                
                if np.any(np.abs(normal) > 1e-15):  # More robust zero check
                    # Calculate midpoint of edge for normal placement
                    midpoint = ((vertices[v1_idx] + vertices[v2_idx]) / 2).tolist()
                    edge_normal_points.append(midpoint)
                    edge_normal_directions.append((normal * NORMAL_SCALE).tolist())
        
        elif isinstance(normal_data, list):
            # Format 2: [normal_vectors] - list of normals for each edge
            for edge_idx, normal in enumerate(normal_data):
                # Check if we have a valid edge (need two vertices)
                if edge_idx + 1 < len(polyline):
                    v1_idx = polyline[edge_idx]
                    v2_idx = polyline[edge_idx + 1]
                    
                    # Convert normal to numpy array if it isn't already
                    normal = np.asarray(normal, dtype=float)
                    
                    if np.any(np.abs(normal) > 1e-15):  # More robust zero check
                        # Calculate midpoint of edge for normal placement
                        midpoint = ((vertices[v1_idx] + vertices[v2_idx]) / 2).tolist()
                        edge_normal_points.append(midpoint)
                        edge_normal_directions.append((normal * NORMAL_SCALE).tolist())
    
    if edge_normal_points:
        exporter.add_normal_arrows(edge_normal_points, edge_normal_directions,
                                    color=(0, 1, 0),  # Green normals
                                    shaft_radius=NORMAL_SHAFT_RADIUS,
                                    head_radius=NORMAL_HEAD_RADIUS)
    
    # Generate colors for polylines using the same rainbow colormap
    num_polylines = len(polylines)
    colors = []
    for i in range(num_polylines):
        # Convert matplotlib's rainbow colors to RGB
        rgba = plt.cm.rainbow(i / max(1, num_polylines - 1))
        colors.append(rgba[:3])  # Take only RGB values, ignore alpha
    
    # Add polylines and their vertices
    for i, (polyline, color) in enumerate(zip(polylines, colors)):
        # Convert vertex indices to 3D points
        polyline_points = np.array([vertices[idx] for idx in polyline])
        
        # Add the polyline as cylinder strips
        exporter.add_cylinder_strips(polyline_points, 
                                    color=color,
                                    radius=POLYLINE_RADIUS,
                                    add_spheres=False)
        
        # Add vertices as spheres with matching color
        exporter.add_spheres(polyline_points,
                            color=color,
                            radius=VERTEX_RADIUS)
    
    # Save the GLTF file
    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")



def plot_hull_with_normals(vertices, edges, faces, vertex_normals, edge_normals, face_normals):
    """
    Simple visualization of a 3D hull with its normals.
    """
    scale_factor = 0.08
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              color='k', s=20)
    
    # Plot edges
    for edge in edges:
        v1, v2 = vertices[edge]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                color='gray', linewidth=1)
    
    # Plot faces
    face_vertices = []
    for face in faces:
        face_vertices.append(vertices[face])
    
    hull_faces = Poly3DCollection(face_vertices, alpha=0.1)
    hull_faces.set_facecolor('lightgray')
    hull_faces.set_edgecolor('none')
    ax.add_collection3d(hull_faces)
    
    # Plot normals
    # Vertex normals (red)
    for vertex, normal in zip(vertices, vertex_normals):
        if np.any(normal != 0):
            ax.quiver(vertex[0], vertex[1], vertex[2],
                     normal[0], normal[1], normal[2],
                     color='red', length=scale_factor, 
                     arrow_length_ratio=0.2, linewidth=1)
    
    # Edge normals (green)
    for edge, normal in zip(edges, edge_normals):
        if np.any(normal != 0):
            midpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2
            ax.quiver(midpoint[0], midpoint[1], midpoint[2],
                     normal[0], normal[1], normal[2],
                     color='green', length=scale_factor, 
                     arrow_length_ratio=0.2, linewidth=1)
    
    # Face normals (blue)
    for face, normal in zip(faces, face_normals):
        if np.any(normal != 0):
            face_center = vertices[face].mean(axis=0)
            ax.quiver(face_center[0], face_center[1], face_center[2],
                     normal[0], normal[1], normal[2],
                     color='blue', length=scale_factor, 
                     arrow_length_ratio=0.2, linewidth=1)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def export_polylines_gltf(polylines, filename = "sketch.gltf"):
    '''
    ''' 
    exporter = GLTFGeometryExporter()


    for polyline in polylines:
        exporter.add_cylinder_strips(polyline, radius=0.002)
    
    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")


### helper -- not used 
# no color showing polyline
def plot_edges_and_normals(vertices, edges, edge_normals, normal_scale=0.1, show_indices=False):
    """
    Plot 3D edges and their normals using matplotlib.
    
    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Array of vertex coordinates
    edges : np.ndarray, shape (M, 2)
        Array of edge vertex indices
    edge_normals : np.ndarray, shape (M, 3)
        Array of edge normal vectors
    normal_scale : float, optional
        Scale factor for normal vector visualization length
    show_indices : bool, optional
        If True, show vertex and edge indices
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot edges
    for i, (v1_idx, v2_idx) in enumerate(edges):
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        
        # Plot edge as a line
        ax.plot([v1[0], v2[0]], 
                [v1[1], v2[1]], 
                [v1[2], v2[2]], 
                'b-', linewidth=2, label='Edges' if i == 0 else "")
        
        # Plot edge normal at midpoint
        if np.any(edge_normals[i] != 0):  # Only plot non-zero normals
            midpoint = (v1 + v2) / 2
            normal = edge_normals[i] * normal_scale
            
            # Plot normal vector as an arrow
            ax.quiver(midpoint[0], midpoint[1], midpoint[2],
                     normal[0], normal[1], normal[2],
                     color='r', alpha=0.6, label='Normals' if i == 0 else "")
            
        # Add edge index if requested
        if show_indices:
            midpoint = (v1 + v2) / 2
            ax.text(midpoint[0], midpoint[1], midpoint[2], 
                   f'E{i}', color='blue')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c='k', s=50, label='Vertices')
    
    # Add vertex indices if requested
    if show_indices:
        for i, vertex in enumerate(vertices):
            ax.text(vertex[0], vertex[1], vertex[2], 
                   f'V{i}', color='black')
    
 
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

def export_hull_gltf(vertices, edges, faces, vertex_normals, edge_normals, face_normals,
                               filename="hull_with_normals.gltf"):
    """
    Save hull geometry and normals as a GLTF file.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Array of shape (N, 3) containing vertex coordinates
    edges : np.ndarray
        Array of shape (M, 2) containing vertex indices that form edges
    faces : np.ndarray
        Array of shape (K, 3) containing vertex indices that form triangular faces
    vertex_normals : np.ndarray
        Array of shape (N, 3) containing normalized vertex normals
    edge_normals : np.ndarray
        Array of shape (M, 3) containing normalized edge normals
    face_normals : np.ndarray
        Array of shape (K, 3) containing normalized face normals
    polylines : list of np.ndarray
        List of polylines, where each polyline is an array of points
    filename : str
        Output GLTF filename
    """
    # Initialize exporter
    exporter = GLTFGeometryExporter()
    
    # Constants for visualization
    EDGE_RADIUS = 0.005
    NORMAL_SHAFT_RADIUS = 0.002
    NORMAL_HEAD_RADIUS = 0.004
    NORMAL_LENGTH = 0.08
    VERTEX_RADIUS = 0.01
    POLYLINE_RADIUS = 0.002
    
    # Add faces as triangles (light gray, semi-transparent)
    face_vertices = vertices.tolist()
    faces_list = faces.tolist()
    
    # Add edges as cylinder strips (dark gray)
    for edge in edges:
        edge_points = [vertices[edge[0]], vertices[edge[1]]]
        exporter.add_cylinder_strips(edge_points, 
                                   color=(0.5, 0.5, 0.5),
                                   radius=EDGE_RADIUS,
                                   add_spheres=False)
    
    # Add vertices as spheres (black)
    exporter.add_spheres(vertices.tolist(), 
                        color=(0.1, 0.1, 0.1),
                        radius=VERTEX_RADIUS)
    
    # Add vertex normals (red arrows)
    vertex_points = []
    vertex_directions = []
    for vertex, normal in zip(vertices, vertex_normals):
        if np.any(normal != 0):
            vertex_points.append(vertex.tolist())
            vertex_directions.append((normal * NORMAL_LENGTH).tolist())
    
    if vertex_points:
        exporter.add_normal_arrows(vertex_points, vertex_directions,
                                 color=(1, 0, 0),
                                 shaft_radius=NORMAL_SHAFT_RADIUS,
                                 head_radius=NORMAL_HEAD_RADIUS)
    
    # Add edge normals (green arrows)
    edge_normal_points = []
    edge_normal_directions = []
    for edge, normal in zip(edges, edge_normals):
        if np.any(normal != 0):
            midpoint = ((vertices[edge[0]] + vertices[edge[1]]) / 2).tolist()
            edge_normal_points.append(midpoint)
            edge_normal_directions.append((normal * NORMAL_LENGTH).tolist())
    
    if edge_normal_points:
        exporter.add_normal_arrows(edge_normal_points, edge_normal_directions,
                                 color=(0, 1, 0),
                                 shaft_radius=NORMAL_SHAFT_RADIUS,
                                 head_radius=NORMAL_HEAD_RADIUS)
    
    # Add face normals (blue arrows)
    face_normal_points = []
    face_normal_directions = []
    for face, normal in zip(faces, face_normals):
        if np.any(normal != 0):
            face_center = vertices[face].mean(axis=0).tolist()
            face_normal_points.append(face_center)
            face_normal_directions.append((normal * NORMAL_LENGTH).tolist())
    
    if face_normal_points:
        exporter.add_normal_arrows(face_normal_points, face_normal_directions,
                                 color=(0, 0, 1),
                                 shaft_radius=NORMAL_SHAFT_RADIUS,
                                 head_radius=NORMAL_HEAD_RADIUS)
    
    # Save the GLTF file
    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")

### helper -- tester 
def test_nearby_faces(edges, hull_vertices, hull_faces, epsilon):
    '''
    Randomly selects an edge and visualizes it along with nearby faces on the convex hull.
    Args:
        edges: Array of vertex index pairs defining edges
        hull_vertices: Array of 3D coordinates of hull vertices
        hull_faces: Array of triangular faces defined by vertex indices
        epsilon: Distance threshold for finding nearby faces
    Displays:
        3D plot showing:
        - All edges of the hull in grey
        - Randomly selected edge highlighted in red
        - Nearby faces highlighted in different colors
    '''
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import colorsys

    # Select random edge and get its vertices
    random_edge = random.choice(edges)
    random_edge = [39, 38]
    print("Selected edge vertex indices:", random_edge)
    line = [hull_vertices[random_edge[0]], hull_vertices[random_edge[1]]]



    # Find nearby faces
    nearby_faces = line_nearby_hull_faces(line, hull_vertices, hull_faces, epsilon=1e-2)

    print('nearby_faces', nearby_faces)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot all edges in grey
    for edge in edges:
        v1, v2 = hull_vertices[edge[0]], hull_vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'gray', alpha=0.3)

    # Plot highlighted random edge in red
    ax.plot([line[0][0], line[1][0]],
            [line[0][1], line[1][1]],
            [line[0][2], line[1][2]], 'r-', linewidth=3, label='Selected Edge')

    # Generate distinct colors for each face using HSV color space
    n_faces = len(nearby_faces)
    colors = [colorsys.hsv_to_rgb(i/n_faces, 0.7, 0.9) for i in range(n_faces)]

    # Plot nearby faces with different colors
    for face_idx, color in zip(nearby_faces, colors):
        triangle = hull_vertices[hull_faces[face_idx]]
        # tri = ax.plot_trisurf(triangle[:,0], triangle[:,1], triangle[:,2],
        #                     color=color, alpha=0.3)
        tri = robust_plot_trisurf(ax, triangle, color='blue', alpha=0.3)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def robust_plot_trisurf(ax, triangle, color='blue', alpha=0.3):
    """
    A robust version of plot_trisurf that handles degenerate cases.
    """
    try:
        # First attempt: direct trisurf
        tri = ax.plot_trisurf(triangle[:,0], triangle[:,1], triangle[:,2],
                             color=color, alpha=alpha)
        return tri
    except RuntimeError:
        try:
            # Second attempt: Add tiny perturbation
            perturbed = triangle + np.random.normal(0, 1e-6, triangle.shape)
            tri = ax.plot_trisurf(perturbed[:,0], perturbed[:,1], perturbed[:,2],
                                color=color, alpha=alpha)
            return tri
        except RuntimeError:
            try:
                # Third attempt: Create manual triangulation
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Create triangles
                triangles = [[triangle[0], triangle[1], triangle[2]]]
                
                # Create polygon collection
                poly = Poly3DCollection(triangles, alpha=alpha)
                poly.set_color(color)
                ax.add_collection3d(poly)
                
                return poly
            except Exception as e:
                print(f"Failed to create surface: {str(e)}")
                # If all else fails, create a simple patch
                ax.plot_trisurf(triangle[:,0], triangle[:,1], triangle[:,2],
                              color=color, alpha=alpha, shade=False)
                return None



if __name__ == "__main__":

    # from PyQt6.QtWidgets import QApplication, QFileDialog

    # app = QApplication([])
    # curve_file, _ = QFileDialog.getOpenFileName(None, "Open File", 'files_with_normals', "obj (*.obj)")

    import argparse
    parser = argparse.ArgumentParser(description='Optimize all edges')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')

    args = parser.parse_args()

    curve_file = args.curve_file

    if curve_file is None:


    

        curve_file = 'files_starter/simple_objs/flowrep_spherecylinder.obj'
        curve_file = 'files_starter/simple_objs/flowrep_trebol.obj'
        # curve_file = 'files_starter/simple_objs/onshape_bishop.obj'
        curve_file = 'files_starter/simple_objs/onshape_simple_mouse.obj'
        # curve_file = 'files_starter/simple_objs/onshape_simple_shape.obj'
        # curve_file = 'files_starter/simple_objs/author2_sofa.obj'

 
    # gltf_sketch_path = 'files_starter/gltfs/sketches/'
    # gltf_hull_path = 'files_starter/gltfs/convex_hull/'
    # gltf_normal_path = 'files_starter/gltfs/edge_normals/'


    gltf_sketch_path = 'files_starter/gltfs/convex_hull/'
    gltf_hull_path = 'files_starter/gltfs/convex_hull/'
    gltf_normal_path = 'files_starter/gltfs/convex_hull/'
    
    curve_name = Path(curve_file).stem

    ### 1. load data and make convex hull 
    V, E, P = load_sketch_polyline_data(curve_file)
    polylines = [[V[i] for i in line] for line in P]


    # export the sketch to gltf
    # export_polylines_gltf(polylines,  gltf_sketch_path + curve_name + '.gltf')
    
    
    
    points = V
    vertices, edges, faces = compute_3d_convex_hull(points)

    plot_3d_geometry(vertices, edges, faces, P)

    ### test line_triangle_distance 
    # for i in range(10):
    #     test_nearby_faces(edges, vertices, faces, epsilon=1e-6)

    # test_nearby_faces(edges, vertices, faces,  epsilon=1e-2)


    # exporter.save('output.gltf')

    

    ### 2.  export the hull and normals to gltf
    vertex_normals, edge_normals, face_normals = compute_hull_normals(vertices, edges, faces)

    plot_hull_with_normals(vertices, edges, faces, vertex_normals, edge_normals, face_normals)

    # export_hull_gltf(vertices, edges, faces, vertex_normals, edge_normals, face_normals, gltf_hull_path + curve_name + '.gltf')


    # ### 3. calculate the edge normal 
    # edge_normals = calculate_sketch_edge_normals(V, E , epsilon=5e-3)
    
    # plot_3d_sketch_with_normals(V, E, polylines, edge_normals)

    # export_sketch_normal_gltf(V, E, polylines, edge_normals, gltf_normal_path + curve_name + '.gltf')


    # ### 4. the edge normal need to be filtered

    # edge_constraints = get_sketch_edge_constraints(V, E)

    # print(len(edge_constraints))


        


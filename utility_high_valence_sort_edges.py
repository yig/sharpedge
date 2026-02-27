import numpy as np
import argparse


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from utility_parallel_transport import perpendicular_normal

from utility_io import load_sketch_polyline_data

def compute_edge_circulation_PCA(edge_indices, vertex_idx, E, V):
    """
    Compute the counter-clockwise circulation ordering of edges around a vertex.
    """
    vertex_pos = V[vertex_idx]

    # Filter for incident edges
    incident_edges = [ei for ei in edge_indices if vertex_idx in E[ei]]
    if len(incident_edges) <= 2:
        return incident_edges

    vectors = []
    vector_array = np.zeros((len(incident_edges), 3))
    
    for i, ei in enumerate(incident_edges):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        vec = V[other_idx] - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, ei))
        vector_array[i] = unit_vec

    centered_data = vector_array - np.mean(vector_array, axis=0)
    _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
    basis_1, basis_2 = Vt[0], Vt[1]

    projected_vectors = []
    for unit_vec, ei in vectors:
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, ei))

    projected_vectors.sort(key=lambda x: x[0])
    ordered_edges = [ei for _, ei in projected_vectors]
    return ordered_edges

def compute_edge_circulation_graph_laplacian(edge_indices, vertex_idx, E, V):
    """
    Compute the counter-clockwise circulation ordering of edges around a vertex.
    """
    vertex_pos = V[vertex_idx]

    # Filter for incident edges
    incident_edges = [ei for ei in edge_indices if vertex_idx in E[ei]]
    if len(incident_edges) <= 2:
        return incident_edges

    vectors = []
    vector_array = np.zeros((len(incident_edges), 3))
    
    for i, ei in enumerate(incident_edges):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        vec = V[other_idx] - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, ei))
        vector_array[i] = unit_vec

    Hn = np.average( vector_array, axis = 0 )
    Hn_norm = np.linalg.norm(Hn)
    ## We will have mean curvature near 0 if the surface is flat or saddle.
    ## In that case, the PCA normal should provide a good plane for projection.
    if Hn_norm < 1e-5:
        return compute_edge_circulation_PCA( edge_indices, vertex_idx, E, V )
    else:
        Hn /= Hn_norm
        basis_1 = perpendicular_normal( Hn )
        basis_2 = np.cross( Hn, basis_1 )

    projected_vectors = []
    for unit_vec, ei in vectors:
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, ei))

    projected_vectors.sort(key=lambda x: x[0])
    ordered_edges = [ei for _, ei in projected_vectors]
    return ordered_edges

def compute_edge_circulation_from_edge_one_normals(edge_indices, vertex_idx, E, V, one_normals):
    """
    Compute the counter-clockwise circulation ordering of edges around a vertex.

    one_normals: A dictionary from an edge index to its one normal vector.
    """
    vertex_pos = V[vertex_idx]

    # Filter for incident edges
    incident_edges = [ei for ei in edge_indices if vertex_idx in E[ei]]
    if len(incident_edges) <= 2:
        return incident_edges

    normal_vectors = []
    normal_vector_array = np.zeros((len(incident_edges), 3))
    
    edge_vectors = []

    for i, ei in enumerate(incident_edges):
        normal_vec = one_normals[ei]
        normal_vectors.append((normal_vec, ei))
        normal_vector_array[i] = normal_vec

        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        edge_vec = V[other_idx] - vertex_pos
        unit_edge_vec = edge_vec / np.linalg.norm(edge_vec)
        edge_vectors.append((unit_edge_vec, ei))

    N = np.average( normal_vector_array, axis = 0 )
    N_norm = np.linalg.norm(N)
    ## If the one normal norm is near 0, fall back to the graph laplacian method.
    if N_norm < 1e-5:
        return compute_edge_circulation_graph_laplacian( edge_indices, vertex_idx, E, V )
    else:
        N /= N_norm
        basis_1 = perpendicular_normal( N )
        basis_2 = np.cross( N, basis_1 )

    projected_vectors = []
    for unit_edge_vec, ei in edge_vectors:
        proj_1 = np.dot(unit_edge_vec, basis_1)
        proj_2 = np.dot(unit_edge_vec, basis_2)
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, ei))

    projected_vectors.sort(key=lambda x: x[0])
    ordered_edges = [ei for _, ei in projected_vectors]
    return ordered_edges

def plot_sorted_edges(vertex_idx, ordered_edge_indices, E, V, ax=None):
    """
    Plot the ordered edges around a vertex in 3D, labeling each edge with its edge index.
    All edges are shown in very light gray, with sorted edges highlighted in color.
    """
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot all edges in very light gray first
    for ei, (v1, v2) in enumerate(E):
        edge_pts = np.array([V[v1], V[v2]])
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color='gray', alpha=0.3, linewidth=1)

    # Plot all vertices
    ax.scatter(V[:,0], V[:,1], V[:,2], color='lightgray', s=20, alpha=0.5)

    # Highlight the central vertex
    vertex_pos = V[vertex_idx]
    ax.scatter(*vertex_pos, color='red', s=100, label=f'Central vertex {vertex_idx}')

    # Plot the sorted edges with colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_edge_indices)))
    
    for i, ei in enumerate(ordered_edge_indices):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        edge_pts = np.array([vertex_pos, V[other_idx]])
        
        # Plot the sorted edge with color
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color=colors[i], linewidth=3)
        
        # Highlight the connected vertex
        ax.scatter(*V[other_idx], color=colors[i], s=60)

        # Annotate with actual edge index
        mid = (vertex_pos + V[other_idx]) / 2
    ax.set_title(f"Circulation around vertex {vertex_idx}")
    ax.legend()
    
    plt.axis('off')
    plt.axis('equal')
    plt.show()

def plot_sorted_edges_with_circulation(vertex_idx, ordered_edge_indices, E, V, ax=None):
    """
    Plot the ordered edges around a vertex in 3D with circulation arrows.
    Shows edge indices and circulation order clearly.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot all edges in very light gray first
    for ei, (v1, v2) in enumerate(E):
        edge_pts = np.array([V[v1], V[v2]])
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color='gray', alpha=0.2, linewidth=0.5)

    # Plot all vertices in light gray
    ax.scatter(V[:,0], V[:,1], V[:,2], color='lightgray', s=10, alpha=0.3)

    # Highlight the central vertex
    vertex_pos = V[vertex_idx]
    ax.scatter(*vertex_pos, color='red', s=150, label=f'Vertex {vertex_idx}', zorder=10)

    # Define colors for the circulation
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_edge_indices)))
    
    # Plot the sorted edges with colors and labels
    connected_vertices = []
    for i, ei in enumerate(ordered_edge_indices):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        connected_vertices.append(V[other_idx])
        
        edge_pts = np.array([vertex_pos, V[other_idx]])
        
        # Plot the edge with color
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color=colors[i], linewidth=3, alpha=0.8, zorder=5)
        
        # Highlight the connected vertex
        ax.scatter(*V[other_idx], color=colors[i], s=80, zorder=8)
        
        # Add edge index label near the connected vertex
        offset = 0.1 * (V[other_idx] - vertex_pos) / np.linalg.norm(V[other_idx] - vertex_pos)
        label_pos = V[other_idx] + offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], f'{ei}', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw circulation arrows between consecutive edges
    draw_circulation_arrows(ax, vertex_pos, connected_vertices, colors)
    
    ax.set_title(f"Edge Circulation around Vertex {vertex_idx}\n"
                f"Order: {' → '.join(map(str, ordered_edge_indices))}", fontsize=14)
    ax.legend()
    
    # Remove axis for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.tight_layout()
    plt.axis('off')
    plt.axis('equal')
    plt.show()

def draw_circulation_arrows(ax, center, connected_vertices, colors):
    """
    Draw curved arrows showing the circulation order between edges
    """
    n = len(connected_vertices)
    if n < 2:
        return
    
    # Create circulation arrows
    for i in range(n):
        # Current and next vertex (with wraparound)
        current_vertex = connected_vertices[i]
        next_vertex = connected_vertices[(i + 1) % n]
        
        # Create a curved arrow between consecutive edges
        draw_curved_arrow_3d(ax, center, current_vertex, next_vertex, 
                           colors[i], colors[(i + 1) % n])

def draw_curved_arrow_3d(ax, center, start_vertex, end_vertex, start_color, end_color):
    """
    Draw a curved arrow in 3D space showing circulation from one edge to the next
    """
    # Calculate vectors from center to vertices
    vec_start = start_vertex - center
    vec_end = end_vertex - center
    
    # Normalize vectors
    vec_start_norm = vec_start / np.linalg.norm(vec_start)
    vec_end_norm = vec_end / np.linalg.norm(vec_end)
    
    # Create intermediate points for the curved arrow
    # Use shorter radius for the circulation arrow
    radius = 0.3 * min(np.linalg.norm(vec_start), np.linalg.norm(vec_end))
    
    # Calculate the arc between the two directions
    cross_product = np.cross(vec_start_norm, vec_end_norm)
    if np.linalg.norm(cross_product) > 1e-6:  # Not parallel
        # Create arc points
        n_points = 20
        t_values = np.linspace(0, 1, n_points)
        
        arc_points = []
        for t in t_values:
            # Spherical interpolation (slerp)
            angle = np.arccos(np.clip(np.dot(vec_start_norm, vec_end_norm), -1, 1))
            if angle > 1e-6:  # Not the same direction
                interp_vec = (np.sin((1-t) * angle) * vec_start_norm + 
                            np.sin(t * angle) * vec_end_norm) / np.sin(angle)
            else:
                interp_vec = vec_start_norm
            
            arc_point = center + radius * interp_vec
            arc_points.append(arc_point)
        
        arc_points = np.array(arc_points)
        
        # Plot the curved arrow
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
                color='purple', alpha=0.6, linewidth=2, linestyle='--')
        
        # Add arrowhead at the end
        if len(arc_points) > 1:
            arrow_start = arc_points[-2]
            arrow_end = arc_points[-1]
            arrow_dir = arrow_end - arrow_start
            arrow_dir = arrow_dir / np.linalg.norm(arrow_dir)
            
            # Create simple arrowhead (just a point for now in 3D)
            ax.scatter(*arrow_end, color='purple', s=30, marker='>', alpha=0.8)

def build_vertex_to_edges_map(edges):
    '''
    Create a mapping from each vertex to all edges that contain it.
    
    Parameters:
    vertices: (n,3) array of vertex coordinates
    edges: (m,2) array of edge vertex index pairs
    
    Returns:
    dict: Mapping from vertex index to list of edge indices
    '''
    vertex_to_edges = defaultdict(list)
    
    for edge_idx, edge in enumerate(edges):
        # Add this edge to both of its vertices' lists
        vertex_to_edges[edge[0]].append(edge_idx)
        vertex_to_edges[edge[1]].append(edge_idx)
    
    for vertex_idx in vertex_to_edges:
        assert len(vertex_to_edges[vertex_idx]) == len(set(vertex_to_edges[vertex_idx])), \
            f"Vertex {vertex_idx} has duplicate edge entries"
    
    return vertex_to_edges

def plot_projection_plane_and_points(vertex_idx, edge_indices, E, V, auto_detect_method=True):
    """
    Visualize the projection plane, the original 3D points (vertices), and their projections.
    Original vertices in black, projections in blue.
    
    Parameters:
    -----------
    vertex_idx : int
        Index of the central vertex
    edge_indices : list
        List of edge indices around the vertex
    E : array
        Edge array
    V : array  
        Vertex array
    auto_detect_method : bool
        If True, automatically detects whether to use PCA or Graph Laplacian based on 
        the same logic as compute_edge_circulation_graph_laplacian
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    vertex_pos = V[vertex_idx]
    ax.scatter(V[:,0], V[:,1], V[:,2], alpha=0.3, s=20)

    # Collect all edge endpoints and vectors
    incident_edges = []
    edge_endpoints = []  # Store the actual points, not just vectors
    vectors = []
    vector_array = []
    
    for ei in edge_indices:
        v1, v2 = E[ei]
        if vertex_idx in [v1, v2]:
            incident_edges.append(ei)
            other_idx = v2 if v1 == vertex_idx else v1
            other_pos = V[other_idx]  # The position of the other endpoint
            
            # Store actual point
            edge_endpoints.append((other_pos, ei))
            
            # Also compute vector (for plane computation)
            vec = other_pos - vertex_pos
            unit_vec = vec / np.linalg.norm(vec)
            vectors.append((unit_vec, ei))
            vector_array.append(unit_vec)
    
    vector_array = np.array(vector_array)
    
    # Plot the central vertex
    ax.scatter(*vertex_pos, color='red', s=100, label=f'Vertex {vertex_idx}')
    
    # Compute projection plane using auto-detection logic (same as graph_laplacian method)
    if auto_detect_method:
        # First, compute mean curvature vector to decide which method to use
        Hn = np.average(vector_array, axis=0)
        Hn_norm = np.linalg.norm(Hn)
        
        print(f"Auto-detection - Mean curvature norm: {Hn_norm:.6f}")
        
        if Hn_norm < 1e-5:
            # Use PCA method (same logic as graph_laplacian fallback)
            print("  Using PCA method (flat/saddle surface detected)")
            centered_data = vector_array - np.mean(vector_array, axis=0)
            _, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
            basis_1, basis_2, basis_3 = Vt[0], Vt[1], Vt[2]
            
            # Print variance information
            total_variance = np.sum(S**2)
            variance_explained = (S[0]**2 + S[1]**2) / total_variance
            print(f"  PCA - Variance explained by projection plane: {variance_explained:.4f}")
            method_title = "Auto-detected: PCA"
        else:
            # Use Graph Laplacian method
            print("  Using Graph Laplacian method (curved surface detected)")
            Hn /= Hn_norm
            basis_1 = perpendicular_normal(Hn)
            basis_2 = np.cross(Hn, basis_1)
            basis_3 = Hn  # Normal to the plane
            method_title = "Auto-detected: Graph Laplacian"
    else:
        # Legacy manual method selection (keeping for backwards compatibility)
        method = 'PCA'  # Default fallback
        if method.lower() == 'pca':
            # PCA method
            centered_data = vector_array - np.mean(vector_array, axis=0)
            _, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
            basis_1, basis_2, basis_3 = Vt[0], Vt[1], Vt[2]
            
            # Print variance information
            total_variance = np.sum(S**2)
            variance_explained = (S[0]**2 + S[1]**2) / total_variance
            print(f"PCA - Variance explained by projection plane: {variance_explained:.4f}")
            method_title = "PCA"
        elif method.lower() == 'graph_laplacian':
            # Graph Laplacian method
            Hn = np.average(vector_array, axis=0)
            Hn_norm = np.linalg.norm(Hn)
            
            print(f"Graph Laplacian - Mean curvature norm: {Hn_norm:.6f}")
            
            if Hn_norm < 1e-5:
                print("  Falling back to PCA (flat/saddle surface)")
                # Fall back to PCA
                centered_data = vector_array - np.mean(vector_array, axis=0)
                _, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
                basis_1, basis_2, basis_3 = Vt[0], Vt[1], Vt[2]
                method_title = "Graph Laplacian (PCA fallback)"
            else:
                # Use mean curvature normal
                Hn /= Hn_norm
                basis_1 = perpendicular_normal(Hn)
                basis_2 = np.cross(Hn, basis_1)
                basis_3 = Hn  # Normal to the plane
                method_title = "Graph Laplacian"
        else:
            raise ValueError("method must be 'PCA' or 'graph_laplacian'")
    
    # Create a grid for the projection plane
    center = vertex_pos
    # Scale factors for the plane size - adjust based on your data
    max_dist = max([np.linalg.norm(point - vertex_pos) for point, _ in edge_endpoints])
    scale = max_dist * 1.2  # Make plane slightly larger than the furthest point
    
    # Create a grid of points on the plane
    u = np.linspace(-1, 1, 10) * scale
    v = np.linspace(-1, 1, 10) * scale
    
    plane_points = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            # Compute point on the plane
            point = center + u[i] * basis_1 + v[j] * basis_2
            plane_points[i, j] = point
    
    # Plot the projection plane as a surface
    ax.plot_surface(plane_points[:, :, 0], 
                    plane_points[:, :, 1], 
                    plane_points[:, :, 2], 
                    alpha=0.2, color='lightblue')
    
    # Plot basis vectors of the plane
    scale_basis = scale * 0.5  # Scale for basis vectors visualization
    
    # Basis 1 (1st principal component or perpendicular to mean curvature)
    ax.quiver(center[0], center[1], center[2], 
              basis_1[0]*scale_basis, basis_1[1]*scale_basis, basis_1[2]*scale_basis, 
              color='r', arrow_length_ratio=0.1, label='Basis 1')
    
    # Basis 2 (2nd principal component or cross product)
    ax.quiver(center[0], center[1], center[2], 
              basis_2[0]*scale_basis, basis_2[1]*scale_basis, basis_2[2]*scale_basis, 
              color='g', arrow_length_ratio=0.1, label='Basis 2')
    
    # Basis 3 (normal to plane)
    normal_color = 'blue' if 'PCA' in method_title else 'purple'
    normal_label = 'PCA Normal' if 'PCA' in method_title else 'Mean Curvature Normal'
    ax.quiver(center[0], center[1], center[2], 
              basis_3[0]*scale_basis, basis_3[1]*scale_basis, basis_3[2]*scale_basis, 
              color=normal_color, arrow_length_ratio=0.1, label=normal_label)
    
    # Plot lines from center to original points and their projections
    angles_and_edges = []  # Store for sorting verification
    
    for point, ei in edge_endpoints:
        # Draw line from center to original point
        ax.plot([vertex_pos[0], point[0]], 
                [vertex_pos[1], point[1]], 
                [vertex_pos[2], point[2]], 
                'k-', linewidth=1.5)
        
        # Plot original point (black)
        ax.scatter(*point, color='black', s=60)
        
        # Add edge index label at the point
        ax.text(*point, f'e{ei}', fontsize=10, color='black')
        
        # Compute projection of this point onto the plane
        vec = point - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)  # Normalize like in the main functions
        
        # Project the unit vector directly onto the plane
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        
        # Compute the projected point in 3D space
        proj_point = vertex_pos + proj_1 * basis_1 + proj_2 * basis_2
        
        # Plot projected point (blue)
        ax.scatter(*proj_point, color='blue', s=60)
        
        # Add projected edge index label
        ax.text(*proj_point, f'p{ei}', fontsize=10, color='blue')
        
        # Draw line from original point to its projection
        ax.plot([point[0], proj_point[0]], 
                [point[1], proj_point[1]], 
                [point[2], proj_point[2]], 
                'k--', alpha=0.4)
        
        # Draw line from center to projected point
        ax.plot([vertex_pos[0], proj_point[0]], 
                [vertex_pos[1], proj_point[1]], 
                [vertex_pos[2], proj_point[2]], 
                'b-', linewidth=1.5, alpha=0.7)
        
        # Compute and show angle in the plane
        angle = np.arctan2(proj_2, proj_1)
        angle_deg = np.degrees(angle)
        # ax.text(*proj_point, f'\n{angle_deg:.1f}°', fontsize=8, color='blue')
        
        # Store for verification
        angles_and_edges.append((angle, ei))
    
    # Print the angle ordering for verification
    angles_and_edges.sort(key=lambda x: x[0])
    print(f"  Projected angles and edge order:")
    for angle, ei in angles_and_edges:
        print(f"    Edge {ei}: {np.degrees(angle):.1f}°")
    
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f"{method_title} Projection: Original Points (Black) vs. Projections (Blue)\nVertex {vertex_idx}")
    
    # Equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax




def visualize_graph_laplacian_method(vertex_idx, ordered_edge_indices, E, V, ax = None):
    """
    Visualize the Graph Laplacian method components
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot all edges in very light gray first
    for ei, (v1, v2) in enumerate(E):
        edge_pts = np.array([V[v1], V[v2]])
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color='gray', alpha=0.2, linewidth=0.5)

    # Plot all vertices in light gray
    ax.scatter(V[:,0], V[:,1], V[:,2], color='lightgray', s=10, alpha=0.3)

    # Highlight the central vertex
    vertex_pos = V[vertex_idx]
    ax.scatter(*vertex_pos, color='red', s=150, label=f'Vertex {vertex_idx}', zorder=10)

    # Define colors for the circulation
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_edge_indices)))
    
    # Plot the sorted edges with colors and labels
    connected_vertices = []
    for i, ei in enumerate(ordered_edge_indices):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        connected_vertices.append(V[other_idx])
        
        edge_pts = np.array([vertex_pos, V[other_idx]])
        
        # Plot the edge with color
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color=colors[i], linewidth=3, alpha=0.8, zorder=5)
        
        # Highlight the connected vertex
        ax.scatter(*V[other_idx], color=colors[i], s=80, zorder=8)
        
        # Add edge index label near the connected vertex
        offset = 0.1 * (V[other_idx] - vertex_pos) / np.linalg.norm(V[other_idx] - vertex_pos)
        label_pos = V[other_idx] + offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], f'{ei}', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f"Edge Circulation around Vertex {vertex_idx} using Graph Laplacian\n"
                f"Order: {' → '.join(map(str, ordered_edge_indices))}", fontsize=14)
    ax.legend()

    vertex_pos = V[vertex_idx]
    incident_edges = [ei for ei in ordered_edge_indices if vertex_idx in E[ei]]
    
    # Calculate edge direction vectors
    edge_directions = []
    for ei in incident_edges:
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        direction = V[other_idx] - vertex_pos
        unit_direction = direction / np.linalg.norm(direction)
        edge_directions.append(unit_direction)

    
    # Calculate and show Hn (average direction)
    Hn = np.average(edge_directions, axis=0)
    Hn_norm = np.linalg.norm(Hn)
    
    if Hn_norm > 1e-5:
        Hn_normalized = Hn / Hn_norm
        
        # Draw Hn vector (blue arrow)
        ax.quiver(vertex_pos[0], vertex_pos[1], vertex_pos[2],
                 Hn_normalized[0] * 0.8, Hn_normalized[1] * 0.8, Hn_normalized[2] * 0.8,
                 color='blue', alpha=0.9, arrow_length_ratio=0.15, linewidth=3)
        
        # Label Hn
        hn_label_pos = vertex_pos + 0.9 * Hn_normalized
        ax.text(hn_label_pos[0], hn_label_pos[1], hn_label_pos[2], 'Hn\n(projection\nnormal)', 
               fontsize=12, color='blue', fontweight='bold', ha='center')
        
        # Draw projection plane
        draw_projection_plane(ax, vertex_pos, Hn_normalized, 'blue', alpha=0.2)
        
        # Add method info
        ax.text2D(0.02, 0.95, f'Hn norm: {Hn_norm:.3f}', transform=ax.transAxes, 
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax.text2D(0.02, 0.95, f'Hn ≈ 0 (saddle point!)', transform=ax.transAxes, 
                 fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    
    # Remove axis for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.tight_layout()
    plt.axis('off')
    plt.axis('equal')
    plt.show()

def draw_projection_plane(ax, center, normal, color, alpha=0.05, size=0.5):
    """
    Draw a projection plane given center point and normal vector
    """
    # Create two perpendicular vectors in the plane
    if abs(normal[0]) < 0.9:
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.array([0, 1, 0])
    
    # Make v1 perpendicular to normal
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    
    # Second perpendicular vector
    v2 = np.cross(normal, v1)
    
    # Create plane mesh
    u = np.linspace(-size, size, 10)
    v = np.linspace(-size, size, 10)
    U, V = np.meshgrid(u, v)
    
    # Plane points
    X = center[0] + U * v1[0] + V * v2[0]
    Y = center[1] + U * v1[1] + V * v2[1]  
    Z = center[2] + U * v1[2] + V * v2[2]
    
    # Plot the plane
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

def visualize_edge_normals_method(vertex_idx, ordered_edge_indices, E, V, one_normals, ax = None):
    """
    Visualize the Edge One Normals method components
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    # Plot all edges in very light gray first
    for ei, (v1, v2) in enumerate(E):
        edge_pts = np.array([V[v1], V[v2]])
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color='gray', alpha=0.2, linewidth=0.5)

    # Plot all vertices in light gray
    ax.scatter(V[:,0], V[:,1], V[:,2], color='lightgray', s=10, alpha=0.3)

    # Highlight the central vertex
    vertex_pos = V[vertex_idx]
    ax.scatter(*vertex_pos, color='red', s=150, label=f'Vertex {vertex_idx}', zorder=10)

    # Define colors for the circulation
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_edge_indices)))
    
    # Plot the sorted edges with colors and labels
    connected_vertices = []
    for i, ei in enumerate(ordered_edge_indices):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        connected_vertices.append(V[other_idx])
        
        edge_pts = np.array([vertex_pos, V[other_idx]])
        
        # Plot the edge with color
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                color=colors[i], linewidth=3, alpha=0.8, zorder=5)
        
        # Highlight the connected vertex
        ax.scatter(*V[other_idx], color=colors[i], s=80, zorder=8)
        
        # Add edge index label near the connected vertex
        offset = 0.1 * (V[other_idx] - vertex_pos) / np.linalg.norm(V[other_idx] - vertex_pos)
        label_pos = V[other_idx] + offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], f'{ei}', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f"Edge Circulation around Vertex {vertex_idx} using One Normal\n"
                f"Order: {' → '.join(map(str, ordered_edge_indices))}", fontsize=14)
    ax.legend()

    vectors = []
    vector_array = np.zeros((len(ordered_edge_indices), 3))
    
    for i, ei in enumerate(ordered_edge_indices):
        vec = one_normals[ei]
        vectors.append((vec, ei))
        vector_array[i] = vec


    N = np.average( vector_array, axis = 0 )

    N_norm = np.linalg.norm(N)
    
    if N_norm > 1e-5:
        N_normalized = N / N_norm
        
        # Draw averaged normal vector (red arrow)
        ax.quiver(vertex_pos[0], vertex_pos[1], vertex_pos[2],
                 N_normalized[0] * 0.9, N_normalized[1] * 0.9, N_normalized[2] * 0.9,
                 color='red', alpha=0.9, arrow_length_ratio=0.15, linewidth=3)
        
        # Label averaged normal
        n_label_pos = vertex_pos + 1.0 * N_normalized
        ax.text(n_label_pos[0], n_label_pos[1], n_label_pos[2], 'N_avg\n(projection\nnormal)', 
               fontsize=12, color='red', fontweight='bold', ha='center')
        
        # Draw projection plane
        draw_projection_plane(ax, vertex_pos, N_normalized, 'red', alpha=0.2)
        
        # Add method info
        ax.text2D(0.02, 0.95, f'N norm: {N_norm:.3f}', transform=ax.transAxes, 
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    else:
        ax.text2D(0.02, 0.95, f'N ≈ 0 (degenerate!)', transform=ax.transAxes, 
                 fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    

     # Remove axis for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.tight_layout()
    plt.axis('off')
    plt.axis('equal')
    plt.show()





# if __name__ == "__main__":

     
#     parser = argparse.ArgumentParser(description='Optimize edges to get normals')
#     parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')

#     args = parser.parse_args()

#     curve_file = args.curve_file

#     if curve_file is None:
#         curve_file = 'sketch/onshape_simple_mouse.obj'
#     V, E, P = load_sketch_polyline_data(curve_file)
    
#     vertex_to_edges_map = build_vertex_to_edges_map(E)
#     print('Loaded vertex_to_edges_map:', vertex_to_edges_map)
    
#     vertices_to_check = [idx for idx, edges in vertex_to_edges_map.items() if len(edges) >= 3]
#     print('vertices_to_check', vertices_to_check)

#     for vertex_idx in vertices_to_check:
#         print(f"\nVisualizing vertex {vertex_idx} with {len(vertex_to_edges_map[vertex_idx])} edges.")
#         edge_indices = vertex_to_edges_map[vertex_idx]
#         sorted_edges = compute_edge_circulation_graph_laplacian(edge_indices, vertex_idx, E, V)
#         print(f"  Sorted edge indices: {sorted_edges}")
#         print(f"  Original edges:")
#         for ei in edge_indices:
#             print(f"    {ei}: {E[ei]}")
            
#         # First plot: show the sorted edges
#         fig1 = plt.figure(figsize=(8, 8))
#         ax1 = fig1.add_subplot(111, projection='3d')
#         # plot_sorted_edges_with_circulation(vertex_idx, sorted_edges, E, V, ax=ax1)
#         # plot_sorted_edges_with_circulation(vertex_idx, sorted_edges, E, V)
#         visualize_graph_laplacian_method(vertex_idx, sorted_edges,E, V, ax1)
#         plt.tight_layout()
#         plt.show()
        
#         # # Second plot: visualize the projection plane and points
#         # fig2, ax2 = plot_projection_plane_and_points(vertex_idx, edge_indices, E, V)
#         # plt.show()
        
#         # Optional: Add a prompt to continue to the next vertex
#         if vertex_idx != vertices_to_check[-1]:  # If not the last vertex
#             input("Press Enter to continue to the next vertex...")


import numpy as np
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

def plot_sorted_edges(vertex_idx, ordered_edge_indices, E, V, ax=None):
    """
    Plot the ordered edges around a vertex in 3D, labeling each edge with its edge index.
    """
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    ax.scatter(V[:,0], V[:,1], V[:,2])

    vertex_pos = V[vertex_idx]
    ax.scatter(*vertex_pos, color='red', s=100, label=f'Central vertex {vertex_idx}')

    for ei in ordered_edge_indices:
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        edge_pts = np.array([vertex_pos, V[other_idx]])
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], linewidth=2)
        ax.scatter(*V[other_idx], color='blue', s=60)

        # Annotate with actual edge index
        mid = (vertex_pos + V[other_idx]) / 2
        ax.text(*mid, f'{ei}', fontsize=10, color='black')

    ax.set_title(f"Circulation around vertex {vertex_idx}")
    ax.legend()
    
    plt.axis('off')
    plt.axis('equal')
    plt.show()

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

def plot_projection_plane_and_points(vertex_idx, edge_indices, E, V):
    """
    Visualize the projection plane, the original 3D points (vertices), and their projections.
    Original vertices in black, projections in blue.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    vertex_pos = V[vertex_idx]
    ax.scatter(V[:,0], V[:,1], V[:,2])

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
            
            # Also compute vector (for SVD)
            vec = other_pos - vertex_pos
            unit_vec = vec / np.linalg.norm(vec)
            vectors.append((unit_vec, ei))
            vector_array.append(unit_vec)
    
    vector_array = np.array(vector_array)
    
    # Plot the central vertex
    ax.scatter(*vertex_pos, color='red', s=100, label=f'Vertex {vertex_idx}')
    
    # Compute projection plane using SVD
    centered_data = vector_array - np.mean(vector_array, axis=0)
    _, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    basis_1, basis_2, basis_3 = Vt[0], Vt[1], Vt[2]
    
    # Print variance information
    total_variance = np.sum(S**2)
    variance_explained = (S[0]**2 + S[1]**2) / total_variance
    print(f"Variance explained by projection plane: {variance_explained:.4f}")
    
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
    
    # Basis 1 (1st principal component)
    ax.quiver(center[0], center[1], center[2], 
              basis_1[0]*scale_basis, basis_1[1]*scale_basis, basis_1[2]*scale_basis, 
              color='r', arrow_length_ratio=0.1, label='Basis 1')
    
    # Basis 2 (2nd principal component)
    ax.quiver(center[0], center[1], center[2], 
              basis_2[0]*scale_basis, basis_2[1]*scale_basis, basis_2[2]*scale_basis, 
              color='g', arrow_length_ratio=0.1, label='Basis 2')
    
    # Basis 3 (normal to plane)
    ax.quiver(center[0], center[1], center[2], 
              basis_3[0]*scale_basis, basis_3[1]*scale_basis, basis_3[2]*scale_basis, 
              color='b', arrow_length_ratio=0.1, label='Normal')
    
    # Plot lines from center to original points
    for point, ei in edge_endpoints:
        # Draw line from center to original point
        ax.plot([vertex_pos[0], point[0]], 
                [vertex_pos[1], point[1]], 
                [vertex_pos[2], point[2]], 
                'k-', linewidth=1.5)
        
        # Plot original point (black)
        ax.scatter(*point, color='black', s=60, label=f'Original Point {ei}' if ei == incident_edges[0] else "")
        
        # Add edge index label at the point
        ax.text(*point, f'e{ei}', fontsize=10, color='black')
        
        # Compute projection of this point onto the plane
        vec = point - vertex_pos
        vec_centered = vec - np.mean(vector_array * np.linalg.norm(vec), axis=0)
        
        # Project onto the plane
        proj_1 = np.dot(vec_centered, basis_1)
        proj_2 = np.dot(vec_centered, basis_2)
        
        # Compute the projected point
        proj_point = center + proj_1 * basis_1 + proj_2 * basis_2
        
        # Plot projected point (blue)
        ax.scatter(*proj_point, color='blue', s=60, 
                   label=f'Projected Point {ei}' if ei == incident_edges[0] else "")
        
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
        ax.text(*proj_point, f'\n{angle_deg:.1f}°', fontsize=8, color='blue')
    
    # Create a custom legend to avoid duplicating entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
    
    ax.set_title(f"Original Points (Black) vs. Projections (Blue) for Vertex {vertex_idx}")
    
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
    
    plt.axis('off')
    plt.axis('equal')

    plt.tight_layout()
    return fig, ax

# Example usage
# Example usage
if __name__ == "__main__":
    curve_file = 'sketches/onshape/onshape_simple_mouse.obj'
    V, E, P = load_sketch_polyline_data(curve_file)
    
    vertex_to_edges_map = build_vertex_to_edges_map(E)
    print('Loaded vertex_to_edges_map:', vertex_to_edges_map)
    
    vertices_to_check = [idx for idx, edges in vertex_to_edges_map.items() if len(edges) >= 4]

    for vertex_idx in vertices_to_check:
        print(f"\nVisualizing vertex {vertex_idx} with {len(vertex_to_edges_map[vertex_idx])} edges.")
        edge_indices = vertex_to_edges_map[vertex_idx]
        sorted_edges = compute_edge_circulation_graph_laplacian(edge_indices, vertex_idx, E, V)
        print(f"  Sorted edge indices: {sorted_edges}")
        print(f"  Original edges:")
        for ei in edge_indices:
            print(f"    {ei}: {E[ei]}")
            
        # First plot: show the sorted edges
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        plot_sorted_edges(vertex_idx, sorted_edges, E, V, ax=ax1)
        plt.tight_layout()
        plt.show()
        
        # Second plot: visualize the projection plane and points
        fig2, ax2 = plot_projection_plane_and_points(vertex_idx, edge_indices, E, V)
        plt.show()
        
        # Optional: Add a prompt to continue to the next vertex
        if vertex_idx != vertices_to_check[-1]:  # If not the last vertex
            input("Press Enter to continue to the next vertex...")


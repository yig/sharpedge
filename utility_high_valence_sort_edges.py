import numpy as np

from opt_edges import * 

def compute_edge_circulation(edges, vertex_idx, V):
    """
    Compute the circulation ordering of edges around a vertex.
    
    Parameters:
    - edges: List of edges, where each edge is a pair of vertex indices [(i1, i2), ...]
    - vertex_idx: The index of the vertex around which to find the circulation
    - V: Array of vertex positions, where V[i] gives the 3D coordinates of vertex i
    
    Returns:
    - ordered_edges: The edges in circulation order
    """
    # Get the position of the central vertex
    vertex_pos = V[vertex_idx]
    
    # Filter edges incident to the vertex
    incident_edges = [edge for edge in edges if vertex_idx in edge]
    
    if len(incident_edges) <= 2:
        return incident_edges  # No meaningful circulation for 0, 1, or 2 edges
    
    # Create unit vectors pointing from the vertex to each adjacent vertex
    vectors = []
    vector_array = np.zeros((len(incident_edges), 3))
    
    for i, edge in enumerate(incident_edges):
        # Get the other vertex index of the edge
        other_idx = edge[1] if edge[0] == vertex_idx else edge[0]
        # Create vector from vertex to other_vertex
        vec = V[other_idx] - vertex_pos
        # Normalize to unit vector
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, edge))
        vector_array[i] = unit_vec
    
    # Center the data (subtract mean)
    centered_data = vector_array - np.mean(vector_array, axis=0)
    
    # Compute SVD
    # V contains the right singular vectors, which are equivalent to PCA components
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    # The first two right singular vectors define the plane
    basis_1 = Vt[0]
    basis_2 = Vt[1]
    
    # Project vectors onto this basis and calculate angles
    projected_vectors = []
    for unit_vec, edge in vectors:
        # Project the vector onto the two basis vectors
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        # Angle in the plane
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, edge))
    
    # Sort by angle
    projected_vectors.sort(key=lambda x: x[0])
    
    # Return the edges in order
    ordered_edges = [edge for _, edge in projected_vectors]
    
    return ordered_edges




if __name__ == "__main__":
    curve_file = 'sketches/onshape/onshape_simple_mouse.obj'
    V, E, P = load_sketch_polyline_data(curve_file)

    vertex_to_edges_map = build_vertex_to_edges_map( E )
    print('vertex_to_edges', vertex_to_edges_map)

    for index, edge_indices in vertex_to_edges_map.items():
        if len(edge_indices) >= 4:
            print(index, edge_indices)
            edges = [ E[edge_idx] for edge_idx in edge_indices ]
            
            ordered_edges = compute_edge_circulation(edges, index, V)
            print(ordered_edges)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from opt_edges import load_sketch_polyline_data, build_vertex_to_edges_map

def compute_edge_circulation(edge_indices, vertex_idx, E, V):
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

def plot_sorted_edges(vertex_idx, ordered_edge_indices, E, V, ax=None):
    """
    Plot the ordered edges around a vertex in 3D, labeling each edge with its edge index.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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
        sorted_edges = compute_edge_circulation(edge_indices, vertex_idx, E, V)
        print(f"  Sorted edge indices: {sorted_edges}")
        print(f"  Original edges:")
        for ei in edge_indices:
            print(f"    {ei}: {E[ei]}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_sorted_edges(vertex_idx, sorted_edges, E, V, ax=ax)
        plt.tight_layout()
        plt.show()

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

def compute_geodesic_voronoi_areas(V, F, sample_points):
    """
    Compute geodesic Voronoi areas for sample points on a surface mesh.

    Parameters:
    V : np.ndarray (N, 3)
        Vertex positions of the mesh.
    F : np.ndarray (M, 3)
        Triangular faces (indices into V).
    sample_points : np.ndarray (P, 3)
        Sample points (assumed to be near the surface).

    Returns:
    np.ndarray (P,)
        Estimated Voronoi area for each sample point.
    """

    # Step 1: Construct mesh adjacency graph with edge lengths
    num_vertices = len(V)
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])  # Extract edges
    edges = np.unique(np.sort(edges, axis=1), axis=0)  # Remove duplicates

    # Compute Euclidean edge lengths
    edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)

    # Create sparse adjacency matrix (weighted by edge lengths)
    row, col = edges[:, 0], edges[:, 1]
    adjacency_matrix = csr_matrix((edge_lengths, (row, col)), shape=(num_vertices, num_vertices))

    # Make it symmetric (undirected graph)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T

    # Step 2: Find closest mesh vertices to sample points
    kdtree = KDTree(V)
    _, closest_vertices = kdtree.query(sample_points)  # Get nearest vertex indices

    # Step 3: Compute geodesic distances from sample points to all mesh vertices
    geodesic_distances = dijkstra(adjacency_matrix, directed=False, indices=closest_vertices)

    # Step 4: Assign each face to the closest sample point
    face_areas = trimesh.Trimesh(vertices=V, faces=F, process=False).area_faces
    voronoi_areas = np.zeros(len(sample_points))

    for f_idx, face in enumerate(F):
        v1, v2, v3 = face
        dists = geodesic_distances[:, [v1, v2, v3]].min(axis=1)  # Min distance to the three vertices
        owner = np.argmin(dists)  # Closest sample point
        voronoi_areas[owner] += face_areas[f_idx]  # Assign area to the owner

    return voronoi_areas

# --- TEST CASE ---
# Generate a test sphere
sphere = trimesh.creation.icosphere(subdivisions=3)
V = sphere.vertices
F = sphere.faces

# Select 50 random points
np.random.seed(42)
sample_indices = np.random.choice(len(V), size=50, replace=False)
sample_points = V[sample_indices]

# Compute Voronoi areas
areas = compute_geodesic_voronoi_areas(V, F, sample_points)

# Output results
print("Sum of Voronoi areas (should be close to 4π for a unit sphere):", np.sum(areas))
print("First 10 areas:", areas[:10])

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
import polyscope as ps
import matplotlib.cm as cm

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
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    
    # Compute Euclidean edge lengths
    edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    
    # Create sparse adjacency matrix (weighted by edge lengths)
    row, col = edges[:, 0], edges[:, 1]
    adjacency_matrix = csr_matrix((edge_lengths, (row, col)), shape=(num_vertices, num_vertices))
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    
    # Step 2: Find closest mesh vertices to sample points
    kdtree = KDTree(V)
    _, closest_vertices = kdtree.query(sample_points)
    
    # Step 3: Compute geodesic distances from sample points to all mesh vertices
    geodesic_distances = dijkstra(adjacency_matrix, directed=False, indices=closest_vertices)
    
    # Step 4: Assign each face to the closest sample point
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    face_areas = mesh.area_faces
    voronoi_areas = np.zeros(len(sample_points))
    
    # Create array to track face ownership
    face_owners = np.zeros(len(F), dtype=int)
    
    for f_idx, face in enumerate(F):
        v1, v2, v3 = face
        dists = geodesic_distances[:, [v1, v2, v3]].min(axis=1)
        owner = np.argmin(dists)
        face_owners[f_idx] = owner
        voronoi_areas[owner] += face_areas[f_idx]
    
    return voronoi_areas, geodesic_distances, mesh, closest_vertices, face_owners

def visualize_with_polyscope(V, F, sample_points, face_owners, voronoi_areas, show_boundaries = False):
    """
    Visualize geodesic Voronoi diagram using Polyscope with matching colors.
    
    Parameters:
    V : np.ndarray
        Mesh vertices
    F : np.ndarray
        Mesh faces
    sample_points : np.ndarray
        Sample points
    face_owners : np.ndarray
        Array indicating which sample point "owns" each face
    voronoi_areas : np.ndarray
        Areas of Voronoi cells
    show_boundaries : bool, optional
        Whether to compute and display region boundaries (default: False)
    """
    # Initialize polyscope
    ps.init()
    
    # Register the mesh
    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    
    # Define a custom colormap for consistent colors
    num_regions = len(sample_points)
    
    # Generate distinct colors for each region using tab20 colormap
    cmap = cm.tab20
    region_colors = np.zeros((num_regions, 3))
    for i in range(num_regions):
        # Use the cmap directly to get colors - modulo 20 to stay within tab20's range
        region_colors[i] = cmap(i % 20)[:3]
    
    # Apply custom colors to the mesh faces
    face_colors = np.zeros((len(F), 3))
    for i in range(len(F)):
        face_colors[i] = region_colors[face_owners[i]]
    
    # Add face colors to the mesh
    ps_mesh.add_color_quantity("region colors", face_colors, defined_on='faces', enabled=True)
    
    # Register sample points as a point cloud
    ps_points = ps.register_point_cloud("sample points", sample_points)
    # ps_points.set_radius(0.02)  # Adjust point size
    ps_points.set_radius(0.005)
    
    # Color the sample points with the same colors as their regions
    point_colors = np.zeros((len(sample_points), 3))
    for i in range(len(sample_points)):
        point_colors[i] = region_colors[i]
    
    ps_points.add_color_quantity("point colors", point_colors, enabled=True)
    
    # Add area values to the point cloud
    area_normalized = voronoi_areas / np.max(voronoi_areas)  # Normalize for better visualization
    ps_points.add_scalar_quantity("voronoi area", voronoi_areas, enabled=False)
    
    # Scale point size by Voronoi area
    sizes = 0.01 + 0.03 * area_normalized  # Min size 0.01, max additional size 0.03
    # ps_points.set_radius(sizes)

    # mesh_points = ps.register_point_cloud('mesh points', V)
    # mesh_points.set_radius(0.005)
    # mesh_points.set_color((0, 0, 0))

    
    # Compute and visualize Voronoi boundaries
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    unique_edges, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    
    if show_boundaries:
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        unique_edges, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        
        # Find boundaries between regions
        boundary_edges = []
        for e in unique_edges:
            v1, v2 = e
            # Find all faces that contain this edge
            containing_faces = []
            for f_idx, face in enumerate(F):
                if (v1 in face and v2 in face):
                    containing_faces.append(f_idx)
            
            # If this edge belongs to faces with different owners, it's a boundary
            if len(containing_faces) >= 2 and face_owners[containing_faces[0]] != face_owners[containing_faces[1]]:
                boundary_edges.append([v1, v2])
        
        # If we found boundaries, register them as a curve network
        if boundary_edges:
            boundary_edges = np.array(boundary_edges)
            ps_boundaries = ps.register_curve_network("region boundaries", 
                                                    V, 
                                                    boundary_edges, 
                                                    color=(0, 0, 0),
                                                    radius=0.003)
    
    ps.set_ground_plane_mode('none')

    # Show the polyscope GUI
    ps.show()

# Example usage
if __name__ == "__main__":
    # Load a simple mesh (e.g., a sphere)
    mesh = trimesh.creation.icosphere(subdivisions=3)  # Higher subdivision for better visualization
    
    # Create some random sample points on the surface
    np.random.seed(42)  # For reproducible results
    num_samples = 12
    random_faces = np.random.choice(len(mesh.faces), num_samples)
    sample_points = np.array([mesh.triangles_center[face] for face in random_faces])
    
    # Compute Voronoi areas and face ownership
    voronoi_areas, geodesic_distances, mesh, closest_vertices, face_owners = compute_geodesic_voronoi_areas(
        mesh.vertices, mesh.faces, sample_points)
    
    print("Voronoi areas:", voronoi_areas)
    
    # Visualize with Polyscope
    visualize_with_polyscope(mesh.vertices, mesh.faces, sample_points, face_owners, voronoi_areas)
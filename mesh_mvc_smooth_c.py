import igl
import numpy as np
import polyscope as ps
import scipy.sparse as sp

def compute_laplacian_magnitude_paper(V, F, adjacency_list):
    """
    c_i = ||x_i - (1/|N_i|) * Σ x_j||
    """
    n = len(V)
    c = np.zeros(n)
    for i in range(n):
        neighbors = adjacency_list[i]
        if len(neighbors) == 0:
            continue
        neighbor_mean = np.mean(V[neighbors], axis=0)
        c[i] = np.linalg.norm(V[i] - neighbor_mean)

    # Step 2: average Laplacian magnitude
    c_avg = np.zeros_like(c)
    for i in range(len(c)):
        neighbors = adjacency_list[i]
        if len(neighbors) == 0:
            c_avg[i] = c[i]
            continue
        c_avg[i] = np.mean(c[neighbors])

    return c

def compute_laplacian_magnitude_igl(v, f, adjacency_list):
    """
    libigl
    """
    l = igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    
    m_diag = m.diagonal()
    m_diag_safe = np.maximum(m_diag, 1e-10)
    minv = sp.diags(1.0 / m_diag_safe)
        
    laplacian_vector = -minv.dot(l.dot(v))  
    
    c = np.linalg.norm(laplacian_vector, axis=1)


    # Step 2: average Laplacian magnitude
    c_avg = np.zeros_like(c)
    for i in range(len(c)):
        neighbors = adjacency_list[i]
        if len(neighbors) == 0:
            c_avg[i] = c[i]
            continue
        c_avg[i] = np.mean(c[neighbors])

    
    return c

def compute_laplacian_magnitude_average(v, f):
    l = igl.cotmatrix(v, f)

    l_diagonal = np.abs(l.diagonal()) 
    minv = sp.diags(1.0 / np.maximum(l_diagonal, 1e-10))

    hn = -minv.dot(l.dot(v))
    c = np.linalg.norm(hn, axis=1)

    A = minv @ l  
    A.setdiag(0)  

    row_sums = np.array(A.sum(axis=1)).flatten()  
    A_normalized = sp.diags(1.0 / row_sums) @ A   


    c_averaged = A_normalized @ c  

    return c_averaged

def read_two_normal(filename):
    """
    Read vertices, edges, and dual normal data from an OBJ file.
    Each edge has exactly two normal vectors.
    
    Args:
        filename (str): Path to input OBJ file
        
    Returns:
        V (ndarray): nx3 array of vertex coordinates (x, y, z)
        E (ndarray): mx2 array of edge vertex pairs (0-based indices)
        normals (dict): Dictionary with keys (edge_idx, which_edge) where:
            - edge_idx is the edge index
            - which_edge is 0 or 1 for the first or second normal
            Values are 3D normal vectors (nx, ny, nz)
    """
    V = []  # Vertices
    E = []  # Edges
    normals = {}  # Dictionary to store normals
    normal_vectors = []  # Temporary list to store normal vectors
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'l':  # Edge
                # Convert from 1-based to 0-based indexing
                E.append([int(parts[1])-1, int(parts[2])-1])
            elif parts[0] == 'vn':  # Normal vector
                normal_vectors.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    # Associate normals with edges
    for i, _ in enumerate(E):
        if 2*i < len(normal_vectors) and 2*i+1 < len(normal_vectors):
            normals[(i, 0)] = normal_vectors[2*i]
            normals[(i, 1)] = normal_vectors[2*i + 1]
    
    # Convert lists to numpy arrays
    V = np.array(V)
    E = np.array(E)
    
    print(f"Read from {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {len(normal_vectors)} normal vectors")
    
    return V, E, normals

def resample_edge_dual_normal(V, E, normals, target_edge_length=0.05):
    """
    Resample edge dual normal geometry to achieve target edge length.
    
    Args:
        V (ndarray): nx3 array of vertex coordinates
        E (ndarray): mx2 array of edge vertex pairs (0-based indices) 
        normals (dict): Dictionary with dual normals for each edge
        target_edge_length (float): Target length for each edge segment
        
    Returns:
        new_V (ndarray): resampled vertices
        new_E (ndarray): resampled edges
        new_normals (dict): resampled normals dictionary
    """
    
    if target_edge_length <= 0:
        raise ValueError("Target edge length must be positive")
    
    if len(E) == 0 or len(V) == 0:
        raise ValueError("Source geometry is empty")
    
    print(f"Resampling geometry to target edge length: {target_edge_length}")
    print(f"Original: {len(V)} vertices, {len(E)} edges")
    
    # Start with original vertices
    new_vertices = V.tolist()
    new_edges = []
    new_normals = {}
    
    def calculate_distance(v1, v2):
        """Calculate Euclidean distance between two vertices"""
        return np.linalg.norm(v2 - v1)
    
    for edge_idx, edge in enumerate(E):
        start_vertex_idx = edge[0]
        end_vertex_idx = edge[1]
        
        start_vertex = V[start_vertex_idx]
        end_vertex = V[end_vertex_idx]
        
        # Get the dual normals for this edge
        normal1 = np.array(normals.get((edge_idx, 0), [0, 0, 1]))
        normal2 = np.array(normals.get((edge_idx, 1), [0, 0, 1]))
        
        # Calculate current edge length
        current_edge_length = calculate_distance(start_vertex, end_vertex)
        
        if current_edge_length < 1e-6:
            print(f"Warning: Skipping degenerate edge {edge_idx} (length: {current_edge_length})")
            continue
        
        # Calculate how many segments we need
        num_segments = max(1, int(np.ceil(current_edge_length / target_edge_length)))
        
        # Create subdivided segments
        current_vertex_idx = start_vertex_idx
        
        for i in range(num_segments):
            if i == num_segments - 1:
                # Last segment connects to the original end vertex
                next_vertex_idx = end_vertex_idx
            else:
                # Create intermediate vertex
                t = (i + 1) / num_segments
                interpolated_vertex = start_vertex + t * (end_vertex - start_vertex)
                
                new_vertices.append(interpolated_vertex)
                next_vertex_idx = len(new_vertices) - 1
            
            # Add the new edge segment
            new_edges.append([current_vertex_idx, next_vertex_idx])
            
            # Copy normals to all segments of this edge
            new_edge_idx = len(new_edges) - 1
            new_normals[(new_edge_idx, 0)] = normal1
            new_normals[(new_edge_idx, 1)] = normal2
            
            current_vertex_idx = next_vertex_idx
    
    # Convert results back to numpy arrays
    new_V = np.array(new_vertices)
    new_E = np.array(new_edges)
    
    print(f"Resampling complete:")
    print(f"- New vertices: {len(new_V)} (added {len(new_V) - len(V)})")
    print(f"- New edges: {len(new_E)} (was {len(E)})")
    if len(E) > 0:
        print(f"- Average segments per original edge: {len(new_E) / len(E):.1f}")
    
    return new_V, new_E, new_normals

def find_constraint_vertices_in_mesh(mesh_vertices, constraint_vertices, tolerance=1e-6):
    """
    Find which mesh vertices correspond to constraint vertices
    
    Returns:
        constraint_indices: list of mesh vertex indices that are constrained
    """
    from scipy.spatial import cKDTree
    
    mesh_tree = cKDTree(mesh_vertices)
    constraint_indices = []
    
    print(f"Finding constraint vertices in mesh (tolerance: {tolerance})...")
    
    distances = []
    for i, constraint_vertex in enumerate(constraint_vertices):
        distance, closest_idx = mesh_tree.query(constraint_vertex)
        distances.append(distance)
        
        if distance <= tolerance:
            constraint_indices.append(closest_idx)
        else:
            print(f"Warning: Constraint vertex {i} not found in mesh (distance: {distance:.6f})")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_constraint_indices = []
    for idx in constraint_indices:
        if idx not in seen:
            seen.add(idx)
            unique_constraint_indices.append(idx)
    
    # Statistics
    if distances:
        print(f"Distance statistics:")
        print(f"  Max distance: {max(distances):.6f}")
        print(f"  Mean distance: {np.mean(distances):.6f}")
        print(f"  Median distance: {np.median(distances):.6f}")
    
    print(f"Found {len(unique_constraint_indices)} unique constraint vertices in mesh")
    print(f"Coverage: {len(unique_constraint_indices)}/{len(constraint_vertices)} "
          f"({100*len(unique_constraint_indices)/len(constraint_vertices):.1f}%)")
    
    return unique_constraint_indices


def smooth_laplacian_magnitude(V, F, adjacency_list, iterations=4, threshold=1e-4, method = 'avg', constraint_indices = None):
    V_new = V.copy()

    for it in range(iterations):
        # Step 1: compute Laplacian magnitude
        if method == 'paper':
            c = compute_laplacian_magnitude_paper(V_new, F, adjacency_list)
        elif method == 'igl':
            c = compute_laplacian_magnitude_igl(V_new, F, adjacency_list)
        else:
            c = compute_laplacian_magnitude_average(V_new, F)
    

        plot(V_new, f, c)




        # Step 3: compute vertex normals using current V_new
        vertex_normals = igl.per_vertex_normals(V_new, F)
        # Actuallty I am not sure 
        # plot_normal(V_new, F, vertex_normals)
        
        # Step 4: update vertex positions along normal direction
        displacement = np.zeros_like(V_new)
        max_disp = 0.0

        for i in range(len(V_new)):
            
            if i in constraint_indices:
                continue
            else:
                direction = vertex_normals[i]
                d = c[i] * direction
                displacement[i] = d
                max_disp = max(max_disp, np.linalg.norm(d))

        
        print(f"[iter {it+1}] max displacement: {max_disp:.6f}")
        if max_disp < threshold:
            break
        V_new += displacement


    return V_new

def plot(v, f, k, title="Curvature"):
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.add_scalar_quantity(title, k, defined_on='vertices', cmap='coolwarm', enabled=True)
    ps_mesh.set_edge_width(1)
    ps.set_ground_plane_mode('none')
    ps.show()

def plot_normal(v, f, n):
    '''
    '''
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.add_vector_quantity("vertex normals", n, defined_on="vertices", enabled=True)
    ps.set_ground_plane_mode('none')
    ps.show()
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Laplacian magnitude smoothing based on FiberMesh')
    parser.add_argument('normal_file', help='Normal .normal file')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    args = parser.parse_args()

    mesh_file = args.mesh_file


    v, f = igl.read_triangle_mesh(mesh_file)
    adj = igl.adjacency_list(f)


    
    original_vertices, original_edges, original_normals = read_two_normal(args.normal_file)
    sketch_vertices, sketch_edges, normals = resample_edge_dual_normal(
        original_vertices, original_edges, original_normals, target_edge_length = 0.05)
    constraint_indices = find_constraint_vertices_in_mesh(
            v, sketch_vertices, tolerance=1e-6)
        


    # 执行 FiberMesh smoothing
    v_smoothed = smooth_laplacian_magnitude(v, f, adj, iterations=5, threshold=1e-5, method='avg', constraint_indices = constraint_indices)

    # 平滑后的 curvature
    # c_smooth = compute_laplacian_magnitude(v_smoothed, adj)
    # plot(v_smoothed, f, c_smooth, title="Smoothed Laplacian magnitude")

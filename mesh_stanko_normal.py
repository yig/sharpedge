import igl
import numpy as np
import scipy as sp
import polyscope as ps
from pprint import pprint
from collections import defaultdict

def solve_system_with_constraints_hard( A, b, constraint_indices, constraint_values ):
    '''
    Given:
        A: A system matrix
        b: A right hand side
        constraint_indices: A sequence of indices to constrain in x
        constraint_values: A sequence of values to constrain x to corresponding to `constraint_indices`
    Returns:
        The solution x to Ax=b such that x[ constraint_indices ] = constraint_values.
    '''

    ## Outline:
    '''
    If solving a system of equations:
    Ax=b
    Assume A is symmetric. Let A_*,k be the column corresponding to a fixed value x_k=b_k. If we update the matrix to zero row k and place 1 at A_k,k, we can keep A symmetric by also zeroing column k. Column k are the coefficients of the known value that should be added to each entry to equal b. We can update the right hand side to subtract the coefficients times the known values.

    b' = b - x_k A_*,k

    For many constraints, we can write:

    b' = b - Ar, where r_k = x_k if x_k is constrained and 0 otherwise.
    Then assign b'_k = r_k for constraints k, zero A's k-th row and column, and set A_k,k = 1.
    '''

    r = np.zeros_like(b)
    r[ constraint_indices ] = constraint_values

    b_prime = b - A @ r
    b_prime[ constraint_indices ] = r[ constraint_indices ]

    A = A.copy()
    A[:,constraint_indices] = 0
    A[constraint_indices,:] = 0
    A[constraint_indices,constraint_indices] = 1.0

    x = sp.sparse.linalg.spsolve( A, b_prime )
    assert np.abs( x[constraint_indices] - constraint_values ).max() < 1e-9
    return x

def cotmatrix( V, F ):
    # return igl.cotmatrix(V, F)
    l = igl.edge_lengths(V,F)
    l_intrinsic, F_intrinsic = igl.intrinsic_delaunay_triangulation(l,F)[:2]

    ## Mollify the intrinsic lengths to avoid numerical issues [Sharp and Crane 2020].
    # Every non-boundary edge is in l_intrinsic twice. Taking the mean double-counts non-boundary edges.
    # That won't matter for a closed mesh, but will slightly undercount boundary edges for a mesh with boundary.
    delta = 1e-4 * np.mean(l_intrinsic)
    
    '''
    eps2 = 0.
    for T in l_intrinsic:
        for i in range(3):
            eps2 = np.maximum( eps2, np.maximum( 0, delta - T[i] - T[(i+1)%3] + T[(i+2)%3] ) )
    '''

    eps = 0.
    for i in range(3):
        eps = np.maximum( eps, np.maximum( 0, delta - l_intrinsic[:,i] - l_intrinsic[:,(i+1)%3] + l_intrinsic[:,(i+2)%3] ).max() )
    
    # assert eps == eps2

    l_intrinsic += eps

    L = igl.cotmatrix_intrinsic(l_intrinsic, F_intrinsic)
    return L

def compute_curvature_stanko( V, F, N, method = 'EG' ):
    Hn = np.zeros( N.shape )
    ## Use the method from Stanko et al. 2016 ("C&G" or "EG" short paper)
    if method == 'C&G':
        for i,j,k in F:
            n = N[i] + N[j] + N[k]
            n /= np.linalg.norm(n)

            Hn[i] += np.cross( n, V[k] - V[j] )
            Hn[j] += np.cross( n, V[i] - V[k] )
            Hn[k] += np.cross( n, V[j] - V[i] )
    elif method == 'EG':
        for i,j,k in F:
            n_jk = N[j] + N[k]
            n_ij = N[i] + N[j]
            n_ki = N[k] + N[i]
            n_jk /= np.linalg.norm(n_jk)
            n_ij /= np.linalg.norm(n_ij)
            n_ki /= np.linalg.norm(n_ki)
            
            Hn[i] += np.cross( n_jk, V[k] - V[j] )
            Hn[j] += np.cross( n_ij, V[i] - V[k] )
            Hn[k] += np.cross( n_ki, V[j] - V[i] )
    else:
        raise NotImplementedError(f"Unknown curvature computation method: {method}")

    # These H are scaled by the mass (voronoi area) of each vertex
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())

    # Divide by 4 because this computes 4Hn
    Hn = .25 * Minv @ Hn
    H = np.linalg.norm(Hn, axis=1) * np.sign( ( Hn * N ).sum(axis=1) )

    # Return Hn and H
    return Hn, H

def smooth_stanko(V, F, N = None, constrained = None, target_curvature_paper = True, curvature_method = 'EG' ):
    print( "smooth_stanko() called with parameters:" )
    pprint( locals() )

    L = cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())
    
    # Step 1: Smooth normals
    if N is None: N = igl.per_vertex_normals(V, F)

    plot_constraint_normals(v, f, constrained, N)
    
    # A = (L.T@Minv@L).tocsc()
    A = (L @ Minv @ L @ Minv @ L).tocsc()  

    B = np.zeros(N.shape)
    N_star = solve_system_with_constraints_hard( A, B, constrained, N[constrained] )
    # Normalize the computed normals
    N_star /= np.linalg.norm( N_star, axis = 1 )[:,None]

    # Step 2: Smooth positions
    V_star = solve_system_with_constraints_hard( A, np.zeros(V.shape), constrained, V[constrained] )

    # Step 3: Compute target curvatures
    Hn, H = compute_curvature_stanko( V_star, F, N_star, method = curvature_method )
    # plot(V, F, H)

    if target_curvature_paper:
        target_LM = -2 * ( L.T @ (np.abs(H[:,None]) * N_star) )
    else:
        target_LM = -2 * ( L.T @ Hn )
    V_new = solve_system_with_constraints_hard( A, target_LM, constrained, V[constrained] )

    max_disp = np.max( np.linalg.norm( np.abs( V_new - V ), axis = 1 ) )
    V = V_new
    
    print(f"max displacement: {max_disp:.6f}")

    plot(V, F, H)

    return V


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

def plot_constraint_normals(v, f, constrained, constrained_normals):
    '''
    '''
    ps.init()

    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.set_edge_width(1)

    # extract constrained vertices and normals
    cons_v = v[constrained]
    cons_n = constrained_normals[constrained]

    # add point cloud for constrained vertices
    ps_points = ps.register_point_cloud("constrained_vertices", cons_v, radius=0.005, color=(1.0, 0.0, 0.0))

    # add vector quantity for constrained normals
    ps_points.add_vector_quantity("constraint_normals", cons_n, length = 0.1, enabled=True, color=(0.0, 1.0, 0.0))

    ps.set_ground_plane_mode("none")
    ps.show()

def extract_boundary_info(faces):
    """
    Extract boundary vertices, boundary edges, and edge-to-face map.
    
    Args:
        faces: (n_faces, 3) ndarray or list of faces
    
    Returns:
        boundary_vertices: sorted list of vertex indices
        boundary_edges: list of (v1, v2)
        edge_to_face: dict {edge: face_index}
    """
    edge_faces = defaultdict(list)

    # 遍历每个三角形，统计它的三条边属于哪些面
    for f_idx, face in enumerate(faces):
        for i in range(3):
            v1, v2 = face[i], face[(i+1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edge_faces[edge].append(f_idx)

    boundary_vertices = set()
    boundary_edges = []
    edge_to_face = {}

    for edge, f_list in edge_faces.items():
        if len(f_list) == 1:  # 边界边
            boundary_edges.append(edge)
            edge_to_face[edge] = f_list[0]
            boundary_vertices.update(edge)

    return sorted(list(boundary_vertices)), boundary_edges, edge_to_face


def map_boundary_edges_to_sketch_edges(mesh_vertices, boundary_edges, sketch_vertices, sketch_edges, tol=1e-5):
    """
    Map boundary edges to sketch edges.
    
    Parameters:
        mesh_vertices: (n_mesh_vertices, 3) array of mesh vertex coordinates
        boundary_edges: (n_boundary_edges, 2) array or list of boundary edge vertex pairs
        sketch_vertices: (n_sketch_vertices, 3) array of sketch vertex coordinates  
        sketch_edges: (n_sketch_edges, 2) array of sketch edge vertex pairs
        tol: tolerance for geometric matching
    
    Returns:
        boundary_to_sketch: dict {boundary_edge: sketch_edge_idx}
    """
    
    boundary_edges_array = np.asarray(boundary_edges)
    mesh_vertices = np.asarray(mesh_vertices)
    sketch_vertices = np.asarray(sketch_vertices)
    sketch_edges = np.asarray(sketch_edges)
    
    print(f"Mapping {len(boundary_edges_array)} boundary edges to {len(sketch_edges)} sketch edges")
    
    # 获取边界边的顶点坐标
    # boundary_edge_vertices: (n_boundary_edges, 2, 3)
    boundary_edge_vertices = mesh_vertices[boundary_edges_array]
    
    # 获取所有 sketch edge 的顶点坐标
    # sketch_edge_vertices: (n_sketch_edges, 2, 3)
    sketch_edge_vertices = sketch_vertices[sketch_edges]
    
    # 向量化比较：创建 (n_boundary_edges, n_sketch_edges, 2, 3) 的形状
    
    # boundary_edge_vertices_expanded: (n_boundary_edges, 1, 2, 3)
    boundary_edge_vertices_expanded = boundary_edge_vertices[:, np.newaxis, :, :]
    # sketch_edge_vertices_expanded: (1, n_sketch_edges, 2, 3)
    sketch_edge_vertices_expanded = sketch_edge_vertices[np.newaxis, :, :, :]
    
    # 正向匹配: boundary_edge[0] <-> sketch_edge[0], boundary_edge[1] <-> sketch_edge[1]
    forward_diff = boundary_edge_vertices_expanded - sketch_edge_vertices_expanded
    forward_distances = np.linalg.norm(forward_diff, axis=3)
    forward_match = np.all(forward_distances < tol, axis=2)
    
    # 反向匹配: boundary_edge[0] <-> sketch_edge[1], boundary_edge[1] <-> sketch_edge[0]
    sketch_edge_vertices_reversed = sketch_edge_vertices[:, [1, 0], :]
    sketch_edge_vertices_reversed_expanded = sketch_edge_vertices_reversed[np.newaxis, :, :, :]
    
    backward_diff = boundary_edge_vertices_expanded - sketch_edge_vertices_reversed_expanded
    backward_distances = np.linalg.norm(backward_diff, axis=3)
    backward_match = np.all(backward_distances < tol, axis=2)
    
    # 总匹配: 正向或反向匹配
    # match_matrix: (n_boundary_edges, n_sketch_edges) 布尔矩阵
    match_matrix = forward_match | backward_match
    
    # 找到每个 boundary edge 的匹配 sketch edge
    boundary_to_sketch = {}

    for boundary_idx, boundary_edge in enumerate(boundary_edges):
        matching_sketch_indices = np.where(match_matrix[boundary_idx])[0]
        if len(matching_sketch_indices) > 0:
            sketch_idx = matching_sketch_indices[0]
            boundary_to_sketch[boundary_edge] = sketch_idx
            
            # 如果有多个匹配，发出警告
            if len(matching_sketch_indices) > 1:
                assert "Warning: Boundary edge {boundary_edge} matches multiple sketch edges: {matching_sketch_indices}"

    # every boundary should match to one sketch edge
    assert len(boundary_edges) == len(boundary_to_sketch) 
    
    return boundary_to_sketch

def average_vertex_normals(edge_solver_normals):
    """
    Compute averaged vertex normals from edge normals.

    Parameters
    ----------
    edge_solver_normals : dict
        {(v0, v1): normal (3,), ...}, where normal is np.ndarray

    Returns
    -------
    vertex_normals : dict
        {vertex_idx: averaged_normal (3,)}
    """
    accum = defaultdict(list)

    # accumulate edge normals for each vertex
    for (v0, v1), n in edge_solver_normals.items():
        n = np.asarray(n, dtype=float)
        accum[v0].append(n)
        accum[v1].append(n)

    vertex_normals = {}
    for v, normals in accum.items():
        avg = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-12:
            avg /= norm
        vertex_normals[v] = avg

    return vertex_normals


# put here so do not need import
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
    
    File format expected:
        v x y z     # vertex coordinates
        l i j       # edge between vertices i and j (1-based indices)
        vn nx ny nz # normal vector for edge
        
    Note:
        Edge indices are automatically converted from 1-based (OBJ file format)
        to 0-based (output) during reading.
        For each edge, both normal vectors are expected to be consecutive.
    """
    import numpy as np
    
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
        normals[(i, 0)] = normal_vectors[2*i]
        normals[(i, 1)] = normal_vectors[2*i + 1]
    
    # Convert lists to numpy arrays
    V = np.array(V)
    E = np.array(E)
    
    # Print summary of read data
    print(f"Read from {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {len(normal_vectors)} normal vectors ({len(normal_vectors) // len(E)} per edge)")
    
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


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Laplacian magnitude smoothing based on FiberMesh')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('normal_file', nargs= '?', help='normal .normal file with dual normals')
    parser.add_argument('--output', '-o', type=str, help='Output file path for smoothed mesh')
    args = parser.parse_args()

    v, f = igl.read_triangle_mesh(args.mesh_file)

    boundary_vertices, boundary_edges, edge_to_face = extract_boundary_info(f.tolist())
    constrained = np.asarray(boundary_vertices)

    if args.normal_file is None:
        v_smoothed = smooth_stanko(v, f, N = None, constrained = constrained, target_curvature_paper = True, curvature_method = 'EG')
    else:
        original_vertices, original_edges, original_normals = read_two_normal(args.normal_file)
        sketch_vertices, sketch_edges, edge_normals = resample_edge_dual_normal(
            original_vertices, original_edges, original_normals, target_edge_length = 0.05)
        
        # the edge_normals is a dictionary
        # (edge_idx, 0) -> normal 0
        # (edge_idx, 1) -> normal 1

        boundary_to_sketch = map_boundary_edges_to_sketch_edges(v, boundary_edges, sketch_vertices, sketch_edges, tol=1e-5)



        face_normals = igl.per_face_normals(v, f, np.array([0.0,0.0,1.0]))  # dummy up dir

        edge_solver_normals = {}
        for edge in boundary_edges:
            f_idx = edge_to_face[edge]
            face_n = face_normals[f_idx]
            
            # print('f_idx, face_n', f_idx, face_n)

            sketch_edge_idx = boundary_to_sketch[edge]
            candidates = [edge_normals[(sketch_edge_idx,0)],edge_normals[(sketch_edge_idx,1)] ]
            # print('sketch_edge_idx', sketch_edge_idx)
            # print('sketch_edge_normals', candidates)
            

            # compute dot products (cosine similarity)
            dots = [np.dot(face_n, cand) for cand in candidates]
            best_idx = np.argmax(dots)  # larger dot = smaller angle
            edge_solver_normals[edge] = candidates[best_idx]
        
        boundary_vertex_normals = average_vertex_normals(edge_solver_normals)
    

        constrained_normals = igl.per_vertex_normals(v, f)
        for boundary_vertex, normal in boundary_vertex_normals.items():
            constrained_normals[boundary_vertex] =  normal


        v_smoothed = smooth_stanko(v, f, N = constrained_normals, constrained = constrained, target_curvature_paper = True, curvature_method = 'EG')

    if args.output:
        igl.write_triangle_mesh(args.output, v_smoothed, f)


import igl
import scipy as sp
import numpy as np

import polyscope as ps



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

def compute_laplacian_magnitude(V, adjacency_list):
    n = len(V)
    c = np.zeros(n)
    for i in range(n):
        neighbors = adjacency_list[i]
        if len(neighbors) == 0:
            continue
        neighbor_mean = np.mean(V[neighbors], axis=0)
        c[i] = np.linalg.norm(V[i] - neighbor_mean)
    return c

def cotangent_weighted_vertex_normals(V, F):
    n = len(V)
    N = np.zeros((n, 3))

    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]
        e0 = vj - vk
        e1 = vk - vi
        e2 = vi - vj

        cot0 = np.dot(e1, -e2) / np.linalg.norm(np.cross(e1, -e2))  # angle at vi
        cot1 = np.dot(e2, -e0) / np.linalg.norm(np.cross(e2, -e0))  # at vj
        cot2 = np.dot(e0, -e1) / np.linalg.norm(np.cross(e0, -e1))  # at vk

        normal = np.cross(vj - vi, vk - vi)

        N[i] += cot0 * normal
        N[j] += cot1 * normal
        N[k] += cot2 * normal

    # Normalize
    norm = np.linalg.norm(N, axis=1)
    valid = norm > 1e-8
    N[valid] /= norm[valid][:, None]

    return N


def gen_weighted_graph_laplacian_matrix(v, f, adjacency_list):
    '''
    Extracted from fibermesh.py 
    '''
    import scipy.sparse as sp
    import igl
    import numpy as np
        
    N = len(v)
    
    A = sp.lil_matrix((N, N))
    
    for vi in range(N):
        neighbor_indices = adjacency_list[vi]
        
        if len(neighbor_indices) == 0:
            A[vi, vi] = 1.0
            continue
        
        weight = 1.0
        neighbor_weight = -weight / len(neighbor_indices)
        
        A[vi, vi] = weight
        
        for ni in neighbor_indices:
            A[vi, ni] = neighbor_weight
    
    A = A.tocsr()
    
    # Get mass matrix diagonal (vertex areas)
    mass_matrix = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    areas = mass_matrix.diagonal()
    
    # Apply area weighting
    area_diagonal = sp.diags(1.0 / areas, format='csr')
    A_weighted = area_diagonal * A
    
    return A_weighted, areas



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Curvature-based feature-preserving mesh smoothing')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    args = parser.parse_args()

    mesh_file = args.mesh_file

    v, f = igl.read_triangle_mesh(mesh_file)

    # use the paper method
    adj = igl.adjacency_list(f)
    c = compute_laplacian_magnitude(v, adj)

    plot(v, f, c, title='Discrete Laplacian norm')

    # use igl method
    l = igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

    minv = sp.sparse.diags(1 / m.diagonal())

    hn = -minv.dot(l.dot(v))
    h = np.linalg.norm(hn, axis=1) 


    print('h', h)
    plot(v, f, h, title='igl h')


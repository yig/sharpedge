import igl
import scipy as sp
import numpy as np

import polyscope as ps

import numpy as np

def estimate_vertex_area(v, f):
    """
    为每个顶点分配局部面积（每个三角形的面积均分到三个顶点）。
    v: 顶点坐标 (n, 3)
    f: 三角形面索引 (m, 3)
    返回: 每个顶点的面积估计值 (n,)
    """
    # 每个三角形的面积
    vec1 = v[f[:, 1]] - v[f[:, 0]]
    vec2 = v[f[:, 2]] - v[f[:, 0]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)  # shape: (m,)

    # 初始化每个顶点的面积为 0
    area_per_vertex = np.zeros(len(v))

    # 将每个三角形面积均分给三个顶点
    for i in range(3):
        np.add.at(area_per_vertex, f[:, i], tri_areas / 3.0)

    return area_per_vertex


def plot(v, f, k, title="Curvature"):
    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", v, f)
    ps_mesh.add_scalar_quantity(title, k, defined_on='vertices', cmap='coolwarm', enabled=True)
    
    ps_mesh.set_edge_width(1)

    ps.set_ground_plane_mode('none')
    ps.show()


def compute_laplacian_magnitude(V, adjacency_list):
    n = len(V)
    c = np.zeros(n)
    for i in range(n):
        neighbors = adjacency_list[i]
        if not neighbors:
            continue
        neighbor_mean = np.mean(V[neighbors], axis=0)
        c[i] = np.linalg.norm(V[i] - neighbor_mean)
    return c


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Curvature-based feature-preserving mesh smoothing')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    args = parser.parse_args()

    mesh_file = args.mesh_file

    v, f = igl.read_triangle_mesh(mesh_file)


    # area = estimate_vertex_area(v, f)
    # print("area min/max:", area.min(), area.max())
    # k = igl.gaussian_curvature(v, f)
    # print('k',k)
    # plot(v, f, k, title = 'gaussian_curvature')



    # v1, v2, k1, k2 = igl.principal_curvature(v, f)
    # h = 0.5 * (k1 + k2)
    # print('h', h)

    # plot(v, f, h, title='mean curvature')

    adj = igl.adjacency_list(f)
    c = compute_laplacian_magnitude(v, adj)

    plot(v, f, c, title='Discrete Laplacian norm')

    l = igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

    minv = sp.sparse.diags(1 / m.diagonal())

    hn = -minv.dot(l.dot(v))
    h = np.linalg.norm(hn, axis=1)


    print('h', h)
    plot(v, f, h, title='igl h')

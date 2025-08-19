import numpy as np
import polyscope as ps
import argparse

from utility_io import load_mesh_obj


def find_degenerate_faces(V, F, eps=1e-15):
    """
    V: (n,3) 顶点坐标
    F: (m,3) 面索引
    eps: 面积容忍阈值
    return: 退化面索引列表
    """
    deg_faces = []
    for i, f in enumerate(F):
        v0, v1, v2 = V[f]
        # 检查重复点
        if len(set(f)) < 3:
            deg_faces.append(i)
            continue
        # 计算面积
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        if area < eps:
            deg_faces.append(i)
    return np.array(deg_faces)




# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh degenerate viewer')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')    
    args = parser.parse_args()

    mesh_file = args.mesh_file
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)

    V = np.asarray(mesh_vertices)
    F = np.asarray(mesh_faces)



    ps.init()

    # 假设 V, F 是 mesh
    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    ps_mesh.set_edge_color((0, 0, 0))
    ps_mesh.set_edge_width(1.0)

    deg_faces = find_degenerate_faces(V, F)

    # 显示退化三角形的 barycenter
    centroids = np.mean(V[F[deg_faces]], axis=1)
    ps.register_point_cloud("degenerate face centers", centroids, radius=0.005, color=(1.0, 0.0, 0.0))

    # 或者显示退化三角形的边
    deg_edges = []
    for f in F[deg_faces]:
        deg_edges.append([f[0], f[1]])
        deg_edges.append([f[1], f[2]])
        deg_edges.append([f[2], f[0]])
    deg_edges = np.array(deg_edges)
    ps.register_curve_network("degenerate edges", V, deg_edges, radius=0.002, color=(1.0, 0.0, 0.0))
    ps.set_ground_plane_mode("none")

    ps.show()

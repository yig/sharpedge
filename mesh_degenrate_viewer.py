import numpy as np
import polyscope as ps
import argparse

from utility_io import load_mesh_obj


def find_degenerate_faces(V, F, eps=1e-8):
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
    return np.array(deg_faces, dtype=int)

def view_degenerate_triangles(V, F):
    
    ps.init()

    # 假设 V, F 是 mesh
    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    ps_mesh.set_edge_color((0, 0, 0))
    ps_mesh.set_edge_width(1.0)

    deg_faces = find_degenerate_faces(V, F)

    print('len(deg_faces)', len(deg_faces))
    print('deg_faces', deg_faces)
    print(F[0])
    print(V[0])
    print(V[1])
    print(V[2])


    # 收集所有退化三角形涉及到的顶点索引
    unique_vs, new_index = np.unique(F[deg_faces].flatten(), return_inverse=True)

    print('unique_vs', unique_vs)
    print('new_index', new_index)

    # 子集顶点
    deg_V = V[unique_vs]

    # 重新映射 deg_edges
    deg_edges = []
    for i in range(0, len(new_index), 3):
        a, b, c = new_index[i:i+3]
        deg_edges.append([a, b])
        deg_edges.append([b, c])
        deg_edges.append([c, a])
    deg_edges = np.array(deg_edges)

    print('deg_edges', deg_edges)

    if unique_vs.size > 0:
        # 用子集顶点注册 curve network
        ps.register_curve_network(
            "degenerate edges", deg_V, deg_edges, radius=0.002, color=(1.0, 0.0, 0.0)
        )


    ps.set_ground_plane_mode("none")

    ps.show()

def find_point_faces(V, F, eps=1e-8):
    '''
    find that triangles that is actual a point
    '''
    deg_faces = []
    for i, f in enumerate(F):
        v0, v1, v2 = V[f]
        if np.allclose(v0, v1, atol= eps) and np.allclose(v0, v2, atol=eps):
            deg_faces.append(i)

    return np.array(deg_faces, dtype=int)
 
def view_point_triangles(V, F):
    '''
    '''
    ps.init()

    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    ps_mesh.set_edge_color((0, 0, 0))
    ps_mesh.set_edge_width(1.0)

    deg_faces = find_point_faces(V, F)

    print('deg_faces',deg_faces)

    # 收集所有三角形涉及到的顶点索引
    # unique_vs, new_index = np.unique(F[deg_faces].flatten(), return_inverse=True)
    # deg_V = V[unique_vs]

    deg_V = V[F[deg_faces].flatten()]

    face_indices = []
    for fi in deg_faces:
        face_indices.extend([fi, fi, fi])  # 每个三角形3个顶点
    face_indices = np.array(face_indices)

    pc = ps.register_point_cloud("point triangles", deg_V, radius=0.005, color=(1, 0, 0))
    pc.add_scalar_quantity("face index", face_indices, enabled=True)



  
# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh degenerate viewer')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')    
    args = parser.parse_args()

    mesh_file = args.mesh_file
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)

    V = np.asarray(mesh_vertices)
    F = np.asarray(mesh_faces)

    # view_point_triangles(V, F)
    view_degenerate_triangles(V, F)

        





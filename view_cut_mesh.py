import polyscope as ps
import numpy as np
from collections import defaultdict
import argparse
import os

def read_obj_file(filename):
    """读取OBJ文件，返回顶点和面片"""
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # 顶点坐标
                coords = line.split()[1:4]
                vertices.append([float(x) for x in coords])
            elif line.startswith('f '):
                # 面片索引 (转换为0-based)
                indices = line.split()[1:4]
                face = []
                for idx in indices:
                    # 处理 "vertex/texture/normal" 格式
                    vertex_idx = int(idx.split('/')[0]) - 1  # 转换为0-based
                    face.append(vertex_idx)
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def extract_boundary_edges(vertices, faces):
    """提取网格的边界边"""
    # 统计每条边被多少个面使用
    edge_count = defaultdict(int)
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            # 确保边的顶点按升序排列
            edge = (min(v1, v2), max(v1, v2))
            edge_count[edge] += 1
    
    # 找出只被一个面使用的边（边界边）
    boundary_edges = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_edges.append(edge)
    
    return boundary_edges

def create_edge_network(vertices, edges):
    """从边列表创建边网络的顶点和边"""
    if not edges:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2)
    
    # 创建边的端点坐标
    edge_vertices = []
    edge_connections = []
    
    for i, (v1, v2) in enumerate(edges):
        # 每条边由两个端点组成
        edge_vertices.append(vertices[v1])
        edge_vertices.append(vertices[v2])
        # 连接这两个端点
        edge_connections.append([2*i, 2*i+1])
    
    return np.array(edge_vertices), np.array(edge_connections)

def visualize_mesh_boundaries(mesh_file):
    """使用Polyscope可视化网格边界"""
    
    # 初始化Polyscope
    ps.init()
    ps.set_program_name("网格边界可视化")
    
    # 读取切割后的网格
    print(f"读取网格文件: {mesh_file}")
    V_cut, F_cut = read_obj_file(mesh_file)
    print(f"顶点数: {len(V_cut)}, 面片数: {len(F_cut)}")
    
    # 添加主网格
    mesh_name = "切割后的网格"
    ps_mesh = ps.register_surface_mesh(mesh_name, V_cut, F_cut)
    ps_mesh.set_color([0.8, 0.8, 0.9])  # 淡蓝色
    ps_mesh.set_transparency(0.8)
    
    # 提取所有边界边
    all_boundaries = extract_boundary_edges(V_cut, F_cut)
    
    if all_boundaries:
        # 创建边界边网络
        boundary_vertices, boundary_edges = create_edge_network(V_cut, all_boundaries)
        
        if len(boundary_vertices) > 0:
            ps_boundaries = ps.register_curve_network(
                "所有边界边", boundary_vertices, boundary_edges
            )
            ps_boundaries.set_color([1.0, 0.0, 0.0])  # 红色
            ps_boundaries.set_radius(0.003)
            print(f"显示了 {len(all_boundaries)} 条边界边")
    

    
    # 设置相机和显示
    ps.set_automatically_compute_scene_extents(True)
    ps.set_ground_plane_mode("none")
    

    # 显示
    ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化网格边界')
    parser.add_argument('cut_mesh', help='切割后的网格文件 (.obj)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cut_mesh):
        print(f"错误: 文件不存在 {args.cut_mesh}")
    

    
    visualize_mesh_boundaries(args.cut_mesh)
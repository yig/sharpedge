import numpy as np
import polyscope as ps

def read_two_normal(filename):
    """
    Read vertices, edges, and dual normal data from an OBJ file.
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
    
    print(f"Read from {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {len(normal_vectors)} normal vectors")
    
    return V, E, normals

def load_mesh_obj(filename):
    """
    Load mesh from OBJ file (vertices and faces)
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
                
            elif parts[0] == 'f':  # Face
                face_indices = []
                for part in parts[1:]:
                    vertex_index = int(part.split('/')[0]) - 1
                    face_indices.append(vertex_index)
                
                if len(face_indices) == 3:
                    faces.append(face_indices)
    
    return np.array(vertices), np.array(faces)

def simple_viewer(normal_file, mesh_file, target_edge_length=0.05):
    """
    简单显示 normal 文件和 mesh 文件，包含验证统计
    """
    
    # 读取数据
    print("Loading normal file...")
    sketch_vertices, sketch_edges, normals = read_two_normal(normal_file)
    
    print("Loading mesh file...")
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)
    
    # 简单的验证：计算 sketch 顶点到 mesh 顶点的最近距离
    from scipy.spatial import cKDTree
    
    print(f"\n=== 简单验证统计 ===")
    print(f"Sketch: {len(sketch_vertices)} vertices, {len(sketch_edges)} edges")
    print(f"Mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
    
    # 构建 mesh 顶点的 KD 树
    mesh_tree = cKDTree(mesh_vertices)
    tolerance = 1e-3
    
    # 检查每个 sketch 顶点到最近 mesh 顶点的距离
    vertices_on_mesh = 0
    max_distance = 0
    total_distance = 0
    
    for i, vertex in enumerate(sketch_vertices):
        distance, _ = mesh_tree.query(vertex)
        if distance <= tolerance:
            vertices_on_mesh += 1
        max_distance = max(max_distance, distance)
        total_distance += distance
    
    # 统计边
    edges_on_mesh = 0
    for edge in sketch_edges:
        v1_idx, v2_idx = edge
        v1_distance, _ = mesh_tree.query(sketch_vertices[v1_idx])
        v2_distance, _ = mesh_tree.query(sketch_vertices[v2_idx])
        
        if v1_distance <= tolerance and v2_distance <= tolerance:
            edges_on_mesh += 1
    
    print(f"容忍距离: {tolerance}")
    print(f"顶点在 mesh 上: {vertices_on_mesh}/{len(sketch_vertices)} "
          f"({100*vertices_on_mesh/len(sketch_vertices):.1f}%)")
    print(f"边在 mesh 上: {edges_on_mesh}/{len(sketch_edges)} "
          f"({100*edges_on_mesh/len(sketch_edges):.1f}%)")
    print(f"最大顶点距离: {max_distance:.6f}")
    print(f"平均顶点距离: {total_distance/len(sketch_vertices):.6f}")
    
    # 初始化 polyscope
    ps.init()
    
    # 添加 mesh - 不透明
    ps_mesh = ps.register_surface_mesh("mesh", mesh_vertices, mesh_faces)
    ps_mesh.set_color([0.8, 0.8, 0.8])
    # 移除透明度设置，保持不透明
    
    # 添加 sketch 的顶点
    ps_points = ps.register_point_cloud("sketch_vertices", sketch_vertices)
    ps_points.set_radius(0.002, True)  # 相对半径
    
    # 确保 edges 是 numpy 数组
    sketch_edges = np.array(sketch_edges)
    
    # 添加 sketch 的边
    ps_edges = ps.register_curve_network("sketch_edges", sketch_vertices, sketch_edges)
    ps_edges.set_radius(0.002)
    
    # 显示 normal 向量 - 分别用不同颜色
    if normals:
        # 分别收集两个 normal 向量
        normal1_starts = []
        normal1_ends = []
        normal2_starts = []
        normal2_ends = []
        
        normal_scale = 0.05  # normal 向量显示长度
        
        for edge_idx, edge in enumerate(sketch_edges):
            v1_idx, v2_idx = edge
            edge_center = (sketch_vertices[v1_idx] + sketch_vertices[v2_idx]) / 2
            
            # 第一个 normal 向量 (红色)
            if (edge_idx, 0) in normals:
                normal_vec = np.array(normals[(edge_idx, 0)])
                normal1_starts.append(edge_center)
                normal1_ends.append(edge_center + normal_vec * normal_scale)
            
            # 第二个 normal 向量 (蓝色)
            if (edge_idx, 1) in normals:
                normal_vec = np.array(normals[(edge_idx, 1)])
                normal2_starts.append(edge_center)
                normal2_ends.append(edge_center + normal_vec * normal_scale)
        
        # 显示第一个 normal 向量组 (红色)
        if normal1_starts:
            normal1_starts = np.array(normal1_starts)
            normal1_ends = np.array(normal1_ends)
            
            normal1_edges = np.array([[i, i + len(normal1_starts)] for i in range(len(normal1_starts))])
            normal1_vertices = np.vstack([normal1_starts, normal1_ends])
            
            ps_normals1 = ps.register_curve_network("normals1", normal1_vertices, normal1_edges)
            ps_normals1.set_radius(0.0025)
            ps_normals1.set_color([1.0, 0.0, 0.0])  # 红色
        
        # 显示第二个 normal 向量组 (蓝色)
        if normal2_starts:
            normal2_starts = np.array(normal2_starts)
            normal2_ends = np.array(normal2_ends)
            
            normal2_edges = np.array([[i, i + len(normal2_starts)] for i in range(len(normal2_starts))])
            normal2_vertices = np.vstack([normal2_starts, normal2_ends])
            
            ps_normals2 = ps.register_curve_network("normals2", normal2_vertices, normal2_edges)
            ps_normals2.set_radius(0.0025)
            ps_normals2.set_color([0.0, 0.0, 1.0])  # 蓝色
    
    print(f"\n=== Polyscope 可视化 ===")
    print("灰色：mesh 表面")
    print("默认色：sketch 顶点和边")
    print("红色：第一个 normal 向量")
    print("蓝色：第二个 normal 向量")
    
    # 显示
    ps.show()

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple viewer for normal and mesh files with resampling')
    parser.add_argument('normal_file', help='Normal .obj file with edges and normals')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('--target-length', '-t', type=float, default=0.05,
                       help='Target edge length for resampling (default: 0.05)')
    
    args = parser.parse_args()
    
    # 运行简单查看器（包含重采样）
    simple_viewer(args.normal_file, args.mesh_file, args.target_length)
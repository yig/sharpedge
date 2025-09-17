import numpy as np
import polyscope as ps
from collections import defaultdict

from utility_io import read_two_normal, load_mesh_obj

def resample_edge_dual_normal(V, E, normals, target_edge_length=0.04):
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

def match_mesh_to_sketch(mesh_vertices, mesh_edges, sketch_vertices, sketch_edges, tol=1e-5):
    """
    vectorize map mesh_edges to sketch_edges。
    
    parameters:
        mesh_vertices: (n_mesh_vertices, 3) ndarray
        mesh_edges: list of tuples or (n_mesh_edges, 2) array
        sketch_vertices: (n_sketch_vertices, 3) ndarray
        sketch_edges: (n_sketch_edges, 2) ndarray
        tol: tolerance
    
    return:
        mesh_sketch_edges: dict {mesh_edge_idx: sketch_edge_idx}
        sketch_to_mesh: dict {sketch_edge_idx: list of mesh_edge_idx}
    """
    # 确保输入是 numpy 数组
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_edges = np.asarray(mesh_edges)
    sketch_vertices = np.asarray(sketch_vertices)
    sketch_edges = np.asarray(sketch_edges)
    
    # 获取所有 mesh edge 的顶点坐标
    # mesh_edges: (n_mesh_edges, 2), mesh_vertices: (n_mesh_vertices, 3)
    # mesh_edge_vertices: (n_mesh_edges, 2, 3)
    mesh_edge_vertices = mesh_vertices[mesh_edges]
    
    # 获取所有 sketch edge 的顶点坐标
    # sketch_edge_vertices: (n_sketch_edges, 2, 3)
    sketch_edge_vertices = sketch_vertices[sketch_edges]
    
    # 为了向量化比较，我们需要比较每个 mesh edge 与每个 sketch edge
    # 使用广播来创建 (n_mesh_edges, n_sketch_edges, 2, 3) 的形状
    
    # mesh_edge_vertices_expanded: (n_mesh_edges, 1, 2, 3)
    mesh_edge_vertices_expanded = mesh_edge_vertices[:, np.newaxis, :, :]
    # sketch_edge_vertices_expanded: (1, n_sketch_edges, 2, 3)
    sketch_edge_vertices_expanded = sketch_edge_vertices[np.newaxis, :, :, :]
    
    # 正向匹配: mesh_edge[0] <-> sketch_edge[0], mesh_edge[1] <-> sketch_edge[1]
    # 计算距离差: (n_mesh_edges, n_sketch_edges, 2, 3)
    forward_diff = mesh_edge_vertices_expanded - sketch_edge_vertices_expanded
    # 计算每个顶点对的欧几里得距离: (n_mesh_edges, n_sketch_edges, 2)
    forward_distances = np.linalg.norm(forward_diff, axis=3)
    # 检查两个顶点是否都匹配: (n_mesh_edges, n_sketch_edges)
    forward_match = np.all(forward_distances < tol, axis=2)
    
    # 反向匹配: mesh_edge[0] <-> sketch_edge[1], mesh_edge[1] <-> sketch_edge[0]
    # 交换 sketch edge 的顶点顺序
    sketch_edge_vertices_reversed = sketch_edge_vertices[:, [1, 0], :]  # 交换第1维的0和1
    sketch_edge_vertices_reversed_expanded = sketch_edge_vertices_reversed[np.newaxis, :, :, :]
    
    backward_diff = mesh_edge_vertices_expanded - sketch_edge_vertices_reversed_expanded
    backward_distances = np.linalg.norm(backward_diff, axis=3)
    backward_match = np.all(backward_distances < tol, axis=2)
    
    # 总匹配: 正向或反向匹配
    # match_matrix: (n_mesh_edges, n_sketch_edges) 布尔矩阵
    match_matrix = forward_match | backward_match
    
    # 找到每个 mesh edge 的匹配 sketch edge
    mesh_sketch_edges = {}
    for mesh_idx in range(len(mesh_edges)):
        # 找到与当前 mesh edge 匹配的所有 sketch edges
        matching_sketch_indices = np.where(match_matrix[mesh_idx])[0]
        if len(matching_sketch_indices) > 0:
            # 如果有多个匹配，取第一个
            mesh_sketch_edges[mesh_idx] = matching_sketch_indices[0]
    
    # 构建 sketch_to_mesh 映射
    sketch_to_mesh = defaultdict(list)
    for mesh_edge, sketch_edge in mesh_sketch_edges.items():
        sketch_to_mesh[sketch_edge].append(mesh_edge)
    
    return mesh_sketch_edges, sketch_to_mesh

def simple_viewer(normal_file, mesh_file, target_edge_length=0.04):
    """
    简单显示 normal 文件和 mesh 文件，包含验证统计
    """
    
    # 读取数据
    print("Loading normal file...")
    original_vertices, original_edges, original_normals = read_two_normal(normal_file)

    print("Resampling processing...")
    sketch_vertices, sketch_edges, normals = resample_edge_dual_normal(
        original_vertices, original_edges, original_normals, target_edge_length)
    
    print("Loading mesh file...")
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)

    # 简单的验证：计算 sketch 顶点到 mesh 顶点的最近距离
    from scipy.spatial import cKDTree
    
    print(f"\n=== 简单验证统计 ===")
    print(f"Sketch: {len(sketch_vertices)} vertices, {len(sketch_edges)} edges")
    print(f"Mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
    
    # 构建 mesh 顶点的 KD 树
    mesh_tree = cKDTree(mesh_vertices)
    tolerance = 1e-5
    
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
    mesh_edges = find_mesh_edges(mesh_faces)

    mesh_sketch_edges, sketch_to_mesh = match_mesh_to_sketch(mesh_vertices, mesh_edges, sketch_vertices, sketch_edges)



    duplicates = {sketch: meshes for sketch, meshes in sketch_to_mesh.items() if len(meshes) > 1}
    print('len(mesh_sketch_edges)', len(mesh_sketch_edges))
    print('len(sketch_to_mesh)', len(sketch_to_mesh))

    # print('duplicates', duplicates)
    # for key, mesh_dups in duplicates.items():
    #     print([mesh_edges[i] for i in mesh_dups])
    print(f"容忍距离: {tolerance}")
    print(f"顶点在 mesh 上: {vertices_on_mesh}/{len(sketch_vertices)} "
          f"({100*vertices_on_mesh/len(sketch_vertices):.1f}%)")
    print(f"边在 mesh 上: {len(mesh_sketch_edges) }/{len(sketch_edges)} "
          f"({100*len(mesh_sketch_edges)/len(sketch_edges):.1f}%)")
    print(f"最大顶点距离: {max_distance:.6f}")
    print(f"平均顶点距离: {total_distance/len(sketch_vertices):.6f}")
    
    # V = mesh_vertices
    # print('V[460], V[815], V[435]', V[460], V[815], V[435])
    # print('V[278], V[543], V[666]', V[278], V[543], V[666])

    # 初始化 polyscope
    ps.init()
    
    # 添加 mesh - 不透明
    ps_mesh = ps.register_surface_mesh("mesh", np.asarray(mesh_vertices), np.asarray(mesh_faces))
    ps_mesh.set_color([0.8, 0.8, 0.8])
    ps_mesh.set_edge_color([0, 0, 0])
    ps_mesh.set_edge_width(1.0)
    # 移除透明度设置，保持不透明
    
    # 添加 sketch 的顶点
    ps_points = ps.register_point_cloud("sketch_vertices", sketch_vertices)
    ps_points.set_radius(0.002, True)  # 相对半径
    
    # 确保 edges 是 numpy 数组
    sketch_edges = np.array(sketch_edges)
    
    # 添加 sketch 的边
    edge_mapping_counts = np.array([len(sketch_to_mesh[i]) for i in range(len(sketch_edges))])
    ps_edges = ps.register_curve_network("sketch_edges", sketch_vertices, sketch_edges)
    ps_edges.add_scalar_quantity("mapping_count", edge_mapping_counts, defined_on='edges', enabled=True)
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
    
    ps.set_ground_plane_mode('none')
    # 显示
    ps.show()

def find_mesh_edges(F):
    '''
    '''
    
    edges = set()
    
    for face in F:
        face_edges = [
            (face[0], face[1]),
            (face[1], face[2]), 
            (face[2], face[0])
        ]
        
        for v1, v2 in face_edges:
            edge = (min(v1, v2), max(v1, v2))
            edges.add(edge)
    
    return list(edges)
    
# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple viewer for normal and mesh files with resampling')
    parser.add_argument('normal_file', help='Normal .obj file with edges and normals')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('--target-length', '-t', type=float, default=0.04,
                       help='Target edge length for resampling (default: 0.04)')
    
    args = parser.parse_args()
    
    # 运行简单查看器（包含重采样）
    simple_viewer(args.normal_file, args.mesh_file, args.target_length)
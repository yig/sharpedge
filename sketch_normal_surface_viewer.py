import numpy as np
import polyscope as ps

from utility_io import read_two_normal, load_mesh_obj

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

def simple_viewer(normal_file, mesh_file, target_edge_length=0.05):
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
    for edge in sketch_edges:
        v1_idx, v2_idx = edge
        v1_distance, _ = mesh_tree.query(sketch_vertices[v1_idx])
        v2_distance, _ = mesh_tree.query(sketch_vertices[v2_idx])
        
        if v1_distance <= tolerance and v2_distance <= tolerance:
            edges_on_mesh += 1

    # 初始化 polyscope
    ps.init()
    
    # 添加 mesh - 不透明
    ps_mesh = ps.register_surface_mesh("mesh", mesh_vertices, mesh_faces)
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
    
    ps.set_ground_plane_mode('none')
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
import numpy as np
import polyscope as ps
from collections import defaultdict
from matplotlib.colors import hsv_to_rgb

from utility_io import read_two_normal, load_mesh_obj, load_sketch_polyline_data

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


def generate_distinct_colors(n):
    """
    生成 n 种视觉上区分度高的颜色
    """
    if n == 0:
        return []
    
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # 黄金角度
        saturation = 0.8 + 0.2 * (i % 3) / 3
        value = 0.7 + 0.3 * ((i + 1) % 4) / 4
        
        rgb = hsv_to_rgb([hue/360, saturation, value])
        colors.append(rgb.tolist())
    
    return colors


def add_sketch_polylines(sketch_file):
    """
    加载并显示额外的 sketch polylines，每条使用不同颜色
    """
    print(f"\n=== 加载额外 sketch 文件: {sketch_file} ===")
    
    try:
        V_sketch, E_sketch, P_sketch = load_sketch_polyline_data(sketch_file)
        V_sketch = np.asarray(V_sketch)
        
        print(f"Sketch 数据: {len(P_sketch)} polylines, {len(V_sketch)} vertices")
        
        # 生成颜色
        colors = generate_distinct_colors(len(P_sketch))
        
        # 显示每条 polyline
        for polyline_idx, polyline in enumerate(P_sketch):
            if len(polyline) < 2:
                continue
            
            polyline_vertices = V_sketch[polyline]
            
            # 创建边
            edges = []
            for i in range(len(polyline) - 1):
                edges.append([i, i + 1])
            
            if not edges:
                continue
            
            edges_array = np.array(edges)
            
            # 注册 curve network
            curve_name = f"sketch_polyline_{polyline_idx}"
            ps_curve = ps.register_curve_network(curve_name, polyline_vertices, edges_array)
            ps_curve.set_radius(0.004)  # 稍粗一点以便看清
            
            color = colors[polyline_idx % len(colors)]
            ps_curve.set_color(color)
            
            # 注册顶点
            # ps_points = ps.register_point_cloud(f"{curve_name}_points", polyline_vertices)
            # ps_points.set_radius(0.005, True)
            # ps_points.set_color(color)
        
        print(f"显示了 {len(P_sketch)} 条 sketch polylines，每条都有独特的颜色")
        
    except Exception as e:
        print(f"Error loading sketch file {sketch_file}: {e}")


def simple_viewer(normal_file, mesh_file, target_edge_length=0.04, sketch_file=None, extra_mesh_file=None):
    """
    简单显示 normal 文件和 mesh 文件，包含验证统计
    可选择添加 sketch polylines 显示
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

    # 初始化 polyscope
    ps.init()

    # 参数
    elev = 22    # 仰角
    azim = -29  # 方位角
    r = 3.0      # 相机距离

    # 转成弧度
    elev = np.deg2rad(elev)
    azim = np.deg2rad(azim)

    # matplotlib 的 vertical_axis='y' → y 是 up
    x = r * np.cos(elev) * np.sin(azim)
    y = r * np.sin(elev)
    z = r * np.cos(elev) * np.cos(azim)

    camera_pos = [x, y, z]

    ps.look_at(camera_pos,[0,0,0])
    
    # 添加 mesh - 不透明
    ps_mesh = ps.register_surface_mesh("mesh", np.asarray(mesh_vertices), np.asarray(mesh_faces))
    ps_mesh.set_color([0.8, 0.8, 0.8])
    ps_mesh.set_edge_color([0, 0, 0])
    ps_mesh.set_smooth_shade(True)

    # ps_mesh.set_edge_width(1.0)
    # 移除透明度设置，保持不透明
    
    # 添加 sketch 的顶点
    # ps_points = ps.register_point_cloud("sketch_vertices", sketch_vertices)
    # ps_points.set_radius(0.002, True)  # 相对半径
    
    # 确保 edges 是 numpy 数组
    sketch_edges = np.array(sketch_edges)
    
    # 添加 sketch 的边

    
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
    
    # 如果提供了额外的 mesh 文件，显示它
    if extra_mesh_file:
        try:
            print(f"\n=== 加载额外 mesh 文件: {extra_mesh_file} ===")
            extra_mesh_vertices, extra_mesh_faces = load_mesh_obj(extra_mesh_file)
            
            # 简单显示额外的 mesh，使用浅蓝色
            ps_extra_mesh = ps.register_surface_mesh("extra_mesh", np.asarray(extra_mesh_vertices), np.asarray(extra_mesh_faces))
            # ps_extra_mesh.set_color([0.7, 0.9, 1.0])  # 浅蓝色
            # ps_extra_mesh.set_edge_color([0.3, 0.3, 0.3])  # 深灰色边
            # ps_extra_mesh.set_edge_width(0.5)
            
            ps_extra_mesh.set_color([0.8, 0.8, 0.8])
            ps_extra_mesh.set_edge_color([0, 0, 0])
            # ps_extra_mesh.set_edge_width(1.0)
            ps_extra_mesh.set_smooth_shade(True)

            print(f"Extra mesh: {len(extra_mesh_vertices)} vertices, {len(extra_mesh_faces)} faces")
            
        except Exception as e:
            print(f"Error loading extra mesh file {extra_mesh_file}: {e}")
    
    # 如果提供了 sketch 文件，添加 sketch polylines
    if sketch_file:
        add_sketch_polylines(sketch_file)
    
    print(f"\n=== Polyscope 可视化 ===")
    print("灰色：mesh 表面")
    print("默认色：sketch 顶点和边")
    print("红色：第一个 normal 向量")
    print("蓝色：第二个 normal 向量")
    if extra_mesh_file:
        print("浅蓝色：额外的 mesh 表面")
    if sketch_file:
        print("彩色线条：额外的 sketch polylines（每条不同颜色）")
    
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
    parser.add_argument('--sketch-file', '-s', help='Additional sketch file to display with colorful polylines')
    parser.add_argument('--extra-mesh', '-m', help='Additional mesh file to display')
    
    args = parser.parse_args()
    
    # 运行简单查看器（包含重采样和可选的 sketch 显示）
    simple_viewer(args.normal_file, args.mesh_file, args.target_length, args.sketch_file, args.extra_mesh)
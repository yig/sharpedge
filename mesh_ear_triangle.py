import numpy as np
import polyscope as ps
import argparse

from scipy.spatial import cKDTree
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

def mesh_to_sketch_dict(sketch_vertices: np.ndarray,
                        mesh_vertices: np.ndarray,
                        tolerance=1e-6):
    """
    返回一个字典 { mesh_idx -> sketch_idx }。
    每个 mesh 顶点至多对应一个 sketch 顶点，
    若没有匹配则该 mesh 顶点不出现在字典中。
    """
    tree = cKDTree(sketch_vertices)

    mapping = {}
    for mesh_idx, mv in enumerate(mesh_vertices):
        # 找 mesh 顶点最近的 sketch 顶点
        dist, sketch_idx = tree.query(mv)
        if dist <= tolerance:
            mapping[mesh_idx] = sketch_idx

    return mapping

def build_vertex_to_edge_mapping(sketch_edges):
    """
    构建从sketch顶点到边的映射
    返回字典: {vertex_idx: [edge_idx1, edge_idx2, ...]}
    """
    vertex_to_edges = {}
    
    for edge_idx, edge in enumerate(sketch_edges):
        v1, v2 = edge[0], edge[1]
        
        if v1 not in vertex_to_edges:
            vertex_to_edges[v1] = []
        vertex_to_edges[v1].append(edge_idx)
        
        if v2 not in vertex_to_edges:
            vertex_to_edges[v2] = []
        vertex_to_edges[v2].append(edge_idx)
    
    return vertex_to_edges

def find_cross_edge_triangles(mesh_faces, vertex_mapping, sketch_edges):
    """
    找到有两个顶点分别位于不同sketch edge上的三角形
    
    Args:
        mesh_faces: mesh的三角形面
        vertex_mapping: mesh顶点到sketch顶点的映射
        sketch_edges: sketch的边
        
    Returns:
        cross_edge_triangles: 跨边的三角形列表
        triangle_info: 每个跨边三角形的详细信息
    """
    
    # 构建sketch顶点到边的映射
    vertex_to_edges = build_vertex_to_edge_mapping(sketch_edges)
    
    cross_edge_triangles = []
    triangle_info = []
    
    print(f"检查 {len(mesh_faces)} 个三角形...")
    
    for tri_idx, triangle in enumerate(mesh_faces):
        # 获取三角形的三个顶点
        mesh_v1, mesh_v2, mesh_v3 = triangle
        
        # 检查这些mesh顶点是否映射到sketch顶点
        sketch_vertices = []
        mapped_vertices = []
        
        for mesh_v in [mesh_v1, mesh_v2, mesh_v3]:
            if mesh_v in vertex_mapping:
                sketch_v = vertex_mapping[mesh_v]
                sketch_vertices.append(sketch_v)
                mapped_vertices.append((mesh_v, sketch_v))
        
        # 需要至少两个顶点映射到sketch
        if len(sketch_vertices) < 2:
            continue
        
        # 找到每个sketch顶点所在的边
        vertex_edges = {}
        for mesh_v, sketch_v in mapped_vertices:
            if sketch_v in vertex_to_edges:
                vertex_edges[sketch_v] = vertex_to_edges[sketch_v]
        
        # 检查是否有顶点在不同的边上
        edge_sets = []
        for sketch_v in sketch_vertices:
            if sketch_v in vertex_edges:
                edge_sets.append(set(vertex_edges[sketch_v]))
        
        if len(edge_sets) < 2:
            continue
        
        # 检查是否有不同的边
        has_different_edges = False
        for i in range(len(edge_sets)):
            for j in range(i + 1, len(edge_sets)):
                # 如果两个顶点的边集合没有交集，说明在不同边上
                if edge_sets[i].isdisjoint(edge_sets[j]):
                    has_different_edges = True
                    break
            if has_different_edges:
                break
        
        if has_different_edges:
            cross_edge_triangles.append(tri_idx)
            
            # 记录详细信息
            info = {
                'triangle_idx': tri_idx,
                'mesh_vertices': [mesh_v1, mesh_v2, mesh_v3],
                'mapped_vertices': mapped_vertices,
                'vertex_edges': vertex_edges
            }
            triangle_info.append(info)
    
    return cross_edge_triangles, triangle_info

def visualize_cross_edge_triangles(mesh_vertices, mesh_faces, cross_edge_triangles, sketch_vertices, sketch_edges):
    """
    可视化跨边的三角形
    """
    ps.init()
    
    # 显示完整mesh（半透明）
    ps_mesh = ps.register_surface_mesh("full_mesh", mesh_vertices, mesh_faces)
    ps_mesh.set_color([0.7, 0.7, 0.7])
    ps_mesh.set_transparency(0.3)
    
    # 显示跨边三角形（红色）
    if len(cross_edge_triangles) > 0:
        cross_faces = mesh_faces[cross_edge_triangles]
        ps_cross = ps.register_surface_mesh("cross_edge_triangles", mesh_vertices, cross_faces)
        ps_cross.set_color([1.0, 0.0, 0.0])
        
        # 添加标量量来区分
        triangle_labels = np.ones(len(cross_faces))
        ps_cross.add_scalar_quantity("cross_edge_label", triangle_labels, defined_on='faces')
    
    # 显示sketch edges（蓝色线条）
    if len(sketch_edges) > 0:
        ps_edges = ps.register_curve_network("sketch_edges", sketch_vertices, sketch_edges)
        ps_edges.set_color([0.0, 0.0, 1.0])
        ps_edges.set_radius(0.002)
    
    # 显示sketch vertices（绿色点）
    ps_vertices = ps.register_point_cloud("sketch_vertices", sketch_vertices)
    ps_vertices.set_color([0.0, 1.0, 0.0])
    ps_vertices.set_radius(0.003)
    
    ps.set_ground_plane_mode("none")
    
    ps.show()

def print_cross_edge_statistics(cross_edge_triangles, triangle_info, mesh_faces):
    """
    打印跨边三角形的统计信息
    """
    print(f"\n=== 跨边三角形统计 ===")
    print(f"总三角形数: {len(mesh_faces)}")
    print(f"跨边三角形数: {len(cross_edge_triangles)}")
    print(f"跨边比例: {len(cross_edge_triangles) / len(mesh_faces) * 100:.2f}%")
    
    if len(triangle_info) > 0:
        print(f"\n前 {min(5, len(triangle_info))} 个跨边三角形的详细信息:")
        for i, info in enumerate(triangle_info[:5]):
            print(f"\n三角形 {info['triangle_idx']}:")
            print(f"  Mesh顶点: {info['mesh_vertices']}")
            print(f"  映射的顶点: {info['mapped_vertices']}")
            for sketch_v, edges in info['vertex_edges'].items():
                print(f"  Sketch顶点 {sketch_v} 在边: {edges}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='找到跨越不同sketch edge的三角形')
    parser.add_argument('normal_file', help='normal .normal file with normals')    
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='顶点匹配容差')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    args = parser.parse_args()

    mesh_file = args.mesh_file
    normal_file = args.normal_file

    # 加载数据
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)
    original_vertices, original_edges, original_normals = read_two_normal(normal_file)
    sketch_vertices, sketch_edges, normals = resample_edge_dual_normal(
        original_vertices, original_edges, original_normals)
    
    # 建立顶点映射
    vertex_mapping = mesh_to_sketch_dict(sketch_vertices, mesh_vertices, tolerance=args.tolerance)
    print(f"顶点映射数量: {len(vertex_mapping)}")

    mesh_vertices = np.asarray(mesh_vertices)
    mesh_faces = np.asarray(mesh_faces)
    
    # 找到跨边三角形
    cross_edge_triangles, triangle_info = find_cross_edge_triangles(
        mesh_faces, vertex_mapping, sketch_edges)
    
    # 打印统计信息
    print_cross_edge_statistics(cross_edge_triangles, triangle_info, mesh_faces)
    
    # 可视化（如果需要）
    if args.visualize:
        visualize_cross_edge_triangles(
            mesh_vertices, mesh_faces, np.array(cross_edge_triangles), 
            sketch_vertices, sketch_edges)
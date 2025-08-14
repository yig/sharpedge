import numpy as np
import trimesh
from collections import defaultdict
import argparse

def detect_nonmanifold(V, F):
    """快速检测非流形边和顶点"""
    
    # 构建边到面的映射
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(F):
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),  
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
    
    # 找非流形边 (>2个面共享)
    nonmanifold_edges = []
    nonmanifold_vertices = set()
    
    for edge, faces in edge_to_faces.items():
        if len(faces) > 2:
            nonmanifold_edges.append(edge)
            nonmanifold_vertices.update(edge)  # 非流形边的端点也是问题顶点
    
    print(f"网格统计: {len(V)} 顶点, {len(F)} 面")
    print(f"非流形边: {len(nonmanifold_edges)}")
    print(f"非流形顶点: {len(nonmanifold_vertices)}")
    
    # 显示前几个问题
    if nonmanifold_edges:
        print("\n前5个非流形边:")
        for i, edge in enumerate(nonmanifold_edges[:5]):
            faces = edge_to_faces[edge]
            print(f"  边 {edge}: 被 {len(faces)} 个面共享")
    
    if nonmanifold_vertices:
        print(f"\n非流形顶点: {sorted(list(nonmanifold_vertices))[:10]}")
    
    return list(nonmanifold_edges), list(nonmanifold_vertices)

def visualize_with_polyscope(V, F, nonmanifold_edges, nonmanifold_vertices):
    """使用polyscope可视化非流形结构"""
    try:
        import polyscope as ps
    except ImportError:
        print("polyscope未安装，跳过可视化")
        print("安装命令: pip install polyscope")
        return
    
    ps.init()
    
    # 注册主网格
    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    ps_mesh.set_color([0.7, 0.7, 0.7])
    
    # 高亮非流形顶点
    if nonmanifold_vertices:
        vertex_colors = np.zeros(len(V))
        vertex_colors[nonmanifold_vertices] = 1.0  # 红色标记
        ps_mesh.add_scalar_quantity("nonmanifold_vertices", vertex_colors, cmap='reds')
        ps_mesh.set_edge_width(1.0)
        ps_mesh.set_edge_color((0, 0, 0))
        
        # 显示非流形顶点为点云
        nonmanifold_points = V[nonmanifold_vertices]
        ps.register_point_cloud("nonmanifold_points", nonmanifold_points, color=[1, 0, 0], radius=0.01)
    
    # 高亮非流形边
    if nonmanifold_edges:
        edge_vertices = []
        for edge in nonmanifold_edges:
            edge_vertices.extend([V[edge[0]], V[edge[1]]])
        
        if edge_vertices:
            edge_vertices = np.array(edge_vertices).reshape(-1, 2, 3)
            ps.register_curve_network("nonmanifold_edges", 
                                    edge_vertices.reshape(-1, 3),
                                    np.arange(len(edge_vertices) * 2).reshape(-1, 2),
                                    color=[1, 0, 0], radius=0.005)
    
    print("\n=== Polyscope 可视化 ===")
    print("红色点: 非流形顶点")
    print("红色线: 非流形边")
    print("按 'q' 退出可视化")
    
    ps.set_ground_plane_mode('none')
    ps.show()

def main():
    parser = argparse.ArgumentParser(description="检测非流形边和顶点")
    parser.add_argument("input", help="输入网格文件")
    parser.add_argument("--no-vis", action="store_true", help="不显示可视化")
    args = parser.parse_args()
    
    # 加载网格
    mesh = trimesh.load(args.input, process=False)
    V, F = mesh.vertices, mesh.faces
    
    # 检测非流形
    nonmanifold_edges, nonmanifold_vertices = detect_nonmanifold(V, F)
    
    # 可视化
    if not args.no_vis and (nonmanifold_edges or nonmanifold_vertices):
        visualize_with_polyscope(V, F, nonmanifold_edges, nonmanifold_vertices)
    elif not nonmanifold_edges and not nonmanifold_vertices:
        print("✅ 未发现非流形问题")
    
    return len(nonmanifold_edges) > 0 or len(nonmanifold_vertices) > 0

if __name__ == "__main__":
    has_problems = main()
    exit(1 if has_problems else 0)
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import argparse

def tri_area(v0, v1, v2):
    """计算三角形面积"""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def diagnose_components(V, F):
    """分析网格的连通分量个数"""
    if len(F) == 0:
        return 0, np.array([])
    edges = set()
    for f in F:
        edges.update([(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
    if not edges:
        return 0, np.array([])
    I, J = zip(*edges)
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(len(V), len(V)))
    A = A + A.T  # 转成无向图
    n_components, labels = connected_components(A)
    return n_components, labels

def compute_mesh_quality(V, F):
    """计算网格质量指标"""
    if len(F) == 0:
        return {"degenerate": 0, "sliver": 0, "min_angle": 90}
    
    areas = np.array([tri_area(V[f[0]], V[f[1]], V[f[2]]) for f in F])
    
    # 计算最小角
    min_angles = []
    for f in F:
        a, b, c = V[f]
        # 边长
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b) 
        ca = np.linalg.norm(a - c)
        
        if ab == 0 or bc == 0 or ca == 0:
            min_angles.append(0)
            continue
            
        # 余弦定理计算角度
        cos_A = (ab**2 + ca**2 - bc**2) / (2 * ab * ca)
        cos_B = (ab**2 + bc**2 - ca**2) / (2 * ab * bc)
        cos_C = (bc**2 + ca**2 - ab**2) / (2 * bc * ca)
        
        cos_A = np.clip(cos_A, -1, 1)
        cos_B = np.clip(cos_B, -1, 1)
        cos_C = np.clip(cos_C, -1, 1)
        
        angles = [np.degrees(np.arccos(x)) for x in [cos_A, cos_B, cos_C]]
        min_angles.append(min(angles))
    
    min_angles = np.array(min_angles)
    
    # 统计
    edge_mean = np.mean([np.linalg.norm(V[F[:, 1]] - V[F[:, 0]]),
                        np.linalg.norm(V[F[:, 2]] - V[F[:, 1]]),
                        np.linalg.norm(V[F[:, 0]] - V[F[:, 2]])])
    area_threshold = 1e-12 * (edge_mean**2)
    
    degenerate_count = np.sum(areas < area_threshold)
    sliver_count = np.sum(min_angles < 5.0)  # 小于5度的瘦长三角形
    
    return {
        "degenerate": degenerate_count,
        "sliver": sliver_count, 
        "min_angle": np.min(min_angles) if len(min_angles) > 0 else 90,
        "mean_angle": np.mean(min_angles) if len(min_angles) > 0 else 60
    }

def clean_mesh_thorough(input_path, output_path, aggressive=False):
    print(f"加载网格文件: {input_path}")
    mesh = trimesh.load(input_path, process=False)  # 不要自动处理
    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    print(f"原始: {len(V)} 顶点, {len(F)} 面")
    
    # 原始质量
    n_before, _ = diagnose_components(V, F)
    quality_before = compute_mesh_quality(V, F)
    print(f"原始连通分量: {n_before}")
    print(f"原始质量 - 退化面: {quality_before['degenerate']}, 瘦长面: {quality_before['sliver']}, 最小角: {quality_before['min_angle']:.1f}°")

    # === 清洗步骤 ===
    
    # Step 1: 合并重复顶点
    print("\n[Step 1] 合并重复顶点...")
    tolerance = 1e-6
    V_rounded = np.round(V / tolerance) * tolerance
    unique_vertices, inverse_indices = np.unique(V_rounded, axis=0, return_inverse=True)
    
    # 重新映射面索引
    F_remapped = inverse_indices[F]
    
    dup_removed = len(V) - len(unique_vertices)
    print(f"移除重复顶点: {dup_removed}")
    V = unique_vertices
    F = F_remapped
    
    # Step 2: 移除退化面
    print("\n[Step 2] 移除退化面...")
    valid_faces = []
    edge_mean = 1.0
    if len(F) > 0:
        edges = np.concatenate([
            np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
            np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1), 
            np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)
        ])
        edge_mean = np.mean(edges) if len(edges) > 0 else 1.0
    
    area_threshold = 1e-12 * (edge_mean**2)
    
    for i, f in enumerate(F):
        # 检查面是否有效
        if len(np.unique(f)) < 3:  # 重复顶点
            continue
        area = tri_area(V[f[0]], V[f[1]], V[f[2]])
        if area > area_threshold:
            valid_faces.append(f)
    
    degenerate_removed = len(F) - len(valid_faces)
    print(f"移除退化面: {degenerate_removed}")
    F = np.array(valid_faces) if valid_faces else np.empty((0, 3), dtype=int)
    
    # Step 3: 移除重复面
    print("\n[Step 3] 移除重复面...")
    if len(F) > 0:
        F_sorted = np.sort(F, axis=1)  # 排序每个面的顶点索引
        unique_faces, unique_idx = np.unique(F_sorted, axis=0, return_index=True)
        F = F[unique_idx]
        duplicate_faces_removed = len(F_sorted) - len(F)
        print(f"移除重复面: {duplicate_faces_removed}")
    
    # Step 4: 移除孤立顶点
    print("\n[Step 4] 移除孤立顶点...")
    if len(F) > 0:
        used_vertices = np.unique(F.flatten())
        vertex_mapping = np.full(len(V), -1, dtype=int)
        vertex_mapping[used_vertices] = np.arange(len(used_vertices))
        
        V = V[used_vertices]
        F = vertex_mapping[F]
        
        isolated_removed = len(vertex_mapping) - len(used_vertices)
        print(f"移除孤立顶点: {isolated_removed}")
    
    # Step 5: 处理质量极差的面 (aggressive模式)
    if aggressive and len(F) > 0:
        print("\n[Step 5] 激进模式：移除极低质量三角形...")
        quality = compute_mesh_quality(V, F)
        
        good_faces = []
        for i, f in enumerate(F):
            a, b, c = V[f]
            # 检查最小角
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            
            if ab == 0 or bc == 0 or ca == 0:
                continue
                
            # 计算最小角
            cos_A = (ab**2 + ca**2 - bc**2) / (2 * ab * ca)
            cos_A = np.clip(cos_A, -1, 1)
            min_angle = np.degrees(np.arccos(cos_A))
            
            if min_angle > 2.0:  # 保留角度大于2度的面
                good_faces.append(f)
        
        low_quality_removed = len(F) - len(good_faces)
        print(f"移除低质量面: {low_quality_removed}")
        F = np.array(good_faces) if good_faces else np.empty((0, 3), dtype=int)
    
    # 创建清洗后的网格
    if len(F) > 0:
        mesh_clean = trimesh.Trimesh(vertices=V, faces=F, process=False)
        
        # 最后做一次基本修复
        mesh_clean.remove_duplicate_faces()
        mesh_clean.remove_degenerate_faces()
        
        # 如果需要，填充小洞
        try:
            mesh_clean.fill_holes()
        except:
            print("填充洞失败，跳过...")
        
    else:
        print("警告：没有有效面剩余！")
        mesh_clean = trimesh.Trimesh(vertices=V, faces=np.empty((0, 3), dtype=int))

    # 最终统计
    n_after, _ = diagnose_components(mesh_clean.vertices, mesh_clean.faces)
    quality_after = compute_mesh_quality(mesh_clean.vertices, mesh_clean.faces)
    
    print("\n====== 清洗报告 ======")
    print(f"顶点数: {len(mesh.vertices)} → {len(mesh_clean.vertices)}")
    print(f"面数:   {len(mesh.faces)} → {len(mesh_clean.faces)}")
    print(f"连通分量: {n_before} → {n_after}")
    print(f"退化面: {quality_before['degenerate']} → {quality_after['degenerate']}")
    print(f"瘦长面: {quality_before['sliver']} → {quality_after['sliver']}")
    print(f"最小角: {quality_before['min_angle']:.1f}° → {quality_after['min_angle']:.1f}°")
    print(f"闭合性: {mesh_clean.is_watertight}")
    print("======================")

    # 保存
    mesh_clean.export(output_path)
    print(f"已保存清洗后网格: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="彻底清洗网格")
    parser.add_argument("input", help="输入 .obj 文件路径")
    parser.add_argument("output", help="输出 .obj 文件路径")
    parser.add_argument("--aggressive", action="store_true", help="激进模式：移除更多低质量面")
    args = parser.parse_args()

    clean_mesh_thorough(args.input, args.output, args.aggressive)
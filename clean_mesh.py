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
    edges = set()
    for f in F:
        edges.update([(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
    I, J = zip(*edges)
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(len(V), len(V)))
    A = A + A.T  # 转成无向图
    n_components, labels = connected_components(A)
    return n_components, labels

def clean_mesh(input_path, output_path):
    print(f"加载网格文件: {input_path}")
    mesh = trimesh.load(input_path, process=False)
    V = mesh.vertices
    F = mesh.faces

    print(f"原始顶点数: {len(V)}, 原始面数: {len(F)}")

    # 清洗前连通性
    n_before, _ = diagnose_components(V, F)
    print(f"清洗前连通分量数: {n_before}")

    # Step 1: 删除退化面（三角形面积过小）
    areas = np.array([tri_area(V[f[0]], V[f[1]], V[f[2]]) for f in F])
    keep_mask = areas > 1e-10
    F_clean = F[keep_mask]
    print(f"删除退化面数: {np.sum(~keep_mask)}")

    # Step 2: 生成新的 mesh 并合并重复点
    mesh_clean = trimesh.Trimesh(vertices=V, faces=F_clean, process=True)

    # 清洗后连通性
    n_after, _ = diagnose_components(mesh_clean.vertices, mesh_clean.faces)
    print(f"清洗后连通分量数: {n_after}")

    # Step 3: 保存
    mesh_clean.export(output_path)
    print(f"已保存清洗后网格: {output_path}")

    # Step 4: 报告
    print("\n====== 清洗报告 ======")
    print(f"原始顶点数: {len(V)} → 清洗后: {len(mesh_clean.vertices)}")
    print(f"原始面数:   {len(F)} → 清洗后: {len(mesh_clean.faces)}")
    print(f"退化面移除: {np.sum(~keep_mask)}")
    print(f"连通分量:   {n_before} → {n_after}")
    print("======================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗含有退化面和重复点的网格")
    parser.add_argument("input", help="输入 .obj 文件路径")
    parser.add_argument("output", help="输出 .obj 文件路径")
    args = parser.parse_args()

    clean_mesh(args.input, args.output)

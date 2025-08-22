import numpy as np
import trimesh as tm
import argparse

parser = argparse.ArgumentParser(description="清洗含有退化面和重复点的网格")
parser.add_argument("mesh_file", help="输入 .obj 文件路径")
args = parser.parse_args()
mesh_file = args.mesh_file

m = tm.load(mesh_file, process=False)  # 不自动修复，先看真相
V = m.vertices.view(np.ndarray)
F = m.faces.view(np.ndarray)

# 1) 面积 & 退化面
areas = m.area_faces
edges_length = getattr(m, 'edges_unique_length', getattr(m, 'edges_length', np.array([1.0])))
Lmean = edges_length.mean() if len(edges_length) else 1.0
eps = 1e-12 * (Lmean**2)
degenerate = np.where(areas < eps)[0]

# 2) 重复顶点（焊接前后对比）
V_round = np.round(V / 1e-15) * 1e-15  # 容差 1e-6
_, idx_unique = np.unique(V_round, axis=0, return_index=True)
dup_verts_count = V.shape[0] - len(idx_unique)

# 3) 瘦长三角形（最小角 & Q 指标）
def triangle_min_angle(va, vb, vc):
    def angle(u, v):
        cu = u/np.linalg.norm(u)
        cv = v/np.linalg.norm(v)
        d = np.clip(cu@cv, -1.0, 1.0)
        return np.degrees(np.arccos(d))
    a = np.asarray(va)
    b = np.asarray(vb)
    c = np.asarray(vc)
    ab, bc, ca = b-a, c-b, a-c
    A = angle(-ab, ca)   # at a
    B = angle(-bc, ab)   # at b
    C = angle(-ca, bc)   # at c
    return min(A,B,C)

mins = []
Qvals = []
for f in F:
    a,b,c = V[f]
    mins.append(triangle_min_angle(a,b,c))
    # Q = (2*sqrt(3)*Area)/sum(edge^2)
    e = np.array([np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)])
    area = np.linalg.norm(np.cross(b-a, c-a))*0.5
    Q = (2.0*np.sqrt(3.0)*area) / np.sum(e**2) if np.sum(e**2)>0 else 0.0
    Qvals.append(Q)

mins = np.array(mins)
Qvals = np.array(Qvals)
sliver_by_angle = np.where(mins < 10.0)[0]
sliver_by_Q = np.where(Qvals < 0.2)[0]

# 4) 连通分量（按面）
face_adj = getattr(m, 'face_adjacency', np.array([]))
if len(face_adj) > 0:
    components = tm.graph.connected_components(face_adj, nodes=np.arange(len(F)))
    num_cc = len(components)
else:
    num_cc = 1

# 5) 非流形边
edges_unique = m.edges_unique
edge_counts = getattr(m, 'edges_unique_counts', 
                     getattr(m, 'edges_unique_face_count', 
                             np.ones(len(edges_unique))))
nonmanifold_edges = edges_unique[np.where(edge_counts > 2)[0]]

# 6) 边界环（开放网格）
boundary_edges = getattr(m, 'edges_boundary', np.array([]))
num_boundary_edges = len(boundary_edges)

# 7) 水密性
is_watertight = getattr(m, 'is_watertight', False)

print("=== Mesh Diagnostics ===")
print(f"Vertices: {len(V)}  Faces: {len(F)}")
print(f"Area min/mean/max: {areas.min():.3e} / {areas.mean():.3e} / {areas.max():.3e}")
print(f"Degenerate faces: {len(degenerate)}")
print(f"Duplicate vertices (tol=1e-15): {dup_verts_count}")
print(f"Min angle (deg): {mins.min():.2f},  %<10°: {(mins<10).mean()*100:.2f}%")
print(f"Quality Q median/5%: {np.median(Qvals):.3f} / {np.percentile(Qvals,5):.3f}")
print(f"Slivers: by angle<10° => {len(sliver_by_angle)}, by Q<0.2 => {len(sliver_by_Q)}")
print(f"Connected components (faces): {num_cc}")
print(f"Non-manifold edges: {len(nonmanifold_edges)}")
print(f"Boundary edges: {num_boundary_edges}   Watertight? {is_watertight}")
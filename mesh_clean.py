import numpy as np
import trimesh

def is_point_triangle(v0, v1, v2, eps=1e-15):
    """
    判断三角形是否退化为一个点 (三个顶点几何上相同)
    """
    return np.allclose(v0, v1, atol=eps) and np.allclose(v1, v2, atol=eps)


def remove_point_degenerate_faces(mesh, eps=1e-15):
    V = mesh.vertices
    F = mesh.faces
    valid_faces = []
    removed_count = 0
    
    for f in F:
        v0, v1, v2 = V[f]
        if is_point_triangle(v0, v1, v2, eps):
            removed_count += 1
            continue
        valid_faces.append(f)
    
    print(f"删除退化到单点的三角形: {removed_count}")
    valid_faces = np.array(valid_faces) if valid_faces else np.empty((0, 3), dtype=int)
    
    # 清理未使用的顶点
    if len(valid_faces) > 0:
        used_vertices = np.unique(valid_faces.flatten())
        new_vertices = V[used_vertices]
        # 重新映射面索引
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        new_faces = np.array([[vertex_map[v] for v in face] for face in valid_faces])
    else:
        new_vertices = np.empty((0, 3))
        new_faces = np.empty((0, 3), dtype=int)
    
    mesh.vertices = new_vertices
    mesh.faces = new_faces
    return mesh




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove degenerate triangles while preserving manifoldness")
    parser.add_argument("input", type=str, help="Input mesh file (.obj/.ply/.stl/...)")
    parser.add_argument("output", type=str, help="Output cleaned mesh file")
    args = parser.parse_args()

    mesh = trimesh.load(args.input, process=False)
    print("Before cleaning:")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces: {len(mesh.faces)}")

    cleaned = remove_point_degenerate_faces(mesh)

    cleaned.export(args.output)
    print(f"Saved cleaned mesh to {args.output}")

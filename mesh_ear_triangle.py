import numpy as np
import polyscope as ps
import argparse

from scipy.spatial import cKDTree


from utility_io import read_two_normal,load_mesh_obj


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
                        tolerance = 1e-6):
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh degenerate viewer')
    parser.add_argument('normal_file', help='normal .normal file with normals')    
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')    
    args = parser.parse_args()


    mesh_file = args.mesh_file
    normal_file = args.normal_file

    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)

    original_vertices, original_edges, original_normals = read_two_normal(normal_file)
    sketch_vertices, sketch_edges, normals = resample_edge_dual_normal(
        original_vertices, original_edges, original_normals)
    
    vertex_mapping = mesh_to_sketch_dict(sketch_vertices, mesh_vertices)
    print(vertex_mapping)


    mesh_vertices = np.asarray(mesh_vertices)
    mesh_faces = np.asarray(mesh_faces)

    # print('vertex_mapping', vertex_mapping)



import numpy as np
import polyscope as ps
import igl

def read_two_normal(filename):
    """
    Read vertices, edges, and dual normal data from an OBJ file.
    Each edge has exactly two normal vectors.
    
    Args:
        filename (str): Path to input OBJ file
        
    Returns:
        V (ndarray): nx3 array of vertex coordinates (x, y, z)
        E (ndarray): mx2 array of edge vertex pairs (0-based indices)
        normals (dict): Dictionary with keys (edge_idx, which_edge) where:
            - edge_idx is the edge index
            - which_edge is 0 or 1 for the first or second normal
            Values are 3D normal vectors (nx, ny, nz)
    """
    V = []  # Vertices
    E = []  # Edges
    normals = {}  # Dictionary to store normals
    normal_vectors = []  # Temporary list to store normal vectors
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'l':  # Edge
                # Convert from 1-based to 0-based indexing
                E.append([int(parts[1])-1, int(parts[2])-1])
            elif parts[0] == 'vn':  # Normal vector
                normal_vectors.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    # Associate normals with edges
    for i, _ in enumerate(E):
        if 2*i < len(normal_vectors) and 2*i+1 < len(normal_vectors):
            normals[(i, 0)] = normal_vectors[2*i]
            normals[(i, 1)] = normal_vectors[2*i + 1]
    
    # Convert lists to numpy arrays
    V = np.array(V)
    E = np.array(E)
    
    print(f"Read from {filename}:")
    print(f"- {len(V)} vertices")
    print(f"- {len(E)} edges")
    print(f"- {len(normal_vectors)} normal vectors")
    
    return V, E, normals

def load_mesh_obj(filename):
    """
    Load mesh from OBJ file (vertices and faces)
    
    Returns:
        vertices: numpy array of vertices (N x 3)
        faces: numpy array of faces (M x 3)
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # Vertex
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
                
            elif parts[0] == 'f':  # Face
                face_indices = []
                for part in parts[1:]:
                    vertex_index = int(part.split('/')[0]) - 1
                    face_indices.append(vertex_index)
                
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    # Split quad into triangles
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])
    
    return np.array(vertices), np.array(faces)

def save_mesh_obj(vertices, faces, filename):
    """
    Save mesh to OBJ file
    """
    with open(filename, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (convert to 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Saved mesh to: {filename}")

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

def find_constraint_vertices_in_mesh(mesh_vertices, constraint_vertices, tolerance=1e-6):
    """
    Find which mesh vertices correspond to constraint vertices
    
    Returns:
        constraint_indices: list of mesh vertex indices that are constrained
    """
    from scipy.spatial import cKDTree
    
    mesh_tree = cKDTree(mesh_vertices)
    constraint_indices = []
    
    print(f"Finding constraint vertices in mesh (tolerance: {tolerance})...")
    
    distances = []
    for i, constraint_vertex in enumerate(constraint_vertices):
        distance, closest_idx = mesh_tree.query(constraint_vertex)
        distances.append(distance)
        
        if distance <= tolerance:
            constraint_indices.append(closest_idx)
        else:
            print(f"Warning: Constraint vertex {i} not found in mesh (distance: {distance:.6f})")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_constraint_indices = []
    for idx in constraint_indices:
        if idx not in seen:
            seen.add(idx)
            unique_constraint_indices.append(idx)
    
    # Statistics
    if distances:
        print(f"Distance statistics:")
        print(f"  Max distance: {max(distances):.6f}")
        print(f"  Mean distance: {np.mean(distances):.6f}")
        print(f"  Median distance: {np.median(distances):.6f}")
    
    print(f"Found {len(unique_constraint_indices)} unique constraint vertices in mesh")
    print(f"Coverage: {len(unique_constraint_indices)}/{len(constraint_vertices)} "
          f"({100*len(unique_constraint_indices)/len(constraint_vertices):.1f}%)")
    
    return unique_constraint_indices

def harmonic_smoothing(mesh_vertices, mesh_faces, constraint_indices):
    """
    Apply harmonic smoothing with k=2
    
    Args:
        mesh_vertices: numpy array of mesh vertices
        mesh_faces: numpy array of mesh faces
        constraint_indices: list of vertex indices to constrain
        
    Returns:
        smoothed_vertices: numpy array of smoothed vertices
    """
    V = np.array(mesh_vertices, dtype=np.float64)
    F = np.array(mesh_faces, dtype=np.int32)
    b = np.array(constraint_indices, dtype=np.int32)
    b = np.unique(b)

    # Check for invalid indices
    if np.any(b >= len(V)) or np.any(b < 0):
        raise ValueError(f"Invalid constraint indices: min={np.min(b)}, max={np.max(b)}, vertices={len(V)}")

    print(f"Applying harmonic smoothing with k=2")
    print(f"Constraints: {len(b)} points")
    print(f"Mesh: {len(V)} vertices, {len(F)} faces")
    print(f"Constraint ratio: {len(b)/len(V):.1%}")

    # Check for isolated vertices
    vertex_valence = np.zeros(len(V), dtype=int)
    for face in F:
        for vi in face:
            vertex_valence[vi] += 1

    isolated = np.where(vertex_valence == 0)[0]
    if len(isolated) > 0:
        print(f"Warning: {len(isolated)} isolated vertices detected")
        # Remove constraints on isolated vertices
        b = np.setdiff1d(b, isolated)
        print(f"Removed {len(isolated)} constraint(s) on isolated vertices")
        if len(b) == 0:
            raise ValueError("All constraint vertices were on isolated vertices")

    # Apply harmonic smoothing with k=2
    V_smoothed = igl.harmonic(V, F, b, V[b], 2)

    if V_smoothed is None:
        raise RuntimeError("Harmonic smoothing failed - returned None")

    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(V_smoothed))
    inf_count = np.sum(np.isinf(V_smoothed))
    
    if nan_count > 0 or inf_count > 0:
        raise RuntimeError(f"Harmonic smoothing produced {nan_count} NaN and {inf_count} Inf values")

    # Calculate displacement statistics
    displacement = np.linalg.norm(V_smoothed - V, axis=1)
    print(f"Smoothing complete:")
    print(f"  Max displacement: {np.max(displacement):.6f}")
    print(f"  Mean displacement: {np.mean(displacement):.6f}")
    print(f"  Std displacement: {np.std(displacement):.6f}")

    return V_smoothed

def simple_viewer(normal_file, mesh_file, target_edge_length=0.05, output_file=None, headless=False):
    """
    Surface smoothing viewer with simplified workflow
    
    Args:
        normal_file: .normal file path
        mesh_file: mesh .obj file path  
        target_edge_length: target edge length for resampling
        output_file: path to save smoothed mesh (optional)
        headless: if True, skip visualization and only save results
    
    Returns:
        tuple: (smoothed_mesh_vertices, sketch_vertices, sketch_edges, normals)
    """
    
    # Step 1: Read original data
    print("=== Reading data ===")
    original_vertices, original_edges, original_normals = read_two_normal(normal_file)
    mesh_vertices, mesh_faces = load_mesh_obj(mesh_file)
    
    # Step 2: Resample processing
    print(f"\n=== Resampling processing ===")
    sketch_vertices, sketch_edges, normals = resample_edge_dual_normal(
        original_vertices, original_edges, original_normals, target_edge_length)
    
    # Step 3: Surface smoothing processing (always applied)
    print(f"\n=== Surface smoothing processing ===")
    
    try:
        # Find constraint vertices
        constraint_indices = find_constraint_vertices_in_mesh(
            mesh_vertices, sketch_vertices, tolerance=1e-6)
        
        if len(constraint_indices) == 0:
            raise ValueError("No constraint vertices found - cannot apply smoothing")
        
        # Apply harmonic smoothing
        smoothed_mesh_vertices = harmonic_smoothing(
            mesh_vertices, mesh_faces, constraint_indices)
        
        print("✓ Harmonic smoothing completed successfully!")
        
        # Save smoothed mesh if output file specified
        if output_file:
            save_mesh_obj(smoothed_mesh_vertices, mesh_faces, output_file)
        
    except Exception as e:
        print(f"✗ Smoothing failed: {e}")
        raise e
    
    # Step 4: Visualization (skip if headless mode)
    if not headless:
        print(f"\n=== Polyscope visualization ===")
        ps.init()
        
        # Add original mesh (gray, opaque)
        ps_mesh_original = ps.register_surface_mesh("original_mesh", mesh_vertices, mesh_faces)
        ps_mesh_original.set_color([0.7, 0.7, 0.7])
        
        # Show smoothed mesh (blue, semi-transparent)
        ps_mesh_smoothed = ps.register_surface_mesh("smoothed_mesh", smoothed_mesh_vertices, mesh_faces)
        ps_mesh_smoothed.set_color([0.4, 0.6, 1.0])  # Blue
        ps_mesh_smoothed.set_transparency(0.8)  # Semi-transparent
        
        # Add resampled sketch vertices (green)
        ps_points = ps.register_point_cloud("sketch_vertices", sketch_vertices)
        ps_points.set_radius(0.003, True)
        ps_points.set_color([0.2, 0.8, 0.2])
        
        # Add resampled sketch edges (green)
        sketch_edges_array = np.array(sketch_edges)
        ps_edges = ps.register_curve_network("sketch_edges", sketch_vertices, sketch_edges_array)
        ps_edges.set_radius(0.003)
        ps_edges.set_color([0.2, 0.8, 0.2])
        
        # Display normal vectors
        if normals:
            normal1_starts = []
            normal1_ends = []
            normal2_starts = []
            normal2_ends = []
            
            normal_scale = 0.02  # Normal vector display length
            
            for edge_idx, edge in enumerate(sketch_edges_array):
                v1_idx, v2_idx = edge
                edge_center = (sketch_vertices[v1_idx] + sketch_vertices[v2_idx]) / 2
                
                # First normal vector (red)
                if (edge_idx, 0) in normals:
                    normal_vec = np.array(normals[(edge_idx, 0)])
                    normal1_starts.append(edge_center)
                    normal1_ends.append(edge_center + normal_vec * normal_scale)
                
                # Second normal vector (orange)
                if (edge_idx, 1) in normals:
                    normal_vec = np.array(normals[(edge_idx, 1)])
                    normal2_starts.append(edge_center)
                    normal2_ends.append(edge_center + normal_vec * normal_scale)
            
            # Display first normal vector group (red)
            if normal1_starts:
                normal1_starts = np.array(normal1_starts)
                normal1_ends = np.array(normal1_ends)
                normal1_edges = np.array([[i, i + len(normal1_starts)] for i in range(len(normal1_starts))])
                normal1_vertices = np.vstack([normal1_starts, normal1_ends])
                
                ps_normals1 = ps.register_curve_network("normals1", normal1_vertices, normal1_edges)
                ps_normals1.set_radius(0.001)
                ps_normals1.set_color([1.0, 0.0, 0.0])
            
            # Display second normal vector group (orange)
            if normal2_starts:
                normal2_starts = np.array(normal2_starts)
                normal2_ends = np.array(normal2_ends)
                normal2_edges = np.array([[i, i + len(normal2_starts)] for i in range(len(normal2_starts))])
                normal2_vertices = np.vstack([normal2_starts, normal2_ends])
                
                ps_normals2 = ps.register_curve_network("normals2", normal2_vertices, normal2_edges)
                ps_normals2.set_radius(0.001)
                ps_normals2.set_color([0.0, 1.0, 0.0])
        
        # Print visualization description
        print("\n=== Visualization description ===")
        print("Gray (opaque): Original mesh surface")
        print("Blue (semi-transparent): Smoothed mesh surface")
        print("Green: Resampled sketch vertices and edges")
        print("Red: First normal vectors")
        print("Green: Second normal vectors")
        print(f"Target edge length: {target_edge_length}")
        if output_file:
            print(f"Smoothed mesh saved to: {output_file}")
        print("Use the left panel in polyscope to control visibility of each element")
        
        ps.set_ground_plane_mode('none')
        ps.show()
    else:
        print(f"\n=== Headless mode ===")
        if output_file:
            print(f"Smoothed mesh saved to: {output_file}")
        else:
            print("No output file specified - results not saved")
    
    return smoothed_mesh_vertices, sketch_vertices, sketch_edges, normals

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Surface smoothing with edge dual normals (Simplified)')
    parser.add_argument('normal_file', help='Normal .obj file with edges and normals')
    parser.add_argument('mesh_file', help='Mesh .obj file with faces')
    parser.add_argument('--target-length', '-t', type=float, default=0.05,
                       help='Target edge length for resampling (default: 0.05)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path for smoothed mesh')
    parser.add_argument('--headless', action='store_true',
                       help='Skip visualization, run in headless mode')
    
    args = parser.parse_args()
    
    # Run simplified processing workflow
    try:
        smoothed_mesh, sketch_verts, sketch_edges, normals = simple_viewer(
            args.normal_file, 
            args.mesh_file, 
            target_edge_length=args.target_length,
            output_file=args.output,
            headless=args.headless
        )
        
        print("✓ Program completed successfully!")
        
    except Exception as e:
        print(f"✗ Program failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
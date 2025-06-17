import numpy as np
from utility_plot_viewer import plot_sketch_data, plot_edge_info
import argparse
from pathlib import Path


def load_sketch_polyline_data(filename):
    """
    Parse OBJ file to extract vertex coordinates and polyline data.
    Args:
        filename (str): File name
    Returns:
        tuple: (V, E, P) where:
        - V: nx3 array of vertex coordinates
        - E: mx2 array of edge vertex indices (no duplicates)
        - P: list of arrays containing vertex indices for each polyline
    """
    vertices = []
    polylines = []
    
    # Read and parse file line by line
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if not parts:
                continue
            
            # Parse vertex coordinates
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            
            # Parse polyline data
            elif parts[0] == 'l':
                # Convert to 0-based indexing and store vertex indices
                polyline = [int(idx) - 1 for idx in parts[1:]]
                polylines.append(np.array(polyline))
    
    # Convert vertices to numpy array
    V = np.array(vertices)
    
    # Extract unique edges from polylines
    edges = set()
    for poly in polylines:
        # Create edges from consecutive vertices in polyline
        for i in range(len(poly) - 1):
            # Sort vertex indices to avoid duplicate edges
            v1, v2 = sorted([poly[i], poly[i + 1]])
            edges.add((v1, v2))
    
    # Convert edges to numpy array
    E = np.array(list(edges))
    
    # Store polylines as list of numpy arrays
    P = polylines
    
    print(f"\nRead from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} unique edges")
    print(f"- {len(polylines)} polylines")
    print()
    
    return V, E, P

def highlight_edges_to_split(V, E, P, edges_to_split, vertex_valence):
    """
    Visualize the sketch with edges to be split highlighted in red
    
    Args:
        V: nx3 numpy array of vertex coordinates 
        E: mx2 numpy array of edge vertex indices (0-based)
        P: list of numpy arrays containing vertex indices for each polyline (0-based)
        edges_to_split: set of edges that will be split
        vertex_valence: dictionary of vertex valences
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all edges in light gray first
    for edge in E:
        v1_idx, v2_idx = edge[0], edge[1]
        v1_coord = V[v1_idx]
        v2_coord = V[v2_idx]
        
        edge_tuple = tuple(sorted([v1_idx, v2_idx]))
        if edge_tuple in edges_to_split:
            # Highlight edges to be split in red
            ax.plot([v1_coord[0], v2_coord[0]], 
                   [v1_coord[1], v2_coord[1]], 
                   [v1_coord[2], v2_coord[2]], 
                   'r-', linewidth=3, label='Edges to split' if edge_tuple == list(edges_to_split)[0] else "")
        else:
            # Regular edges in light gray
            ax.plot([v1_coord[0], v2_coord[0]], 
                   [v1_coord[1], v2_coord[1]], 
                   [v1_coord[2], v2_coord[2]], 
                   'lightgray', linewidth=1)
    
    # Plot vertices with different colors based on valence
    high_valence_vertices = [v for v, val in vertex_valence.items() if val > 2]
    low_valence_vertices = [v for v, val in vertex_valence.items() if val <= 2]
    
    if high_valence_vertices:
        high_valence_coords = V[high_valence_vertices]
        ax.scatter(high_valence_coords[:, 0], 
                  high_valence_coords[:, 1], 
                  high_valence_coords[:, 2], 
                  c='red', s=50, alpha=0.8, label=f'High valence vertices (>{2})')
    
    if low_valence_vertices:
        low_valence_coords = V[low_valence_vertices]
        ax.scatter(low_valence_coords[:, 0], 
                  low_valence_coords[:, 1], 
                  low_valence_coords[:, 2], 
                  c='blue', s=30, alpha=0.6, label=f'Low valence vertices (≤{2})')
    
    # Add vertex labels for high valence vertices
    for v_idx in high_valence_vertices:
        coord = V[v_idx]
        valence = vertex_valence[v_idx]
        ax.text(coord[0], coord[1], coord[2], f'{v_idx}({valence})', 
               fontsize=8, color='darkred')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.title(f'Edges to Split Visualization\n'
             f'{len(edges_to_split)} edges will be split (shown in red)')
    
    print(f"\n=== EDGE SPLITTING PREVIEW ===")
    print(f"Vertex valences: {dict(sorted(vertex_valence.items()))}")
    print(f"Edges to split: {sorted(edges_to_split)} (total: {len(edges_to_split)})")
    print(f"High valence vertices (>2): {sorted(high_valence_vertices)}")
    
    plt.show()
    

def split_edges_preprocessing_compatible(V, E, P):
    """
    Edge splitting that works with your data format (0-based indexing, numpy arrays)
    
    Args:
        V: nx3 numpy array of vertex coordinates 
        E: mx2 numpy array of edge vertex indices (0-based)
        P: list of numpy arrays containing vertex indices for each polyline (0-based)
    
    Returns:
        V_new: updated vertex array with midpoints
        E_new: updated edge array
        P_new: updated polylines with split edges
    """
    
    # Step 1: Calculate valence for each vertex using the provided edges
    vertex_valence = {}
    
    for edge in E:
        v1, v2 = edge[0], edge[1]
        vertex_valence[v1] = vertex_valence.get(v1, 0) + 1
        vertex_valence[v2] = vertex_valence.get(v2, 0) + 1
    
    print(f"Vertex valences: {dict(sorted(vertex_valence.items()))}")
    
    # Step 2: Find edges that need splitting (both endpoints have valence > 2)
    edges_to_split = []
    
    for edge in E:
        v1, v2 = edge[0], edge[1]
        if vertex_valence.get(v1, 0) > 2 and vertex_valence.get(v2, 0) > 2:
            edges_to_split.append(tuple(sorted([v1, v2])))
    
    edges_to_split = set(edges_to_split)
    print(f"Edges to split: {sorted(edges_to_split)} (total: {len(edges_to_split)})")
    
    # Step 3: Create midpoint vertices
    V_new = V.copy()
    edge_to_midpoint = {}
    
    for edge in edges_to_split:
        v1_idx, v2_idx = edge
        v1_coord = V[v1_idx]
        v2_coord = V[v2_idx]
        
        # Calculate midpoint
        midpoint = (v1_coord + v2_coord) / 2
        
        # Add new vertex (0-based indexing)
        new_vertex_idx = len(V_new)
        V_new = np.vstack([V_new, midpoint])
        edge_to_midpoint[edge] = new_vertex_idx
        
        print(f"Created midpoint vertex {new_vertex_idx} for edge {edge} at {midpoint}")
    
    # Step 4: Update polylines
    P_new = []
    
    for poly_idx, polyline in enumerate(P):
        new_polyline = []
        
        for i in range(len(polyline)):
            # Add current vertex
            current_vertex = polyline[i]
            new_polyline.append(current_vertex)
            
            # Check if we need to insert midpoint AFTER this vertex
            if i < len(polyline) - 1:  # Not the last vertex
                next_vertex = polyline[i + 1]
                edge = tuple(sorted([current_vertex, next_vertex]))
                
                if edge in edge_to_midpoint:
                    midpoint_vertex = edge_to_midpoint[edge]
                    new_polyline.append(midpoint_vertex)
                    print(f"  Polyline {poly_idx}: Inserted vertex {midpoint_vertex} between {current_vertex} and {next_vertex}")
        
        P_new.append(np.array(new_polyline))
    
    # Step 5: Update edge list
    edges_new = set()
    for poly in P_new:
        for i in range(len(poly) - 1):
            v1, v2 = sorted([poly[i], poly[i + 1]])
            edges_new.add((v1, v2))
    
    E_new = np.array(list(edges_new))
    
    return V_new, E_new, P_new


def save_split_result(filename, V, P):
    """
    Save the result back to OBJ format (converting back to 1-based indexing)
    """
    with open(filename, 'w') as f:
        # Write vertices
        for vertex in V:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        # Write polylines (convert back to 1-based indexing)
        for polyline in P:
            indices_1based = [str(idx + 1) for idx in polyline]
            f.write(f"l {' '.join(indices_1based)}\n")





def validate_splitting_result(V_orig, P_orig, V_new, P_new):
    """
    Validate that the splitting was done correctly
    """
    print("\n=== VALIDATION ===")
    
    # Check that original vertices are preserved
    if not np.allclose(V_orig, V_new[:len(V_orig)]):
        print("❌ Original vertices were modified!")
        return False
    
    # Check that new vertices are indeed midpoints
    added_vertices = len(V_new) - len(V_orig)
    print(f"✅ Added {added_vertices} new vertices")
    
    # Check polyline integrity
    for i, (orig_poly, new_poly) in enumerate(zip(P_orig, P_new)):
        if len(new_poly) < len(orig_poly):
            print(f"❌ Polyline {i} became shorter!")
            return False
        
        # Check that original vertices are still there in order
        orig_positions = []
        for orig_v in orig_poly:
            try:
                pos = list(new_poly).index(orig_v)
                orig_positions.append(pos)
            except ValueError:
                print(f"❌ Original vertex {orig_v} missing from polyline {i}!")
                return False
        
        # Check that original vertices maintain their relative order
        if orig_positions != sorted(orig_positions):
            print(f"❌ Original vertex order changed in polyline {i}!")
            return False
    
    print("✅ All validation checks passed")
    return True


# Example usage and testing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    
    
    args = parser.parse_args()

    curve_file = args.curve_file

    curve_name = Path(curve_file).stem


    # Load original for comparison
    V_orig, E_orig, P_orig = load_sketch_polyline_data(curve_file)

    plot_sketch_data(V_orig, P_orig)
    # plot_edge_info(V_orig, E_orig)

    # Process the file
    V_new, E_new, P_new = split_edges_preprocessing_compatible(V_orig, E_orig, P_orig)


    plot_sketch_data(V_new, P_new)
    # plot_edge_info(V_new, E_new)
    
    
    # Validate
    validate_splitting_result(V_orig, P_orig, V_new, P_new)

    new_file = 'sketches_split/' + curve_name + '.obj'

    save_split_result(new_file, V_new, P_new)
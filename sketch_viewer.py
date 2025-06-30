import argparse
import numpy as np 

from utility_io import load_normal_data, load_sketch_polyline_data
from utility_viewer_ps import plot_normal_data, plot_cdt_skecth_with_polylines


import argparse
import numpy as np 

def determine_and_load_obj(filename):
    """
    Determines the file type by analyzing content and loads accordingly.
    
    Args:
        filename (str): Path to the input OBJ file
        
    Returns:
        tuple: One of:
            - (V, E, N) for normal data files: vertices, edges, and normals
            - (V, E, P) for sketch polyline files: vertices, edges, and polylines
            - (V, L) for CDT files: vertices and lines
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
            # Check for characteristic markers of each format
            has_normals = 'vn ' in content
            has_polylines = any(line.strip().startswith('l ') and len(line.strip().split()) > 3 
                              for line in content.splitlines())
            
            if has_normals:
                print(f"Detected normal data format in {filename}")
                return load_normal_data(filename), "normal"
            elif has_polylines:
                print(f"Detected sketch polyline format in {filename}")
                return load_sketch_polyline_data(filename), "polyline"
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Could not find file '{filename}'")
    except Exception as e:
        raise ValueError(f"Error processing file '{filename}': {str(e)}")

        
def analyze_edge_issues(V, E, zero_threshold=1e-15, near_zero_threshold=1e-10):
    """
    Analyze mesh for zero-length edges, near-zero length edges, and duplicate vertices.
    
    Args:
        V (np.ndarray): Vertex array of shape (n_vertices, 3)
        E (np.ndarray): Edge array of shape (n_edges, 2) containing vertex indices
        zero_threshold (float): Threshold for considering edge length as exactly zero
        near_zero_threshold (float): Threshold for considering edge length as near-zero
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Find zero and near-zero length edges
    zero_edges = []
    near_zero_edges = []
    
    for i, edge in enumerate(E):
        v1, v2 = V[edge[0]], V[edge[1]]
        length = np.linalg.norm(v2 - v1)
        
        if length <= zero_threshold:
            zero_edges.append({
                'edge_index': i,
                'vertices': (int(edge[0]), int(edge[1])),
                'length': float(length)
            })
        elif length <= near_zero_threshold:
            near_zero_edges.append({
                'edge_index': i,
                'vertices': (int(edge[0]), int(edge[1])),
                'length': float(length)
            })
            
    # Find duplicate vertices
    duplicate_vertices = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if np.allclose(V[i], V[j], atol=zero_threshold):
                duplicate_vertices.append({
                    'vertex_pair': (i, j),
                    'coordinates': V[i].tolist()
                })
                
    return {
        "zero_length_edges": {
            "count": len(zero_edges),
            "threshold": zero_threshold,
            "edges": zero_edges
        },
        "near_zero_length_edges": {
            "count": len(near_zero_edges),
            "threshold": near_zero_threshold,
            "edges": near_zero_edges
        },
        "duplicate_vertices": {
            "count": len(duplicate_vertices),
            "threshold": zero_threshold,
            "pairs": duplicate_vertices
        }
    }
    
def print_edge_issues(analysis):
    """
    Print mesh analysis results in a readable format.
    
    Args:
        analysis (dict): Analysis results from analyze_edge_issues function
    """
    print("\n=== Edge Issues Analysis ===\n")
    
    # Zero length edges
    zero = analysis["zero_length_edges"]
    print(f"Zero Length Edges (threshold: {zero['threshold']:.2e}):")
    print(f"  Count: {zero['count']}")
    if zero['count'] > 0:
        print("  Details:")
        for edge in zero['edges']:
            print(f"    Edge {edge['edge_index']}: vertices {edge['vertices']}, length {edge['length']:.2e}")
            
    # Near-zero length edges
    near_zero = analysis["near_zero_length_edges"]
    print(f"\nNear-Zero Length Edges (threshold: {near_zero['threshold']:.2e}):")
    print(f"  Count: {near_zero['count']}")
    if near_zero['count'] > 0:
        print("  Details:")
        for edge in near_zero['edges']:
            print(f"    Edge {edge['edge_index']}: vertices {edge['vertices']}, length {edge['length']:.2e}")
            
    # Duplicate vertices
    dupes = analysis["duplicate_vertices"]
    print(f"\nDuplicate Vertices (threshold: {dupes['threshold']:.2e}):")
    print(f"  Count: {dupes['count']}")
    if dupes['count'] > 0:
        print("  Details:")
        for pair in dupes['pairs']:
            print(f"    Vertices {pair['vertex_pair']}: coordinates {pair['coordinates']}")
            
    print("\n=========================\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View sketch file (normal, polyline, or CDT format)')
    parser.add_argument('sketch_file', nargs='?', help='Sketch file to view.')
    args = parser.parse_args()

    sketch_file = args.sketch_file

    result, file_type = determine_and_load_obj(sketch_file)

    # Initialize variables
    V = E = N = P = None

    # Unpack based on file type
    if file_type == "normal":
        V, E, N = result
    elif file_type == "polyline":
        V, E, P = result

    print('Vertices shape:', V.shape if V is not None else None)
    print('Edges shape:', E.shape if E is not None else None)
    if N is not None:
        print('Normals shape:', N.shape)
    if P is not None:
        print('Number of polylines:', len(P))
    
    analysis = analyze_edge_issues(V, E)
    print_edge_issues(analysis)
    
    # Plot based on file type
    if file_type == "normal":
        plot_normal_data(V, E, N)
    elif file_type == "polyline":
        plot_cdt_skecth_with_polylines(V, E, P)  

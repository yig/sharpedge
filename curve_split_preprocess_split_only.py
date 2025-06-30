"""
curve_split_preprocess.py
Split polyline on high valence vertices.
So that the polyline become separate polylines.
"""
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

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
    
    print(f"Read from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} unique edges")
    print(f"- {len(polylines)} polylines")
    print()
    
    return V, E, P

def build_vertex_to_edges_map(edges):
    """
    Create a mapping from each vertex to all edges that contain it.
    Parameters:
        edges: (m,2) array of edge vertex index pairs
    Returns:
        dict: Mapping from vertex index to list of edge indices
    """
    vertex_to_edges = defaultdict(list)
    for edge_idx, edge in enumerate(edges):
        # Add this edge to both of its vertices' lists
        vertex_to_edges[edge[0]].append(edge_idx)
        vertex_to_edges[edge[1]].append(edge_idx)
    
    for vertex_idx in vertex_to_edges:
        assert len(vertex_to_edges[vertex_idx]) == len(set(vertex_to_edges[vertex_idx])), \
            f"Vertex {vertex_idx} has duplicate edge entries"
    
    return vertex_to_edges

def find_high_valence_vertices(edges, min_valence=3):
    """
    Find vertices with valence >= min_valence.
    Parameters:
        edges: (m,2) array of edge vertex index pairs
        min_valence: minimum valence to consider as "high"
    Returns:
        list: vertex indices with high valence
    """
    vertex_to_edges = build_vertex_to_edges_map(edges)
    high_valence_vertices = []
    
    for vertex_idx, edge_list in vertex_to_edges.items():
        if len(edge_list) >= min_valence:
            high_valence_vertices.append(vertex_idx)
    
    return high_valence_vertices

def split_polylines_at_vertices(polylines, split_vertices):
    """
    Split polylines at specified vertices.
    Parameters:
        polylines: list of numpy arrays containing vertex indices
        split_vertices: list of vertex indices to split at
    Returns:
        list: new polylines after splitting
    """
    split_set = set(split_vertices)
    new_polylines = []
    
    for poly in polylines:
        if len(poly) < 2:
            continue
            
        # Find split points in this polyline
        split_points = []
        for i, vertex in enumerate(poly):
            if vertex in split_set and 0 < i < len(poly) - 1:  # Don't split at endpoints
                split_points.append(i)
        
        if not split_points:
            # No splits needed, keep original polyline
            new_polylines.append(poly)
        else:
            # Split the polyline
            start_idx = 0
            for split_idx in split_points:
                # Create segment from start to split point (inclusive)
                segment = poly[start_idx:split_idx + 1]
                if len(segment) >= 2:
                    new_polylines.append(segment)
                start_idx = split_idx  # Next segment starts at split point
            
            # Add final segment
            final_segment = poly[start_idx:]
            if len(final_segment) >= 2:
                new_polylines.append(final_segment)
    
    return new_polylines

def save_polylines_to_obj(vertices, polylines, filename):
    """
    Save vertices and polylines to OBJ file.
    Parameters:
        vertices: (n,3) array of vertex coordinates
        polylines: list of numpy arrays containing vertex indices
        filename: output filename
    """
    with open(filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write polylines (convert back to 1-based indexing)
        for poly in polylines:
            if len(poly) >= 2:
                indices = [str(idx + 1) for idx in poly]
                f.write(f"l {' '.join(indices)}\n")

def plot_polylines_3d(vertices, polylines, title="Polylines", high_valence_vertices=None):
    """
    Plot polylines in 3D.
    Parameters:
        vertices: (n,3) array of vertex coordinates
        polylines: list of numpy arrays containing vertex indices
        title: plot title
        high_valence_vertices: list of vertex indices to highlight
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot polylines
    colors = plt.cm.tab10(np.linspace(0, 1, len(polylines)))
    for i, poly in enumerate(polylines):
        if len(poly) >= 2:
            coords = vertices[poly]
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                   color=colors[i % len(colors)], linewidth=2, alpha=0.7)
    
    # Highlight high valence vertices
    if high_valence_vertices:
        high_val_coords = vertices[high_valence_vertices]
        ax.scatter(high_val_coords[:, 0], high_val_coords[:, 1], high_val_coords[:, 2],
                  c='red', s=100, alpha=0.8, label=f'High valence vertices ({len(high_valence_vertices)})')
        ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    
def analyze_polylines(vertices, edges, polylines):
    """
    Analyze polylines and provide statistics.
    """
    vertex_to_edges = build_vertex_to_edges_map(edges)
    
    print("Polyline Analysis:")
    print(f"- Total vertices: {len(vertices)}")
    print(f"- Total edges: {len(edges)}")
    print(f"- Total polylines: {len(polylines)}")
    
    # Valence statistics
    valences = [len(edge_list) for edge_list in vertex_to_edges.values()]
    print(f"- Vertex valence statistics:")
    print(f"  - Min: {min(valences) if valences else 0}")
    print(f"  - Max: {max(valences) if valences else 0}")
    print(f"  - Mean: {np.mean(valences):.2f}")
    
    # High valence vertices
    high_val_vertices = find_high_valence_vertices(edges, min_valence=3)
    print(f"- High valence vertices (≥3): {len(high_val_vertices)}")
    
    # Polyline length statistics
    poly_lengths = [len(poly) for poly in polylines]
    print(f"- Polyline length statistics:")
    print(f"  - Min: {min(poly_lengths) if poly_lengths else 0}")
    print(f"  - Max: {max(poly_lengths) if poly_lengths else 0}")
    print(f"  - Mean: {np.mean(poly_lengths):.2f}")
    print()
    
    return high_val_vertices

def main():
    parser = argparse.ArgumentParser(description='Split polylines at high valence vertices')
    parser.add_argument('curve_file', help='Input OBJ file with polylines')
    parser.add_argument('output_file', nargs='?', help='Output OBJ file')
    parser.add_argument('--min_valence', type=int, default=2,
                       help='Minimum valence to consider as "high" (default: 3)')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview without splitting')
    parser.add_argument('--plot', action='store_true',
                       help='Show 3D plots of before/after')
    
    args = parser.parse_args()
    
    if not args.output_file:
        curve_path = Path(args.curve_file)
        args.output_file = curve_path.stem + '_split' + curve_path.suffix
    
    # Load original data
    V_orig, E_orig, P_orig = load_sketch_polyline_data(args.curve_file)
    
    # Analyze original polylines
    print("=== ORIGINAL POLYLINES ===")
    high_val_vertices = analyze_polylines(V_orig, E_orig, P_orig)
    
    if args.preview:
        print("PREVIEW MODE: No files will be modified")
        print(f"Would split at {len(high_val_vertices)} high valence vertices")
        print(f"High valence vertex indices: {high_val_vertices}")
        
        if args.plot:
            plot_polylines_3d(V_orig, P_orig, "Original Polylines (Preview)", high_val_vertices)
            plt.show()
        return
    
    # Split polylines at high valence vertices
    P_split = split_polylines_at_vertices(P_orig, high_val_vertices)
    
    # Create edges for split polylines
    edges_split = set()
    for poly in P_split:
        for i in range(len(poly) - 1):
            v1, v2 = sorted([poly[i], poly[i + 1]])
            edges_split.add((v1, v2))
    E_split = np.array(list(edges_split))
    
    # Analyze split polylines
    print("=== SPLIT POLYLINES ===")
    analyze_polylines(V_orig, E_split, P_split)
    
    # Save split polylines
    save_polylines_to_obj(V_orig, P_split, args.output_file)
    print(f"Split polylines saved to: {args.output_file}")
    
    # Show plots if requested
    if args.plot:
        # Plot original
        plot_polylines_3d(V_orig, P_orig, "Original Polylines", high_val_vertices)
        
        # Plot split
        plt.figure()
        plot_polylines_3d(V_orig, P_split, "Split Polylines")
        
        plt.show()

if __name__ == "__main__":
    main()
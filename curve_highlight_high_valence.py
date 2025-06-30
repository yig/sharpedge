from utility_io import load_sketch_polyline_data
import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def build_vertex_to_edges_map(edges):
    '''
    Create a mapping from each vertex to all edges that contain it.
    Parameters:
    edges: (m,2) array of edge vertex index pairs
    Returns:
    dict: Mapping from vertex index to list of edge indices
    '''
    vertex_to_edges = defaultdict(list)
    for edge_idx, edge in enumerate(edges):
        # Add this edge to both of its vertices' lists
        vertex_to_edges[edge[0]].append(edge_idx)
        vertex_to_edges[edge[1]].append(edge_idx)
    
    for vertex_idx in vertex_to_edges:
        assert len(vertex_to_edges[vertex_idx]) == len(set(vertex_to_edges[vertex_idx])), \
            f"Vertex {vertex_idx} has duplicate edge entries"
    
    return vertex_to_edges

def find_high_valence_vertices(vertex_to_edges, min_valence=3):
    """
    Find vertices with high valence (many edges meeting).
    Parameters:
    vertex_to_edges: dict mapping vertex index to list of edge indices
    min_valence: minimum valence to consider as "high"
    Returns:
    dict: mapping from vertex index to its valence
    """
    high_valence = {}
    for vertex_idx, edge_list in vertex_to_edges.items():
        valence = len(edge_list)
        if valence >= min_valence:
            high_valence[vertex_idx] = valence
    
    return high_valence

def plot_polylines_with_valence(vertices, edges, polylines, high_valence_vertices=None, 
                                title="Polylines with High-Valence Vertices", 
                                show_vertex_labels=False):
    """
    Plot 3D polylines with highlighted high-valence vertices.
    Parameters:
    vertices: (n,3) array of vertex coordinates
    edges: (m,2) array of edge vertex index pairs
    polylines: list of polylines (lists of vertex indices)
    high_valence_vertices: dict mapping vertex index to valence
    title: plot title
    show_vertex_labels: whether to show vertex index labels
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot polylines
    if polylines is not None and len(polylines) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(polylines)))
        
        for i, polyline in enumerate(polylines):
            if len(polyline) < 2:
                continue
            
            # Get coordinates for this polyline
            polyline_coords = vertices[polyline]
            
            ax.plot(polyline_coords[:, 0], polyline_coords[:, 1], polyline_coords[:, 2], 
                   color=colors[i], linewidth=2, alpha=0.7, label=f'Polyline {i+1}')
    
    # Plot edges if no polylines or as backup
    elif edges is not None and len(edges) > 0:
        print("Plotting individual edges (no polylines found)")
        for edge in edges:
            edge_coords = vertices[edge]
            ax.plot(edge_coords[:, 0], edge_coords[:, 1], edge_coords[:, 2], 
                   color='blue', linewidth=1, alpha=0.6)
    
    # Plot all vertices as small points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              color='black', s=15, alpha=0.4, zorder=3, label='Vertices')
    
    # Highlight high-valence vertices
    if high_valence_vertices:
        intersection_coords = []
        intersection_valences = []
        
        for vertex_idx, valence in high_valence_vertices.items():
            intersection_coords.append(vertices[vertex_idx])
            intersection_valences.append(valence)
        
        if intersection_coords:
            intersection_coords = np.array(intersection_coords)
            
            # Size markers based on valence
            sizes = [80 + 30 * (v - 3) for v in intersection_valences]
            
            ax.scatter(intersection_coords[:, 0], intersection_coords[:, 1], 
                      intersection_coords[:, 2], 
                      c='red', s=sizes, zorder=10, alpha=0.9, 
                      label='High-Valence Vertices', marker='o', 
                      edgecolors='darkred', linewidths=2)
            
            # Add valence labels
            for coord, valence in zip(intersection_coords, intersection_valences):
                ax.text(coord[0], coord[1], coord[2], f'{valence}', 
                       fontsize=12, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11)
    
    # Add vertex labels if requested
    if show_vertex_labels:
        for i, vertex in enumerate(vertices):
            ax.text(vertex[0], vertex[1], vertex[2], f'{i}', 
                   fontsize=8, alpha=0.6, color='gray')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                         vertices[:, 1].max() - vertices[:, 1].min(),
                         vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax

def analyze_and_plot_valence(vertices, edges, polylines, min_valence=3, 
                            show_vertex_labels=False):
    """
    Complete analysis and visualization of vertex valence.
    """
    print(f"Data summary:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} edges")
    print(f"- {len(polylines) if polylines else 0} polylines")
    
    # Build vertex-to-edges mapping
    vertex_to_edges = build_vertex_to_edges_map(edges)
    
    # Find high-valence vertices
    high_valence_vertices = find_high_valence_vertices(vertex_to_edges, min_valence)
    
    print(f"\nValence analysis:")
    print(f"- Found {len(high_valence_vertices)} vertices with valence >= {min_valence}")
    
    if high_valence_vertices:
        print("High-valence vertices:")
        for vertex_idx, valence in sorted(high_valence_vertices.items()):
            print(f"  Vertex {vertex_idx}: valence {valence}")
    
    # Create visualization
    fig, ax = plot_polylines_with_valence(
        vertices, edges, polylines, high_valence_vertices,
        title=f"Polylines with High-Valence Vertices (valence >= {min_valence})",
        show_vertex_labels=show_vertex_labels
    )
    
    return vertex_to_edges, high_valence_vertices, fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and visualize vertex valence in polylines')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('--min-valence', type=int, default=3,
                       help='Minimum valence to highlight (default: 3)')
    parser.add_argument('--show-labels', action='store_true',
                       help='Show vertex index labels')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    curve_file = args.curve_file
    
    if not curve_file:
        print("Please provide a curve file")
        print("Usage: python script.py curve_file.obj [--min-valence 3] [--show-labels] [--no-plot]")
        exit(1)
    
    try:
        # Load data
        V, E, P = load_sketch_polyline_data(curve_file)
        
        # Analyze and plot
        vertex_to_edges, high_valence_vertices, fig, ax = analyze_and_plot_valence(
            V, E, P, 
            min_valence=args.min_valence,
            show_vertex_labels=args.show_labels
        )
        
        if not args.no_plot:
            plt.show()
            print("\nVisualization complete. Red circles show high-valence vertices with their valence numbers.")
        else:
            print("Plotting disabled with --no-plot flag")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{curve_file}'")
    except Exception as e:
        print(f"Error loading data: {e}")
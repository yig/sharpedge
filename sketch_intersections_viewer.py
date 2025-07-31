import argparse
import numpy as np 

from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_vertex_valence(E):
    """
    Calculate the valence (degree) of each vertex based on edges.
    
    Args:
        E: mx2 numpy array of edge vertex indices
        
    Returns:
        dict: Dictionary mapping vertex index to its valence
    """
    vertex_valence = {}
    
    for edge in E:
        v1, v2 = edge[0], edge[1]
        vertex_valence[v1] = vertex_valence.get(v1, 0) + 1
        vertex_valence[v2] = vertex_valence.get(v2, 0) + 1
    
    return vertex_valence

def plot_intersection_vertices(V, E, P, valence_threshold=2, show_plot=True):
    """
    Plot the sketch with intersection vertices (high valence) highlighted.
    
    Args:
        V: nx3 numpy array of vertex coordinates
        E: mx2 numpy array of edge vertex indices (0-based)
        P: list of numpy arrays containing vertex indices for each polyline (0-based)
        valence_threshold: int, vertices with valence > threshold are considered intersections
        show_plot: bool, whether to display the plot (default: True)
    
    Returns:
        fig: matplotlib figure object
        intersection_vertices: list of intersection vertex indices
    """
    # Calculate vertex valences
    vertex_valence = calculate_vertex_valence(E)
    
    # Identify intersection vertices (high valence)
    intersection_vertices = [v for v, val in vertex_valence.items() if val > valence_threshold]
    regular_vertices = [v for v, val in vertex_valence.items() if val <= valence_threshold]
    

    fig = plt.figure(figsize=(8,8))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')

    
    # Plot polylines with consistent colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(P)))  # Generate distinct colors
    
    for poly_idx, polyline in enumerate(P):
        color = colors[poly_idx]
        points = V[polyline]
        
        # Plot the polyline
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 
               color=color, linewidth=2, alpha=0.8, 
               label=f'Polyline {poly_idx}')
        
        # Plot vertices along the polyline
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  color=color, s=10, alpha=0.7)
    
        # Add a text label at the midpoint of the polyline
        # mid_idx = len(points) // 2
        # mid_point = points[mid_idx]
        # ax.text(mid_point[0], mid_point[1], mid_point[2], 
        #         f'{poly_idx}', fontsize=9, color=color)

    # Note: Regular vertices are already plotted as part of polylines above
    
    # Plot intersection vertices (high valence) in red - HIGHLIGHTED
    if intersection_vertices:
        intersection_coords = V[intersection_vertices]
        # Highlight intersection vertices with larger red markers
        ax.scatter(intersection_coords[:, 0],
                  intersection_coords[:, 1],
                  intersection_coords[:, 2],
                  c='red', s=50, alpha=1.0, 
                  marker='o', edgecolors='darkred', linewidth=3,
                  label=f'Intersection vertices (>{valence_threshold})', zorder=10)
        
        # Add labels for intersection vertices
        for v_idx in intersection_vertices:
            coord = V[v_idx]
            valence = vertex_valence[v_idx]
            ax.text(coord[0] + 0.02, coord[1] + 0.02, coord[2] + 0.02, 
                   f'{v_idx}({valence})',
                   fontsize=10, color='darkred', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # plt.title(f'Intersection Vertices Visualization\n'
    #           f'{len(intersection_vertices)} intersection vertices found '
    #           f'(valence > {valence_threshold})')
    
    # Print summary
    print(f"\n=== INTERSECTION VERTICES ANALYSIS ===")
    print(f"Total vertices: {len(V)}")
    print(f"Valence threshold: {valence_threshold}")
    print(f"Intersection vertices: {len(intersection_vertices)}")
    print(f"Regular vertices: {len(regular_vertices)}")
    print(f"\nIntersection vertex details:")
    for v_idx in sorted(intersection_vertices):
        coord = V[v_idx]
        valence = vertex_valence[v_idx]
        print(f"  Vertex {v_idx}: valence={valence}, coord=({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
    
    plt.axis('off')
    plt.axis('equal')
    if show_plot:
        plt.show()
    
    return fig, intersection_vertices



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View sketch file (normal, polyline, or CDT format)')
    parser.add_argument('sketch_file', nargs='?', help='Sketch file to view.')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot (for batch mode).')
    args = parser.parse_args()

    sketch_file = args.sketch_file
    show_plot = not args.no_show

    V, E, P = load_sketch_polyline_data(sketch_file)
    # plot_sketch_data(V, P)

    fig, intersections = plot_intersection_vertices(V, E, P, valence_threshold=2, show_plot=show_plot)

    if args.no_show:
        fig.savefig("intersection_vertices.png", dpi=300, bbox_inches='tight')

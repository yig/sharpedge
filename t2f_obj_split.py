import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_obj_file(content):
    """Parse OBJ file content to extract vertices and polylines."""
    vertices = []
    polylines = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('v '):
            # Parse vertex: v x y z
            coords = list(map(float, line.split()[1:4]))
            vertices.append(coords)
        elif line.startswith('l '):
            # Parse polyline: l v1 v2 v3 ... (1-indexed)
            indices = list(map(int, line.split()[1:]))
            # Convert to 0-indexed
            indices = [i - 1 for i in indices]
            polylines.append(indices)
    
    return np.array(vertices), polylines

def consolidate_nearby_vertices(vertices, polylines, tolerance=1e-7):
    """Consolidate vertices that are within tolerance distance."""
    n = len(vertices)
    vertex_map = {}  # Maps old index to new index
    unique_vertices = []
    
    print(f"Consolidating vertices with tolerance: {tolerance}")
    
    for i in range(n):
        # Check if this vertex is close to any existing unique vertex
        found_match = False
        for j, unique_vertex in enumerate(unique_vertices):
            dist = np.linalg.norm(vertices[i] - unique_vertex)
            if dist < tolerance:
                vertex_map[i] = j
                found_match = True
                break
        
        if not found_match:
            vertex_map[i] = len(unique_vertices)
            unique_vertices.append(vertices[i])
    
    # Update polylines with new indices
    updated_polylines = []
    for polyline in polylines:
        new_polyline = [vertex_map[old_idx] for old_idx in polyline]
        
        # Remove consecutive duplicate indices in the polyline
        filtered_polyline = []
        for idx in new_polyline:
            if not filtered_polyline or idx != filtered_polyline[-1]:
                filtered_polyline.append(idx)
        
        # Only keep polylines with at least 2 unique vertices
        if len(filtered_polyline) > 1:
            updated_polylines.append(filtered_polyline)
    
    consolidated_count = len(vertices) - len(unique_vertices)
    if consolidated_count > 0:
        print(f"Consolidated {consolidated_count} nearby vertices")
    
    return np.array(unique_vertices), updated_polylines, vertex_map

def find_high_valence_vertices(vertices, polylines, min_valence=3):
    """Find vertices where more than min_valence-1 polylines meet."""
    vertex_valence = defaultdict(set)  # vertex_idx -> set of polyline indices
    vertex_positions = defaultdict(list)  # vertex_idx -> list of (polyline_idx, position)
    
    # Count how many polylines use each vertex
    for polyline_idx, polyline in enumerate(polylines):
        for pos, vertex_idx in enumerate(polyline):
            vertex_valence[vertex_idx].add(polyline_idx)
            vertex_positions[vertex_idx].append((polyline_idx, pos))
    
    # Find high-valence vertices (intersections)
    high_valence_vertices = {}
    for vertex_idx, polyline_set in vertex_valence.items():
        valence = len(polyline_set)
        if valence >= min_valence:
            high_valence_vertices[vertex_idx] = valence
            print(f"High-valence vertex {vertex_idx + 1} (1-indexed): {valence} polylines meeting")
    
    return high_valence_vertices, vertex_positions

def split_polylines_at_high_valence(polylines, high_valence_vertices, vertex_positions):
    """Split polylines at high-valence vertices."""
    split_polylines = []
    
    for polyline_idx, polyline in enumerate(polylines):
        # Find split points in this polyline
        split_points = []
        
        for pos, vertex_idx in enumerate(polyline):
            if vertex_idx in high_valence_vertices:
                # Always split at high-valence vertices that are interior points
                # or endpoints that connect to other polylines
                if pos == 0 or pos == len(polyline) - 1:
                    # Endpoint - check if it connects to other polylines
                    other_polylines = [
                        p_idx for p_idx, p_pos in vertex_positions[vertex_idx]
                        if p_idx != polyline_idx
                    ]
                    if other_polylines:
                        split_points.append(pos)
                else:
                    # Interior point - always split
                    split_points.append(pos)
        
        # If no split points, keep original polyline
        if not split_points:
            split_polylines.append(polyline)
            continue
        
        # Split the polyline
        segments = []
        current_segment = []
        
        for pos, vertex_idx in enumerate(polyline):
            current_segment.append(vertex_idx)
            
            # If this is a split point and we have at least 2 vertices
            if pos in split_points and len(current_segment) >= 2:
                segments.append(current_segment[:])  # Save current segment
                current_segment = [vertex_idx]  # Start new segment with split vertex
        
        # Add final segment if it has at least 2 vertices
        if len(current_segment) >= 2:
            segments.append(current_segment)
        
        # Add all valid segments
        for segment in segments:
            if len(segment) >= 2:
                split_polylines.append(segment)
    
    return split_polylines

def plot_polylines(vertices, polylines, high_valence_vertices=None, title="Polylines Visualization", 
                   show_vertex_labels=False):
    """Plot 3D polylines with highlighted high-valence vertices."""
    # Always use 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot polylines
    colors = plt.cm.tab10(np.linspace(0, 1, len(polylines)))
    
    for i, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        
        # Get coordinates for this polyline
        polyline_coords = vertices[polyline]
        
        ax.plot(polyline_coords[:, 0], polyline_coords[:, 1], polyline_coords[:, 2], 
               color=colors[i], linewidth=2, alpha=0.7, label=f'Polyline {i+1}')
    
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
            sizes = [50 + 20 * (v - 3) for v in intersection_valences]
            
            scatter = ax.scatter(intersection_coords[:, 0], intersection_coords[:, 1], 
                               intersection_coords[:, 2], 
                               c='red', s=sizes, zorder=5, alpha=0.8, 
                               label='High-Valence Vertices', marker='o', edgecolors='darkred')
            
            # Add valence labels
            for i, (coord, valence) in enumerate(zip(intersection_coords, intersection_valences)):
                ax.text(coord[0], coord[1], coord[2], f'{valence}', 
                       fontsize=10, fontweight='bold', color='white',
                       ha='center', va='center')
    
    # Plot all vertices as small points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              color='black', s=10, alpha=0.3, zorder=3)
    
    # Add vertex labels if requested
    if show_vertex_labels:
        for i, vertex in enumerate(vertices):
            ax.text(vertex[0], vertex[1], vertex[2], f'{i+1}', fontsize=8, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend (but limit to reasonable number of entries)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 15:  # Only show legend if not too many polylines
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax

def write_obj_file(vertices, polylines, output_filename):
    """Write vertices and polylines back to OBJ format."""
    with open(output_filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write polylines
        for polyline in polylines:
            # Convert back to 1-indexed
            indices_1based = [str(i + 1) for i in polyline]
            f.write(f"l {' '.join(indices_1based)}\n")

def process_obj_file(input_content, tolerance=1e-7, min_valence=3, plot_results=True):
    """Main function to process OBJ file and split curves at high-valence intersections."""
    # Parse the input
    vertices, polylines = parse_obj_file(input_content)
    
    print(f"Original: {len(vertices)} vertices, {len(polylines)} polylines")
    
    # Step 1: Consolidate nearby vertices
    vertices, polylines, vertex_map = consolidate_nearby_vertices(vertices, polylines, tolerance)
    print(f"After consolidation: {len(vertices)} vertices, {len(polylines)} polylines")
    
    # Plot original (consolidated) polylines
    if plot_results:
        fig1, ax1 = plot_polylines(vertices, polylines, title="Consolidated Polylines")
        plt.show()
    
    # Step 2: Find high-valence vertices
    high_valence_vertices, vertex_positions = find_high_valence_vertices(vertices, polylines, min_valence)
    
    if not high_valence_vertices:
        print("No high-valence vertices found - no splitting needed")
        return vertices, polylines, {}, vertex_map
    
    print(f"Found {len(high_valence_vertices)} high-valence vertices")
    
    # Plot with highlighted high-valence vertices
    if plot_results:
        fig2, ax2 = plot_polylines(
            vertices, polylines, high_valence_vertices,
            title="High-Valence Vertices (Red circles with valence numbers)"
        )
        plt.show()
    
    # Step 3: Split polylines at high-valence vertices
    split_polylines = split_polylines_at_high_valence(polylines, high_valence_vertices, vertex_positions)
    
    print(f"After splitting: {len(vertices)} vertices, {len(split_polylines)} polylines")
    print(f"Split into {len(split_polylines) - len(polylines)} additional segments")
    
    # Plot final result
    if plot_results:
        fig3, ax3 = plot_polylines(
            vertices, split_polylines, high_valence_vertices,
            title="Final Result: Split Polylines at High-Valence Vertices"
        )
        plt.show()
    
    return vertices, split_polylines, high_valence_vertices, vertex_map

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split polylines at high-valence intersections')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('output_file', nargs='?', help='The curve sketch to write.')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--tolerance', type=float, default=1e-7, 
                       help='Tolerance for vertex consolidation (default: 1e-7)')
    parser.add_argument('--min-valence', type=int, default=3,
                       help='Minimum valence to consider as intersection (default: 3)')
    args = parser.parse_args()

    input_file = args.curve_file
    output_file = args.output_file
    plot_enabled = not args.no_plot

    if not input_file:
        print("Please provide an input OBJ file")
        print("Usage: python script.py input.obj [output.obj] [options]")
        exit(1)
    
    if not output_file:
        output_file = "output_split.obj"

    try:
        with open(input_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        exit(1)
    
    # Process the file
    vertices, split_polylines, high_valence_vertices, vertex_map = process_obj_file(
        content, tolerance=args.tolerance, min_valence=args.min_valence, plot_results=plot_enabled
    )
    
    # Write the result
    write_obj_file(vertices, split_polylines, output_file)
    print(f"\nSplit curves written to {output_file}")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Found {len(high_valence_vertices)} intersection points")
    print(f"- Split {len(split_polylines)} total polyline segments")
    
    if plot_enabled:
        print("- Plots displayed (close windows to continue)")
    else:
        print("- Use --no-plot flag removed to see visualizations")
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

def find_coincident_vertices(vertices, tolerance=1e-10):
    """Find groups of vertices that are coincident within tolerance."""
    n = len(vertices)
    coincident_groups = []
    processed = set()
    
    for i in range(n):
        if i in processed:
            continue
            
        group = [i]
        processed.add(i)
        
        for j in range(i + 1, n):
            if j in processed:
                continue
                
            # Calculate distance between vertices
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < tolerance:
                group.append(j)
                processed.add(j)
        
        if len(group) > 1:
            coincident_groups.append(group)
    
    return coincident_groups

def split_polylines_at_intersections(vertices, polylines, tolerance=1e-10):
    """Split polylines where vertices are coincident and more than 2 polylines meet."""
    # Find coincident vertex groups
    coincident_groups = find_coincident_vertices(vertices, tolerance)
    
    # Create mapping from vertex index to group representative
    vertex_to_group = {}
    for group in coincident_groups:
        representative = min(group)  # Use lowest index as representative
        for vertex_idx in group:
            vertex_to_group[vertex_idx] = representative
    
    # Count how many polylines pass through each intersection point
    intersection_usage = defaultdict(set)  # representative -> set of polyline indices
    vertex_to_polylines = defaultdict(list)  # vertex -> list of (polyline_idx, position_in_polyline)
    
    for polyline_idx, polyline in enumerate(polylines):
        for pos, vertex_idx in enumerate(polyline):
            if vertex_idx in vertex_to_group:
                representative = vertex_to_group[vertex_idx]
                intersection_usage[representative].add(polyline_idx)
                vertex_to_polylines[representative].append((polyline_idx, pos))
            else:
                # Handle vertices that are not part of coincident groups
                intersection_usage[vertex_idx].add(polyline_idx)
                vertex_to_polylines[vertex_idx].append((polyline_idx, pos))
    
    # Find intersections where more than 2 polylines meet (changed from >3 to >2)
    split_intersections = set()
    for vertex_key, polyline_set in intersection_usage.items():
        if len(polyline_set) > 2:
            split_intersections.add(vertex_key)
            print(f"Intersection at vertex {vertex_key + 1} (1-indexed) has {len(polyline_set)} polylines meeting - will split")
    
    split_polylines = []
    
    for polyline_idx, polyline in enumerate(polylines):
        # Find positions in this polyline where we need to split
        split_positions = []
        
        for pos, vertex_idx in enumerate(polyline):
            # Check if this vertex is a split intersection
            vertex_key = vertex_to_group.get(vertex_idx, vertex_idx)
            
            if vertex_key in split_intersections:
                # Only split at interior vertices or endpoints that connect to other polylines
                if pos == 0 or pos == len(polyline) - 1:
                    # Check if this endpoint connects to other polylines
                    other_polylines_at_vertex = [
                        p_idx for p_idx, p_pos in vertex_to_polylines[vertex_key] 
                        if p_idx != polyline_idx
                    ]
                    if other_polylines_at_vertex:
                        split_positions.append(pos)
                else:
                    # Interior vertex - always split
                    split_positions.append(pos)
        
        # If no split positions, keep the original polyline
        if not split_positions:
            split_polylines.append(polyline)
            continue
        
        # Split the polyline at the identified positions
        current_segment = []
        
        for pos, vertex_idx in enumerate(polyline):
            current_segment.append(vertex_idx)
            
            # If this position is a split point and we have a valid segment
            if pos in split_positions and len(current_segment) >= 2:
                split_polylines.append(current_segment[:])  # Add copy of segment
                current_segment = [vertex_idx]  # Start new segment with this vertex
        
        # Add the final segment if it's valid
        if len(current_segment) >= 2:
            split_polylines.append(current_segment)
    
    return split_polylines, coincident_groups, split_intersections

def remove_duplicate_vertices(vertices, polylines, tolerance=1e-10):
    """Remove duplicate vertices and update polyline indices accordingly."""
    n = len(vertices)
    vertex_map = {}  # Maps old index to new index
    unique_vertices = []
    
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
    
    return np.array(unique_vertices), updated_polylines, vertex_map

def plot_polylines(vertices, polylines, split_intersections=None, title="Polylines Visualization", 
                   vertex_to_group=None, show_vertex_labels=False):
    """
    Plot polylines with highlighted intersection vertices.
    
    Parameters:
    - vertices: numpy array of vertex coordinates
    - polylines: list of polylines (lists of vertex indices)
    - split_intersections: set of vertex indices that are intersection points
    - title: plot title
    - vertex_to_group: mapping from vertex index to group representative (for coincident vertices)
    - show_vertex_labels: whether to show vertex index labels
    """
    # Determine if we need 2D or 3D plot
    if vertices.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        is_3d = False
    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        is_3d = True
    
    # Plot polylines
    colors = plt.cm.tab10(np.linspace(0, 1, len(polylines)))
    
    for i, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        
        # Get coordinates for this polyline
        polyline_coords = vertices[polyline]
        
        if is_3d:
            ax.plot(polyline_coords[:, 0], polyline_coords[:, 1], polyline_coords[:, 2], 
                   color=colors[i], linewidth=2, alpha=0.7, label=f'Polyline {i+1}')
        else:
            ax.plot(polyline_coords[:, 0], polyline_coords[:, 1], 
                   color=colors[i], linewidth=2, alpha=0.7, label=f'Polyline {i+1}')
    
    # Highlight intersection vertices
    if split_intersections:
        intersection_coords = []
        for vertex_idx in split_intersections:
            # Handle case where vertex_idx might be a group representative
            if vertex_to_group:
                # Find all original vertices that map to this representative
                original_vertices = [v for v, rep in vertex_to_group.items() if rep == vertex_idx]
                if original_vertices:
                    intersection_coords.extend([vertices[v] for v in original_vertices])
                else:
                    intersection_coords.append(vertices[vertex_idx])
            else:
                intersection_coords.append(vertices[vertex_idx])
        
        if intersection_coords:
            intersection_coords = np.array(intersection_coords)
            if is_3d:
                ax.scatter(intersection_coords[:, 0], intersection_coords[:, 1], intersection_coords[:, 2], 
                          color='red', s=100, zorder=5, label='Intersection Points', marker='o')
            else:
                ax.scatter(intersection_coords[:, 0], intersection_coords[:, 1], 
                          color='red', s=100, zorder=5, label='Intersection Points', marker='o')
    
    # Plot all vertices as small points
    if is_3d:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  color='black', s=20, alpha=0.5, zorder=3)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], 
                  color='black', s=20, alpha=0.5, zorder=3)
    
    # Add vertex labels if requested
    if show_vertex_labels:
        for i, vertex in enumerate(vertices):
            if is_3d:
                ax.text(vertex[0], vertex[1], vertex[2], f'{i+1}', fontsize=8, alpha=0.7)
            else:
                ax.text(vertex[0], vertex[1], f'{i+1}', fontsize=8, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if is_3d:
        ax.set_zlabel('Z')
    
    # Add legend (but limit to reasonable number of entries)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 20:  # Only show legend if not too many polylines
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

def process_obj_file(input_content, tolerance=1e-10, plot_results=True):
    """Main function to process OBJ file and split curves at intersections."""
    # Parse the input
    vertices, polylines = parse_obj_file(input_content)
    
    print(f"Original: {len(vertices)} vertices, {len(polylines)} polylines")
    
    # Plot original polylines
    if plot_results:
        fig1, ax1 = plot_polylines(vertices, polylines, title="Original Polylines")
        plt.show()
    
    # Split polylines at intersections
    split_polylines, coincident_groups, split_intersections = split_polylines_at_intersections(
        vertices, polylines, tolerance
    )
    
    print(f"After splitting: {len(vertices)} vertices, {len(split_polylines)} polylines")
    print(f"Found {len(coincident_groups)} groups of coincident vertices")
    
    # Create vertex mapping for plotting
    vertex_to_group = {}
    for group in coincident_groups:
        representative = min(group)
        for vertex_idx in group:
            vertex_to_group[vertex_idx] = representative
    
    # Plot split polylines with highlighted intersections
    if plot_results:
        fig2, ax2 = plot_polylines(
            vertices, split_polylines, split_intersections, 
            title="Split Polylines with Intersection Points",
            vertex_to_group=vertex_to_group
        )
        plt.show()
    
    # Remove duplicate vertices and update indices
    unique_vertices, final_polylines, vertex_map = remove_duplicate_vertices(
        vertices, split_polylines, tolerance
    )
    
    print(f"After removing duplicates: {len(unique_vertices)} vertices, {len(final_polylines)} polylines")
    print(f"Removed {len(vertices) - len(unique_vertices)} duplicate vertices")
    
    # Plot final result
    if plot_results:
        # Map split intersections to new vertex indices
        final_intersections = set()
        for old_intersection in split_intersections:
            if old_intersection in vertex_map:
                final_intersections.add(vertex_map[old_intersection])
        
        fig3, ax3 = plot_polylines(
            unique_vertices, final_polylines, final_intersections,
            title="Final Result: Deduplicated Vertices and Split Polylines"
        )
        plt.show()
    
    # Print coincident groups for debugging
    for i, group in enumerate(coincident_groups):
        print(f"Coincident group {i + 1}: vertices {[v + 1 for v in group]} (1-indexed)")
        # Show the actual coordinates
        group_coords = vertices[group]
        print(f"  Coordinates: {group_coords}")
    
    return unique_vertices, final_polylines, coincident_groups, vertex_map

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('output_file', nargs='?', help='The curve sketch to write.')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--tolerance', type=float, default=1e-8, 
                       help='Tolerance for vertex coincidence detection')
    args = parser.parse_args()

    input_file = args.curve_file
    output_file = args.output_file
    plot_enabled = not args.no_plot

    if not input_file:
        print("Please provide an input OBJ file")
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
    vertices, split_polylines, coincident_groups, vertex_map = process_obj_file(
        content, tolerance=args.tolerance, plot_results=plot_enabled
    )
    
    # Write the result
    write_obj_file(vertices, split_polylines, output_file)
    print(f"Split curves with deduplicated vertices written to {output_file}")
    
    # Optionally, print the vertex mapping for debugging
    print("\nVertex mapping (old_index -> new_index):")
    for old_idx, new_idx in sorted(vertex_map.items()):
        if old_idx != new_idx:  # Only show remapped vertices
            print(f"  {old_idx + 1} -> {new_idx + 1} (1-indexed)")
    
    if plot_enabled:
        print("\nPlots have been displayed. Close the plot windows to continue.")
    else:
        print("Plotting was disabled with --no-plot flag.")
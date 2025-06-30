import numpy as np
from collections import defaultdict
import argparse

def remove_duplicates_complete(obj_content, tolerance=1e-8):
    """
    Remove duplicate vertices and duplicate polylines from OBJ file content.
    Vertices closer than tolerance are considered duplicates.
    Polylines with identical vertex sequences are considered duplicates.
    """
    lines = obj_content.strip().split('\n')
    vertices = []
    line_definitions = []
    other_lines = []
    
    # Parse the OBJ file
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            # Parse vertex coordinates
            coords = list(map(float, line[2:].split()))
            vertices.append(coords)
        elif line.startswith('l '):
            # Parse line indices
            indices = list(map(int, line[2:].split()))
            line_definitions.append(indices)
        else:
            # Keep other lines (comments, etc.)
            if line:
                other_lines.append(line)
    
    print(f"Original vertices: {len(vertices)}")
    print(f"Original polylines: {len(line_definitions)}")
    
    # Convert to numpy array for easier processing
    vertices_array = np.array(vertices)
    
    # STEP 1: Find unique vertices within tolerance
    unique_vertices = []
    vertex_mapping = {}  # old_index -> new_index
    
    for i, vertex in enumerate(vertices_array):
        found_duplicate = False
        
        # Check against existing unique vertices
        for j, unique_vertex in enumerate(unique_vertices):
            # Calculate distance between vertices
            distance = np.linalg.norm(vertex - unique_vertex)
            
            if distance < tolerance:
                # This vertex is a duplicate
                vertex_mapping[i] = j
                found_duplicate = True
                break
        
        if not found_duplicate:
            # This is a new unique vertex
            vertex_mapping[i] = len(unique_vertices)
            unique_vertices.append(vertex)
    
    print(f"Unique vertices after deduplication: {len(unique_vertices)}")
    print(f"Removed {len(vertices) - len(unique_vertices)} duplicate vertices")
    
    # STEP 2: Update line definitions with new vertex indices
    updated_line_definitions = []
    for line_def in line_definitions:
        updated_indices = []
        for old_index in line_def:
            # Convert from 1-based to 0-based, then map, then back to 1-based
            new_index = vertex_mapping[old_index - 1] + 1
            updated_indices.append(new_index)
        updated_line_definitions.append(updated_indices)
    
    # STEP 3: Remove duplicate polylines
    unique_polylines = []
    seen_polylines = set()
    
    for line_def in updated_line_definitions:
        # Create normalized representation of the polyline
        # Sort the line to handle different orderings of the same line
        normalized_line = tuple(sorted(line_def))
        
        # Also check for reversed polylines (same line, different direction)
        reversed_line = tuple(sorted(line_def, reverse=True))
        
        # Check if we've seen this polyline before
        if normalized_line not in seen_polylines and reversed_line not in seen_polylines:
            unique_polylines.append(line_def)
            seen_polylines.add(normalized_line)
    
    print(f"Unique polylines after deduplication: {len(unique_polylines)}")
    print(f"Removed {len(updated_line_definitions) - len(unique_polylines)} duplicate polylines")
    
    # STEP 4: Generate output
    output_lines = []
    
    # Add other lines first (comments, etc.)
    for line in other_lines:
        output_lines.append(line)
    
    # Add unique vertices
    for vertex in unique_vertices:
        vertex_line = f"v {vertex[0]} {vertex[1]} {vertex[2]}"
        output_lines.append(vertex_line)
    
    # Add unique polylines
    for line_def in unique_polylines:
        line_str = "l " + " ".join(map(str, line_def))
        output_lines.append(line_str)
    
    return "\n".join(output_lines)

def remove_duplicates_advanced(obj_content, tolerance=1e-8, remove_degenerate=True):
    """
    Advanced version that also removes degenerate polylines (lines with duplicate vertices).
    """
    lines = obj_content.strip().split('\n')
    vertices = []
    line_definitions = []
    other_lines = []
    
    # Parse the OBJ file
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            coords = list(map(float, line[2:].split()))
            vertices.append(coords)
        elif line.startswith('l '):
            indices = list(map(int, line[2:].split()))
            line_definitions.append(indices)
        else:
            if line:
                other_lines.append(line)
    
    print(f"Original vertices: {len(vertices)}")
    print(f"Original polylines: {len(line_definitions)}")
    
    # Convert to numpy array
    vertices_array = np.array(vertices)
    
    # Find unique vertices
    unique_vertices = []
    vertex_mapping = {}
    
    for i, vertex in enumerate(vertices_array):
        found_duplicate = False
        
        for j, unique_vertex in enumerate(unique_vertices):
            distance = np.linalg.norm(vertex - unique_vertex)
            
            if distance < tolerance:
                vertex_mapping[i] = j
                found_duplicate = True
                break
        
        if not found_duplicate:
            vertex_mapping[i] = len(unique_vertices)
            unique_vertices.append(vertex)
    
    print(f"Unique vertices: {len(unique_vertices)} (removed {len(vertices) - len(unique_vertices)})")
    
    # Update line definitions with new vertex indices
    updated_line_definitions = []
    for line_def in line_definitions:
        updated_indices = []
        for old_index in line_def:
            new_index = vertex_mapping[old_index - 1] + 1
            updated_indices.append(new_index)
        
        # Remove degenerate polylines (if enabled)
        if remove_degenerate:
            # Remove consecutive duplicate vertices
            cleaned_indices = []
            prev_index = None
            for idx in updated_indices:
                if idx != prev_index:
                    cleaned_indices.append(idx)
                    prev_index = idx
            
            # Only keep polylines with at least 2 different vertices
            if len(cleaned_indices) >= 2:
                updated_line_definitions.append(cleaned_indices)
        else:
            updated_line_definitions.append(updated_indices)
    
    # Remove duplicate polylines with more sophisticated comparison
    unique_polylines = []
    seen_polylines = set()
    
    for line_def in updated_line_definitions:
        # Create multiple representations to catch all duplicates
        representations = []
        
        # Original order
        representations.append(tuple(line_def))
        
        # Reversed order (same polyline, different direction)
        representations.append(tuple(reversed(line_def)))
        
        # Sorted order (for unordered comparison)
        representations.append(tuple(sorted(line_def)))
        
        # Check if any representation has been seen
        is_duplicate = False
        for rep in representations:
            if rep in seen_polylines:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_polylines.append(line_def)
            # Add all representations to seen set
            for rep in representations:
                seen_polylines.add(rep)
    
    print(f"Unique polylines: {len(unique_polylines)} (removed {len(updated_line_definitions) - len(unique_polylines)})")
    
    # Generate output
    output_lines = []
    
    for line in other_lines:
        output_lines.append(line)
    
    for vertex in unique_vertices:
        vertex_line = f"v {vertex[0]} {vertex[1]} {vertex[2]}"
        output_lines.append(vertex_line)
    
    for line_def in unique_polylines:
        line_str = "l " + " ".join(map(str, line_def))
        output_lines.append(line_str)
    
    return "\n".join(output_lines)

def process_obj_file(input_filename, output_filename=None, tolerance=1e-8, advanced=True):
    """
    Process an OBJ file to remove duplicate vertices and polylines.
    """
    try:
        with open(input_filename, 'r') as f:
            content = f.read()
        
        if advanced:
            processed_content = remove_duplicates_advanced(content, tolerance)
        else:
            processed_content = remove_duplicates_complete(content, tolerance)
        
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(processed_content)
            print(f"Processed file saved as: {output_filename}")
        else:
            print("\nProcessed content:")
            print(processed_content)
        
        return processed_content
        
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Example for your specific case:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('output_file',nargs='?', help='The curve sketch to write.')

    args = parser.parse_args()

    curve_file = args.curve_file
    output_file = args.output_file

    process_obj_file(curve_file, output_file)

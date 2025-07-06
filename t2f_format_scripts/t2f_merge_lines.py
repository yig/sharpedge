"""
OBJ Polyline Converter
Converts OBJ format from individual line segments to connected polylines.
Input: v x y z, g polylineX, l v1 v2, l v2 v3, ...
Output: v x y z, l 1 2 3 4 5 ... (single line connecting all vertices in polyline)
"""
import argparse

def parse_obj_file(filename):
    """Parse OBJ file and extract vertices and polylines."""
    vertices = []
    polylines = []
    current_polyline = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Parse vertex
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
            
            # Parse line segment
            elif line.startswith('l '):
                parts = line.split()
                v1, v2 = int(parts[1]), int(parts[2])  # Keep 1-based indexing
                current_polyline.append((v1, v2))
            
            # New group indicates start of new polyline
            elif line.startswith('g '):
                if current_polyline:
                    polylines.append(current_polyline)
                    current_polyline = []
    
    # Add the last polyline if it exists
    if current_polyline:
        polylines.append(current_polyline)
    
    return vertices, polylines

def extract_ordered_vertices(polyline_segments):
    """Extract ordered vertex sequence from line segments."""
    if not polyline_segments:
        return []
    
    # Build adjacency list
    adjacency = {}
    all_vertices = set()
    
    for v1, v2 in polyline_segments:
        all_vertices.add(v1)
        all_vertices.add(v2)
        
        if v1 not in adjacency:
            adjacency[v1] = []
        if v2 not in adjacency:
            adjacency[v2] = []
        
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)
    
    # Find start vertex (vertex with only one connection)
    start_vertex = None
    for vertex in all_vertices:
        if len(adjacency[vertex]) == 1:
            start_vertex = vertex
            break
    
    if start_vertex is None:
        # If no endpoint found, it might be a closed loop, start with any vertex
        start_vertex = list(all_vertices)[0]
    
    # Traverse the polyline to get ordered vertices
    ordered_vertices = [start_vertex]
    current = start_vertex
    previous = None
    
    while True:
        # Get next vertices (excluding the one we came from)
        next_vertices = [v for v in adjacency[current] if v != previous]
        
        if not next_vertices:
            break
        
        next_vertex = next_vertices[0]
        
        # Avoid adding the same vertex twice (for closed loops)
        if next_vertex in ordered_vertices:
            break
            
        ordered_vertices.append(next_vertex)
        previous = current
        current = next_vertex
    
    return ordered_vertices

def convert_to_connected_polylines(input_filename, output_filename):
    """Convert OBJ file to connected polyline format."""
    print(f"Reading {input_filename}...")
    vertices, polylines = parse_obj_file(input_filename)
    print(f"Found {len(vertices)} vertices and {len(polylines)} polylines")
    
    with open(output_filename, 'w') as file:
        # Write all vertices first
        for x, y, z in vertices:
            file.write(f"v {x} {y} {z}\n")
        
        # Process each polyline
        for i, polyline_segments in enumerate(polylines):
            print(f"Processing polyline {i}...")
            ordered_vertices = extract_ordered_vertices(polyline_segments)
            
            if ordered_vertices:
                # Write single line command connecting all vertices in this polyline
                vertex_sequence = ' '.join(map(str, ordered_vertices))
                file.write(f"l {vertex_sequence}\n")
    
    print(f"Conversion complete! Output written to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert True2Form OBJ file to connected polylines.')
    parser.add_argument('input_file', help='Input OBJ file')
    parser.add_argument('output_file', nargs='?', help='Optional output file (if omitted, print to stdout)')
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    if output_file:
        convert_to_connected_polylines(input_file, output_file)
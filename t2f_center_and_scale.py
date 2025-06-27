import numpy as np
import sys

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
            polylines.append(indices)
    
    return np.array(vertices), polylines

def normalize_vertices(vertices):
    """
    Normalize vertices to:
    1. Center at origin (0,0,0)
    2. Scale so bounding box diagonal = 1
    """
    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    print(f"Original bounding box:")
    print(f"  Min: [{min_coords[0]:.6f}, {min_coords[1]:.6f}, {min_coords[2]:.6f}]")
    print(f"  Max: [{max_coords[0]:.6f}, {max_coords[1]:.6f}, {max_coords[2]:.6f}]")
    
    # Calculate center and size
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    diagonal = np.linalg.norm(size)
    
    print(f"  Center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"  Size: [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
    print(f"  Diagonal: {diagonal:.6f}")
    
    # Step 1: Translate to center at origin
    centered_vertices = vertices - center
    
    # Step 2: Scale so diagonal = 1
    if diagonal > 0:
        scale_factor = 1.0 / diagonal
        normalized_vertices = centered_vertices * scale_factor
    else:
        normalized_vertices = centered_vertices
        scale_factor = 1.0
    
    # Verify normalization
    new_min = np.min(normalized_vertices, axis=0)
    new_max = np.max(normalized_vertices, axis=0)
    new_center = (new_min + new_max) / 2
    new_size = new_max - new_min
    new_diagonal = np.linalg.norm(new_size)
    
    print(f"\nNormalized bounding box:")
    print(f"  Min: [{new_min[0]:.6f}, {new_min[1]:.6f}, {new_min[2]:.6f}]")
    print(f"  Max: [{new_max[0]:.6f}, {new_max[1]:.6f}, {new_max[2]:.6f}]")
    print(f"  Center: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
    print(f"  Size: [{new_size[0]:.6f}, {new_size[1]:.6f}, {new_size[2]:.6f}]")
    print(f"  Diagonal: {new_diagonal:.6f}")
    print(f"  Scale factor applied: {scale_factor:.6f}")
    
    return normalized_vertices, center, scale_factor

def write_obj_file(vertices, polylines, output_filename):
    """Write vertices and polylines back to OBJ format."""
    with open(output_filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        # Write polylines
        for polyline in polylines:
            indices_str = ' '.join(map(str, polyline))
            f.write(f"l {indices_str}\n")

def process_obj_file(input_content):
    """Main function to normalize OBJ file."""
    # Parse the input
    vertices, polylines = parse_obj_file(input_content)
    
    print(f"Loaded: {len(vertices)} vertices, {len(polylines)} polylines")
    
    # Normalize vertices
    normalized_vertices, original_center, scale_factor = normalize_vertices(vertices)
    
    return normalized_vertices, polylines, original_center, scale_factor

def main():
    if len(sys.argv) != 3:
        print("Usage: python normalize_obj.py input.obj output.obj")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read input file
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Process the file
    normalized_vertices, polylines, original_center, scale_factor = process_obj_file(content)
    
    
    # Write the result
    write_obj_file(normalized_vertices, polylines, output_file)
    
    print(f"\nNormalized model written to: {output_file}")
    print(f"Transformation applied:")
    print(f"  1. Translated by: [{-original_center[0]:.6f}, {-original_center[1]:.6f}, {-original_center[2]:.6f}]")
    print(f"  2. Scaled by: {scale_factor:.6f}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Curve to OBJ Converter with Duplicate Vertex Elimination

Converts .curve format to OBJ format.
Removes duplicate vertices to create cleaner, more efficient output.

Input format (.curve):
- First line: number of curves
- For each curve:
  - Line 1: num_points open_closed_flag capacity
  - Next num_points lines: x y z coordinates

Output format (.obj):
- v x y z (for each unique vertex)
- l v1 v2 v3 ... (for each curve, using 1-based indices)
"""

def convert_curve_to_obj_no_duplicates(input_file, output_file, tolerance=1e-6):
    """
    Convert .curve format to OBJ format with duplicate vertex removal
    
    Args:
        input_file: path to input .curve file
        output_file: path to output .obj file
        tolerance: tolerance for considering vertices as duplicates (default: 1e-6)
    """
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the input
    line_idx = 0
    num_curves = int(lines[line_idx].strip())
    line_idx += 1
    
    # Dictionary to store unique vertices with their indices
    # Key: (x, y, z) rounded to tolerance, Value: 1-based index
    vertex_map = {}
    unique_vertices = []  # List of unique vertices in order
    curves = []  # Store curve definitions (list of vertex indices)
    
    vertex_count = 0  # Keep track of unique vertex count
    total_vertices_processed = 0
    
    print(f"Processing {num_curves} curves...")
    
    for curve_idx in range(num_curves):
        # Parse curve info: num_points, open/closed flag, capacity
        curve_info = list(map(int, lines[line_idx].strip().split()))
        num_points = curve_info[0]
        line_idx += 1
        
        print(f"Curve {curve_idx + 1}: {num_points} points")
        
        # Store vertex indices for this curve
        curve_vertices = []
        
        # Read all points for this curve
        for point_idx in range(num_points):
            coords = list(map(float, lines[line_idx].strip().split()))
            total_vertices_processed += 1
            
            # Round coordinates to handle floating point precision issues
            rounded_coords = tuple(round(coord / tolerance) * tolerance for coord in coords)
            
            # Check if this vertex already exists
            if rounded_coords in vertex_map:
                # Use existing vertex index
                vertex_index = vertex_map[rounded_coords]
            else:
                # Add new unique vertex
                vertex_count += 1
                vertex_index = vertex_count
                vertex_map[rounded_coords] = vertex_index
                unique_vertices.append(coords)  # Store original coordinates
            
            curve_vertices.append(vertex_index)
            line_idx += 1
        
        curves.append(curve_vertices)
    
    # Write OBJ file
    with open(output_file, 'w') as f:
        # Write header comment
        f.write("# OBJ file converted from .curve format\n")
        f.write(f"# Original vertices: {total_vertices_processed}\n")
        f.write(f"# Unique vertices: {len(unique_vertices)}\n")
        f.write(f"# Duplicates removed: {total_vertices_processed - len(unique_vertices)}\n")
        f.write(f"# Total curves: {len(curves)}\n")
        f.write(f"# Tolerance used: {tolerance}\n")
        f.write("\n")
        
        # Write all unique vertices
        f.write("# Vertices\n")
        for vertex in unique_vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        f.write("\n# Lines (curves)\n")
        # Write all curves as lines
        for i, curve in enumerate(curves):
            f.write(f"# Curve {i + 1}\n")
            line_str = "l " + " ".join(map(str, curve))
            f.write(line_str + "\n")
    
    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Original vertices: {total_vertices_processed}")
    print(f"Unique vertices: {len(unique_vertices)}")
    print(f"Duplicates removed: {total_vertices_processed - len(unique_vertices)}")
    print(f"Compression ratio: {len(unique_vertices)/total_vertices_processed:.2%}")
    print(f"Total curves: {len(curves)}")

def convert_from_text_no_duplicates(curve_data_text, output_file, tolerance=1e-6):
    """
    Convert curve data directly from text content to OBJ format with duplicate removal
    
    Args:
        curve_data_text: string containing the curve data
        output_file: path to output .obj file
        tolerance: tolerance for considering vertices as duplicates
    """
    
    lines = curve_data_text.strip().split('\n')
    
    # Parse the input
    line_idx = 0
    num_curves = int(lines[line_idx].strip())
    line_idx += 1
    
    # Dictionary to store unique vertices with their indices
    vertex_map = {}
    unique_vertices = []
    curves = []
    
    vertex_count = 0
    total_vertices_processed = 0
    
    print(f"Processing {num_curves} curves...")
    
    for curve_idx in range(num_curves):
        if line_idx >= len(lines):
            break
            
        # Parse curve info: num_points, open/closed flag, capacity
        curve_info = list(map(int, lines[line_idx].strip().split()))
        num_points = curve_info[0]
        line_idx += 1
        
        print(f"Curve {curve_idx + 1}: {num_points} points")
        
        curve_vertices = []
        
        # Read all points for this curve
        for point_idx in range(num_points):
            if line_idx >= len(lines):
                break
            coords = list(map(float, lines[line_idx].strip().split()))
            total_vertices_processed += 1
            
            # Round coordinates to handle floating point precision
            rounded_coords = tuple(round(coord / tolerance) * tolerance for coord in coords)
            
            # Check if this vertex already exists
            if rounded_coords in vertex_map:
                vertex_index = vertex_map[rounded_coords]
            else:
                vertex_count += 1
                vertex_index = vertex_count
                vertex_map[rounded_coords] = vertex_index
                unique_vertices.append(coords)
            
            curve_vertices.append(vertex_index)
            line_idx += 1
        
        curves.append(curve_vertices)
    
    # Write OBJ file
    with open(output_file, 'w') as f:
        # Write header comment
        f.write("# OBJ file converted from Speaker.curve format\n")
        f.write(f"# Original vertices: {total_vertices_processed}\n")
        f.write(f"# Unique vertices: {len(unique_vertices)}\n")
        f.write(f"# Duplicates removed: {total_vertices_processed - len(unique_vertices)}\n")
        f.write(f"# Total curves: {len(curves)}\n")
        f.write(f"# Tolerance used: {tolerance}\n")
        f.write("\n")
        
        # Write all unique vertices
        f.write("# Vertices\n")
        for vertex in unique_vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        f.write("\n# Lines (curves)\n")
        # Write all curves as lines
        for i, curve in enumerate(curves):
            f.write(f"# Curve {i + 1}\n")
            line_str = "l " + " ".join(map(str, curve))
            f.write(line_str + "\n")
    
    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Original vertices: {total_vertices_processed}")
    print(f"Unique vertices: {len(unique_vertices)}")
    print(f"Duplicates removed: {total_vertices_processed - len(unique_vertices)}")
    print(f"Compression ratio: {len(unique_vertices)/total_vertices_processed:.2%}")
    print(f"Total curves: {len(curves)}")

def analyze_duplicates(input_file, tolerance=1e-6):
    """
    Analyze the input file to report duplicate statistics without converting
    
    Args:
        input_file: path to input .curve file
        tolerance: tolerance for considering vertices as duplicates
    """
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    line_idx = 0
    num_curves = int(lines[line_idx].strip())
    line_idx += 1
    
    vertex_map = {}
    total_vertices = 0
    
    for curve_idx in range(num_curves):
        curve_info = list(map(int, lines[line_idx].strip().split()))
        num_points = curve_info[0]
        line_idx += 1
        
        for point_idx in range(num_points):
            coords = list(map(float, lines[line_idx].strip().split()))
            total_vertices += 1
            
            rounded_coords = tuple(round(coord / tolerance) * tolerance for coord in coords)
            vertex_map[rounded_coords] = vertex_map.get(rounded_coords, 0) + 1
            line_idx += 1
    
    unique_vertices = len(vertex_map)
    duplicates = total_vertices - unique_vertices
    
    print(f"Duplicate Analysis:")
    print(f"Total vertices: {total_vertices}")
    print(f"Unique vertices: {unique_vertices}")
    print(f"Duplicate vertices: {duplicates}")
    print(f"Compression ratio: {unique_vertices/total_vertices:.2%}")
    
    # Show most frequently duplicated vertices
    if duplicates > 0:
        freq_sorted = sorted(vertex_map.items(), key=lambda x: x[1], reverse=True)
        print(f"\nMost frequently duplicated vertices:")
        for i, (coords, count) in enumerate(freq_sorted[:5]):
            if count > 1:
                print(f"  {coords}: appears {count} times")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert .curve file to .obj with duplicate vertex removal")
    parser.add_argument("input_file", help="Path to the input .curve file")
    parser.add_argument("output_file", help="Path to the output .obj file")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Tolerance for considering vertices as duplicates (default: 1e-6)")

    args = parser.parse_args()

    print("Analyzing input file for duplicates...")
    analyze_duplicates(args.input_file, args.tolerance)
    print("\n" + "=" * 50 + "\n")

    convert_curve_to_obj_no_duplicates(args.input_file, args.output_file, args.tolerance)


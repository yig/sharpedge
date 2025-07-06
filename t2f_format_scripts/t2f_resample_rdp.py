"""
t2f_resample_rdp.py

This script resamples 3D sketch polylines using the Ramer–Douglas–Peucker (RDP) algorithm.
It reduces the number of vertices to simplify the curves while preserving the shape.
This helps reduce computational load in later processing stages.

Input: An OBJ file with 'v' and 'l' elements.
Output: A simplified OBJ file with merged and resampled polylines.

Usage:
    python t2f_resample_rdp.py input.obj output.obj
"""


from rdp import rdp
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data, plot_polylines
import argparse


def rdp_closed(polyline, epsilon=2e-3):
    """Apply RDP to a closed curve properly"""
    if len(polyline) < 4:
        return polyline
    
    # Check if curve is actually closed (first and last points are same/close)
    first_point = np.array(polyline[0])
    last_point = np.array(polyline[-1])
    is_closed = np.linalg.norm(first_point - last_point) < 1e-6
    
    if not is_closed:
        # Open curve, use regular RDP
        result = rdp(polyline, epsilon=epsilon)
        # Ensure it's a list of lists
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result
    
    # For closed curves, we need to be more careful
    # Remove the duplicate last point temporarily
    open_polyline = polyline[:-1] if is_closed else polyline
    
    # Apply RDP to the open version
    simplified_open = rdp(open_polyline, epsilon=epsilon)
    
    # Convert to list if it's a numpy array
    if isinstance(simplified_open, np.ndarray):
        simplified_open = simplified_open.tolist()
    
    # Add back the closing point (ensure it's also a list)
    if len(simplified_open) > 0:
        closing_point = simplified_open[0] if isinstance(simplified_open[0], list) else simplified_open[0].tolist()
        simplified_closed = simplified_open + [closing_point]
    else:
        simplified_closed = simplified_open
    
    return simplified_closed

def write_polylines_to_obj(polylines, out_file_path):
    # First collect all vertices and build mapping
    all_vertices = []
    vert_to_idx = {}
    new_vidx = []  # Store new vertex indices for each original vertex
    
    # Process all vertices with rounding
    for polyline in polylines:
        for v in polyline:
            # Round to 6 decimal places
            v_rounded = tuple(round(x, 6) for x in v)
            if v_rounded not in vert_to_idx:
                vert_to_idx[v_rounded] = len(all_vertices)
                all_vertices.append(v_rounded)
            new_vidx.append(vert_to_idx[v_rounded] + 1)  # +1 for OBJ indexing
    
    # Write the OBJ file
    with open(out_file_path, 'w') as writer:
        # Write unique vertices rounded to 6 decimal places
        for v in all_vertices:
            writer.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
        
        # Write polylines using the new indices
        curr_idx = 0
        for polyline in polylines:
            txt = "l"
            for _ in range(len(polyline)):
                txt += " {}".format(new_vidx[curr_idx])
                curr_idx += 1
            writer.write(txt + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')    
    parser.add_argument('output_file',nargs='?', help='The curve sketch to write.')
    args = parser.parse_args()


    curve_file = args.curve_file
    output_file = args.output_file

    V, E, P = load_sketch_polyline_data(curve_file)

    # plot_sketch_data(V, P)


    polylines = [[V[index] for index in p] for p in P]

    resampled_polylines = []

    for polyline in polylines:
        points = rdp_closed(polyline, epsilon=2e-3) 
        resampled_polylines.append( points )

    # plot_polylines(resampled_polylines)

    write_polylines_to_obj(resampled_polylines, output_file)



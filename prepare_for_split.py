import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from utility_segment_distance import segment_to_segment_distance
from utility_io import load_sketch_polyline_data



def check_duplicates_np(vertices, lines):
   # For both vertices and lines - compare unique count to total count
   v_unique = len(np.unique(vertices, axis=0)) == len(vertices)
   l_unique = len(np.unique(np.array(lines), axis=0)) == len(lines)
   return v_unique, l_unique

def point_on_segment(point, segment_start, segment_end, h = 1e-6):
   """
   Check if point lies on line segment, excluding endpoints.
   Uses normalized vectors and projections for numerical stability.
   
   Args:
       point: (x,y) coordinates of test point
       segment_start: (x,y) coordinates of segment start 
       segment_end: (x,y) coordinates of segment end
       
   Returns:
       bool: True if point lies on segment (excluding endpoints)
   """
   # First check if point is an endpoint
   if np.linalg.norm(point - segment_start) < h or np.linalg.norm(point - segment_end) < h:
       return False

   # Get normalized direction vectors
   segment_vec = segment_end - segment_start  
   segment_length = np.linalg.norm(segment_vec)
   segment_dir = segment_vec / segment_length
   point_vec = point - segment_start
   point_dir = point_vec / np.linalg.norm(point_vec)

   # Check if vectors are parallel (cross product near 0)
   if not np.isclose(np.linalg.norm(np.cross(segment_dir, point_dir)), 0):
       return False

   # Check if projection lies between endpoints
   projection = np.dot(point - segment_start, segment_dir)
   return 0 < projection < segment_length

def find_vertex_line_overlaps(vertices, line_indices):
   """
   Find vertices that lie on line segments excluding endpoints.
   Returns list of (vertex_index, line_index) tuples.
   
   Args:
       vertices: List of (x,y,z) coordinate pairs
       lines: List of (start_idx, end_idx) pairs indexing into vertices
   """
   overlaps = []
   for i, (start_idx, end_idx) in enumerate(line_indices):
       start, end = vertices[start_idx], vertices[end_idx]
       # Check each vertex against current line segment
       for j, point in enumerate(vertices):
           if j not in (start_idx, end_idx) and point_on_segment(point, start, end):
               overlaps.append((j, i))
   return overlaps

def find_line_line_overlaps(vertices, line_indices):
    """
    Find overlaps between non-connected line segments
    Args:
        vertices: List of (x,y) coordinates
        line_indices: List of (start_idx, end_idx) pairs indexing vertices
    Returns:
        List of (line1_idx, line2_idx) tuples for overlapping segments
    """
    overlaps = []
    for i, (start0, end0) in enumerate(line_indices):
        for j, (start1, end1) in enumerate(line_indices[i+1:], i+1):
            # Skip if lines share endpoint
            if start0 in (start1, end1) or end0 in (start1, end1):
                continue

            # Get actual vertex coordinates
            line0_start = vertices[start0] 
            line0_end = vertices[end0]
            line1_start = vertices[start1]
            line1_end = vertices[end1]

            # Check for overlap
            dist, closest_point_on_a, closest_point_on_b = segment_to_segment_distance(line0_start, line0_end, line1_start, line1_end)
            if dist < 1e-2:
                overlaps.append((i, j, closest_point_on_a, closest_point_on_b))

    return overlaps


def split_point_line_overlaps(line_indices, overlaps):
    '''
    overlaps: (vertex_index, line_index)
    '''

    # I'll make those changes
    # if the vertex_index lie on line_index
    # can I just split the line_index to 2 part (start_index, point_index), (point_index, end_index)
    overlap_line_indices  = [pair[1] for pair in overlaps]

    line_indices_splitted = []

    for vertex_index, line_index in overlaps:
        start_index, end_index = line_indices[line_index]
        line_indices_splitted.append((start_index, vertex_index))
        line_indices_splitted.append((vertex_index, end_index))

    for i in range(len(line_indices)):
        if i not in overlap_line_indices:
            line_indices_splitted.append(line_indices[i])
    
    return line_indices_splitted

        

def plot_lines_and_intersections(vertices, line_indices, intersections):
    '''
    Plot polylines and their intersections in 3D.
    
    Args:
        lines: list of lines
        intersections: list of (point, polyline_idx, segment_idx)
    '''
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vertical_axis='y', elev=30, azim=45)
    ax.set_aspect('equal')
    
    # Plot polylines
    for l0, l1 in line_indices:
        p0 = vertices[l0]
        p1 = vertices[l1]

        pts = np.asarray( [p0, p1])
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        ax.plot(xs, ys, zs)
        # ax.scatter(xs, ys, zs, color = 'black', s = 60, alpha = 0.2)
    
    # Plot intersections
    for index,( _, _, point_a, point_b) in enumerate(intersections):
        if point_a is not None:
            print(index)
            ax.scatter(*point_a, color='r', s=50, alpha=0.4)
            ax.scatter(*point_b, color='b', s=50, alpha=0.4)

    plt.axis('off')
    plt.axis('equal')
    plt.show()





def split_line_line_overlaps(vertices, line_indices, overlaps):
    '''
    '''
    # overlap_line_indices = [idx for pair in overlaps for idx in pair]
    overlap_line_indices = [idx for pair in overlaps for idx in (pair[0], pair[1])]

    print('overlap_line_indices', overlap_line_indices)

    splitted_lines = []

    for i, j, _ in overlaps:
        start0, end0 = line_indices[i]
        start1, end1 = line_indices[j]

        line0_start = vertices[start0] 
        line0_end = vertices[end0]
        line1_start = vertices[start1]
        line1_end = vertices[end1]

        dist, closest_point_on_a, closest_point_on_b = segment_to_segment_distance(line0_start, line0_end, line1_start, line1_end)
        split_point = (closest_point_on_a + closest_point_on_b) / 2
        splitted_lines.append((line0_start, split_point))
        splitted_lines.append((split_point, line0_end))

        splitted_lines.append((line1_start, split_point))
        splitted_lines.append((split_point, line1_end))

    for i in range(len(line_indices)):
        print('i, overlap_line_indices', i, overlap_line_indices)
        if i not in overlap_line_indices:
            start_i, end_i = line_indices[i]
            start, end = vertices[start_i], vertices[end_i]
            splitted_lines.append((start, end))



    return splitted_lines


def write_lines_to_obj(lines, out_file_path):
    """Write 3D line segments to OBJ file with deduplicated vertices
    
    Args:
        lines: List of line segments, each containing two 3D points
        filename: Output OBJ file path
    """
    vertices = []
    vert_dict = {}
    indices = []

    # Process vertices, deduplicating with fixed precision
    for p0, p1 in lines:
        for p in (p0, p1):
            p_tuple = tuple(round(x, 6) for x in p)
            if p_tuple not in vert_dict:
                vert_dict[p_tuple] = len(vertices)  
                vertices.append(p)
                
        # Store vertex indices for line segment
        idx0 = vert_dict[tuple(round(x, 6) for x in p0)]
        idx1 = vert_dict[tuple(round(x, 6) for x in p1)]
        indices.append((idx0, idx1))

    # Write OBJ file
    with open(out_file_path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for i0, i1 in indices:
            f.write(f'l {i0+1} {i1+1}\n') 

    print(f"\nWrite to {out_file_path}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(lines)} lines")



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='cdt sketch file to view')
    parser.add_argument('sketch_file', nargs='?', help='Sketch file.')
    # parser.add_argument('output_file', nargs='?', help='Sketch file without self-intersection.')

    args = parser.parse_args()

    sketch_file = args.sketch_file
    # output_file = args.output_file

    # sketch_file = 'sketches/flowrep/flowrep_spherecylinder.obj'



    vertices, edges, polylines  = load_sketch_polyline_data( sketch_file )

    print('len(np.unique(vertices, axis=0))', len(np.unique(vertices, axis=0)))
    print('len(vertices)', len(vertices))

    print('len(np.unique(np.array(line_indices), axis=0))', len(np.unique(np.array(edges), axis=0)))
    print('len(lines)',len(edges))

    v_unique, l_unique = check_duplicates_np(vertices, edges)

    lines = [[ vertices[l0], vertices[l1]] for l0, l1 in edges]

    # write_lines_to_obj(lines, output_file)
    # print(lines)
    # print('len(lines)', len(lines))

    print('v_unique', v_unique)
    print('l_unique', l_unique)

    # 1. make sure the vertices and lines are unique
    assert v_unique is True and l_unique is True 

    # 2. there should be no vertice lie on the line segment
    overlaps = find_vertex_line_overlaps(vertices, edges)
    print('vertex line overlaps', overlaps)

    # print('line_indices', line_indices)
    # print('len(line_indices)', len(line_indices))

    if len(overlaps) != 0:
        print('after split len(line_indices)', len(edges))

    # 3. find line line intersection
    # since there are no point lie on line. 
    # if the 2 points share one endpoit. 
    # I can just ignore that.

    overlaps = find_line_line_overlaps(vertices, edges)
    print('find_line_line_overlaps overlaps', overlaps)

    print(len(overlaps))

    if len(overlaps) != 0:
        plot_lines_and_intersections(vertices, edges, overlaps )


    if len(overlaps) == 0:
        print('do not need split')
        exit(1)


    



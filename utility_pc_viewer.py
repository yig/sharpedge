from marching_cube import plot_points_normal

import argparse


import numpy as np 

def load_pc_data(filename):
    """
    Load points and normals from a file in the format:
    v x y z
    vn nx ny nz
    :param filename: Input file name (string)
    :return: points, normals as lists of 3D vectors
    """
    points = []
    normals = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    for i in range(0, len(lines)-1, 2):
        if lines[i].startswith('v ') and lines[i+1].startswith('vn '):
            v_parts = lines[i].split()
            vn_parts = lines[i+1].split()
            
            points.append((float(v_parts[1]), float(v_parts[2]), float(v_parts[3])))
            normals.append((float(vn_parts[1]), float(vn_parts[2]), float(vn_parts[3])))
    
    return np.asarray(points), np.asarray(normals)

parser = argparse.ArgumentParser(description='point cloud version to view.')

# Add arguments
parser.add_argument('point_normal_file', nargs='?',
                    help='Input file containing point and  normal data (.pc)')

args = parser.parse_args()

point_normal_file = args.point_normal_file

points, normals = load_pc_data(point_normal_file)

print('len(points)', len(points))
print('len(normals)', len(normals))


plot_points_normal(points, normals)



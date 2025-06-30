import glob
import subprocess

from pathlib import Path
import pathlib

import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



def optimize_edge_normals( sketch_folder ):
    '''
    '''
    sketches = glob.glob(sketch_folder + '/*.obj')

    print(sketches)

    for sketch_file in sketches:
        curve_name = Path(sketch_file).stem 

        # normal_file = 'normals_gltfs/opt_one_n/' + curve_name + '.normal'
        # gltf_file = 'normals_gltfs/opt_one_n/' + curve_name + '.gltf'
        # subprocess.run(['python', 'opt_edges.py', str(sketch_file), str(normal_file), str(gltf_file), '-p', '1' , '--show_plot', 'false', '--save_debug_gltf', 'false'])
        subprocess.run(['python', 'opt_edges.py', str(sketch_file)])



        # normal_file = 'normals_gltfs/opt_two_n/' + curve_name + '.normal'
        # gltf_file = 'normals_gltfs/opt_two_n/' + curve_name + '.gltf'
        # subprocess.run(['python', 'opt_edges.py', str(sketch_file), str(normal_file), str(gltf_file), '-p', '2' , '--show_plot', 'false', '--save_debug_gltf', 'false'])
    



# sketch_folders = [p for p in pathlib.Path("sketches").iterdir() if p.is_dir()]

# print(sketch_folders)



# sketch_folder = 'sketches/t2f'
# sketch_folder = 'sketches/flowrep'
# sketch_folder = 'sketches/t2f'
# sketch_folder = 'sketches/ils'
# sketch_folder = 'sketches/onshape'
# sketch_folder = 'sketches/author_vr'
# sketch_folder = 'sketches/flowsurf'
# sketch_folder = 'sketches/scaffold_3d'

# sketch_folders = [
#     'sketches/t2f',
#     'sketches/flowrep',
#     'sketches/ils',
#     'sketches/onshape',
#     'sketches/author_vr',
#     'sketches/flowsurf',
#     'sketches/scaffold_3d'
# ]

# for sketch_folder in sketch_folders:
#     optimize_edge_normals(sketch_folder)


optimize_edge_normals(sketch_folder='3d-sketches/flowrep')
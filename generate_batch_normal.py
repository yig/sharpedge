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
        normal_file = 'normal/' + curve_name + '.normal'
        gltf_file = 'normal/' + curve_name + '.gltf'
        subprocess.run(['python', 'opt_edges.py', str(sketch_file), str(normal_file), str(gltf_file), '--show_plot', 'false', '--save_debug_gltf', 'true'])
    



# sketch_folders = [p for p in pathlib.Path("sketches").iterdir() if p.is_dir()]

# print(sketch_folders)



# sketch_folder = 'sketches/t2f'

# sketch_folder = 'sketches/flowrep'
sketch_folder = 'sketches/t2f'
sketch_folder = 'sketches/ils'
sketch_folder = 'sketches/onshape'
# sketch_folder = 'sketches/author_vr'
# sketch_folder = 'sketches/flowsurf'


optimize_edge_normals(sketch_folder)


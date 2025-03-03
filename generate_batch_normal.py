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

    for sketch_file in sketches:
        curve_name = Path(sketch_file).stem 
        normal_file = 'normal/' + curve_name + '.normal'
        subprocess.run(['python', 'opt_edges.py', str(sketch_file), str(normal_file), '--show_plot', 'false'])
    



# sketch_folders = [p for p in pathlib.Path("sketches").iterdir() if p.is_dir()]

# print(sketch_folders)


sketch_folder = 'sketches/flowsurf'
sketch_folder = 'sketches/ils'
optimize_edge_normals(sketch_folder)


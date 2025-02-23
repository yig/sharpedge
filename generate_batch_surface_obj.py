import glob
import subprocess
import shutil

from pathlib import Path
from plot2gltf import GLTFGeometryExporter
from utility_io import load_sketch_polyline_data, load_obj


import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def export_sketch_surface_gltf(polylines, SV, SF, filename):
    '''
    Export the sketch and the surface to file, as gltf.

    polylines: polyline of the sketch 
    SV: surface vertices
    SF: surface faces
    
    '''
    # Initialize exporter
    exporter = GLTFGeometryExporter()

    POLYLINE_RADIUS = 0.002

    for polyline in polylines:
        exporter.add_cylinder_strips(polyline, radius=POLYLINE_RADIUS,add_spheres=False)
    exporter.add_triangles(SV, SF, color=(0.5, 0.5, 0.5)) 

    exporter.save(filename)
    print(f"GLTF file saved as: {filename}")



def generate_all_surface_from_normal_files( normal_files, folder = 'surface' ):
    ## generate all the surface obj 
    # get all the normal files
    for normal_file in normal_files:
        base_name = Path(normal_file).stem 
        surface_file =  folder + '/' + base_name + '.obj'
        subprocess.run(['python', 'marching_cube.py', str(normal_file), str(surface_file)])

def generate_sketch_and_surface_gltf(normal_files, surface_folder = 'surface', gltf_folder = 'gltfs/surfaces/'):        
    ## export all the sketch_
    for normal_file in normal_files:
        base_name = Path(normal_file).stem 
        surface_file = surface_folder + '/' + base_name + '.obj'
        sketch_file_name = base_name + '.obj'
        gltf_surface_file = gltf_folder + '/' + base_name + '.gltf'
        # find sketch_file_path in folder and subfolder of sketches
        sketch_file_path = next(Path('sketches').rglob(sketch_file_name), None)
        print(sketch_file_path)
        if sketch_file_path is None:
            print(f"Could not find {sketch_file_name} in sketches directory")
        else:
            V, E, P  = load_sketch_polyline_data(sketch_file_path)
            polylines = [[V[p] for p in polyline] for polyline in P]
            SV, SF = load_obj(surface_file)
            export_sketch_surface_gltf(polylines, SV, SF, gltf_surface_file)


def copy_all_normal_gltfs(normal_gltf_files):
   copy_folder = Path('gltfs/normals')
   copy_folder.mkdir(parents=True, exist_ok=True)
   
   for normal_gltf in normal_gltf_files:
       shutil.copy2(normal_gltf, copy_folder)



def optimize_edge_normals( sketch_folder ):
    '''
    '''
    sketches = glob.glob(sketch_folder + '/*.obj')

    for sketch_file in sketches:
        curve_name = Path(sketch_file).stem 
        normal_file = 'normal/' + curve_name + '.normal'
        subprocess.run(['python', 'opt_edges.py', str(sketch_file), str(normal_file)])
    



    

# optimize_edge_normals('sketches/flowrep')

normal_files = natural_sort( glob.glob('normal/onshape*.normal')  )
normal_files = natural_sort( glob.glob('normal/t2f*.normal')  )
normal_files = natural_sort( glob.glob('normal/scaffolds3d*.normal')  )
normal_files = natural_sort( glob.glob('normal/ils*.normal')  )
# normal_files = natural_sort( glob.glob('normal/flowrep*.normal')  )

print(normal_files)

generate_all_surface_from_normal_files( normal_files , folder= 'gltfs/convex_hull_area_surface_2nd')
generate_sketch_and_surface_gltf(normal_files, surface_folder='gltfs/convex_hull_area_surface_2nd', gltf_folder = 'gltfs/convex_hull_area_surface_2nd' )

# normal_gltf_files = glob.glob('normal/*.gltf')
# copy_all_normal_gltfs(normal_gltf_files)

import os


# Template for the command
template = "python generate_sketch_and_surface.py /Users/yuxue/Developer/Reverse_Rendering/marching_cube/resampled_sketch_rdp/flowrep/{} marchingcube_surface_rdp/{} rdp_sketch_surface/{}.gltf"

# List of base filenames (without extension)
base_files = [
    "flowrep_trebol",
    "flowrep_bathtub",
    "flowrep_boat",
    "flowrep_bottle",
    "flowrep_ellipsetorus",
    "flowrep_phone",
    "flowrep_spherecylinder"
]

# Generate commands for each file
for base in base_files:
    obj_file = f"{base}.obj"
    # Extract name without 'flowrep_' prefix for the output gltf
    output_name = base.replace('flowrep_', 'flowrep_')
    command = template.format(obj_file, obj_file, output_name)
    print(command)
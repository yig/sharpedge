#!/bin/bash

NORMAL_FILE=$(realpath "$1") 
TARGET_EDGE_LENGTH=${2:-0.04}
TARGET_REMESH_LENGTH=${3:-0.015}  


# given a normal
# need to have ./main ./cut_mesh ./cg_remesh
# 1. generate the mesh using ./main --TARGET_EDGE_LENGTH
# 2. collapse the mesh using collapse_zero_edges.py
# 3. cut the collapsed mesh using ./cut_mesh collapsed_mesh -TARGET_EDGE_LENGTH, 
#    this is because at the first step we use TARGET_EDGE_LENGTH resampled the sketch edge
# 4. remesh the cuted mesh using ./cg_remesh -TARGET_REMESH_LENGTH
#    the boundary will be kept, and the other triangles will be remeshed
# 5. optimize the mesh using Stanko's method.


# A few notice, the ./main, ./cut_mesh ./cg_remesh are in the data folder.
# I want to generate the meshes in the data folder 
# The python file are in current directory 

# met error exit 
set -e  

BASENAME=$(basename "$NORMAL_FILE" .normal)

# All derived filenames use absolute paths
DATA_DIR="$(pwd)/data"

SURFACE_MESH="${DATA_DIR}/${BASENAME}_isosurface.obj"
COLLAPSED_MESH="${DATA_DIR}/${BASENAME}_isosurface_collapsed.obj"
CUTTED_MESH="${DATA_DIR}/${BASENAME}_isosurface_collapsed_cut.obj"
REMESHED_MESH="${DATA_DIR}/${BASENAME}_isosurface_collapsed_cut_remesh.obj"
OPTIMIZED_MESH="${DATA_DIR}/${BASENAME}_isosurface_collapsed_cut_remesh_opt.obj"

# 1. generate the mesh file
pushd data > /dev/null

# the
./main_autosave "$NORMAL_FILE" --t "$TARGET_EDGE_LENGTH" --headless

popd > /dev/null

echo "Generated mesh: $SURFACE_MESH"

# 2. collapse the mesh file
python collapse_zero_edges.py "$SURFACE_MESH"

echo "Generated mesh: $COLLAPSED_MESH"

# 3. cut the mesh 
pushd data > /dev/null

./cut_mesh "$NORMAL_FILE" "$COLLAPSED_MESH" -t "$TARGET_EDGE_LENGTH" 

popd > /dev/null

echo "Generated mesh: $CUTTED_MESH"

#4. remesh the mesh 
pushd data > /dev/null

./cgal_remesh "$CUTTED_MESH" -t "$TARGET_REMESH_LENGTH" 

popd > /dev/null
echo "Generated mesh: $REMESHED_MESH"


# 5. optimize the mesh
python mesh_stanko_normal.py "$REMESHED_MESH" "$NORMAL_FILE" -t "$TARGET_EDGE_LENGTH" 

echo "Generated mesh : $OPTIMIZED_MESH"


SURFACE_ISOSURFACE_DIR="${DATA_DIR}/surface_isosurface"
SURFACE_COLLAPSED_DIR="${DATA_DIR}/surface_collapsed"
SURFACE_CUT_DIR="${DATA_DIR}/surface_cut"
SURFACE_REMESHED_DIR="${DATA_DIR}/surface_remeshed"
SURFACE_OPT_DIR="${DATA_DIR}/surface_opt"


# Copy each mesh to its category folder
cp "$SURFACE_MESH"    "$SURFACE_ISOSURFACE_DIR/"
cp "$COLLAPSED_MESH"  "$SURFACE_COLLAPSED_DIR/"
cp "$CUTTED_MESH"     "$SURFACE_CUT_DIR/"
cp "$REMESHED_MESH"   "$SURFACE_REMESHED_DIR/"
cp "$OPTIMIZED_MESH"  "$SURFACE_OPT_DIR/"

echo "✅ Copied to category folders:"
echo "  → Isosurface:      $SURFACE_ISOSURFACE_DIR/"
echo "  → Collapsed:       $SURFACE_COLLAPSED_DIR/"
echo "  → Cut:             $SURFACE_CUT_DIR/"
echo "  → Remeshed:        $SURFACE_REMESHED_DIR/"
echo "  → Optimized:       $SURFACE_OPT_DIR/"


# Create output folder
NAME_PREFIX=$(echo "$BASENAME" | sed 's/_2n$//')
OUTPUT_FOLDER="${DATA_DIR}/${NAME_PREFIX}"
mkdir -p "$OUTPUT_FOLDER"
OUTPUT_FOLDER=$(realpath "$OUTPUT_FOLDER")


mv "$SURFACE_MESH" "$OUTPUT_FOLDER/"
mv "$COLLAPSED_MESH" "$OUTPUT_FOLDER/"
mv "$CUTTED_MESH" "$OUTPUT_FOLDER/"
mv "$REMESHED_MESH" "$OUTPUT_FOLDER/"
mv "$OPTIMIZED_MESH" "$OUTPUT_FOLDER/"

OPTIMIZED_MESH="$OUTPUT_FOLDER/$(basename "$OPTIMIZED_MESH")"

# 6. Finally viusalize 
python sketch_normal_surface_viewer.py "$NORMAL_FILE" "$OPTIMIZED_MESH" -t "$TARGET_EDGE_LENGTH" 




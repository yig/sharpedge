#!/bin/bash

NORMAL_FILE=$(realpath "$1") 
TARGET_EDGE_LENGTH=${2:-0.04}
TARGET_REMESH_LENGTH=${3:-0.015}  

PYTHON="uv run --with-requirements requirements.freeze.minimal.txt --python 3.12"

SIGNED_HEAT_3D="./main"
CUT_MESH="./cut_mesh"
CGAL_REMESH="./cgal_remesh"

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

"${SIGNED_HEAT_3D}" "$NORMAL_FILE" --t "$TARGET_EDGE_LENGTH" --headless

popd > /dev/null



# 2. collapse the mesh file
${PYTHON} collapse_zero_edges.py "$SURFACE_MESH"

echo "Generated mesh: $COLLAPSED_MESH"

# 3. cut the mesh 
pushd data > /dev/null

"${CUT_MESH}" "$NORMAL_FILE" "$COLLAPSED_MESH" -t "$TARGET_EDGE_LENGTH" 

popd > /dev/null

echo "Generated mesh: $CUTTED_MESH"

#  #4. remesh the mesh 
#  pushd data > /dev/null

#  "${CGAL_REMESH}" "$CUTTED_MESH" -t "$TARGET_REMESH_LENGTH" 

#  popd > /dev/null
#  echo "Generated mesh: $REMESHED_MESH"

# #  5. optimize the mesh
#  python mesh_stanko_normal.py "$REMESHED_MESH" "$NORMAL_FILE" -t "$TARGET_EDGE_LENGTH"


# 4. remesh the mesh 
pushd data > /dev/null
if "${CGAL_REMESH}" "$CUTTED_MESH" -t "$TARGET_REMESH_LENGTH"; then
    echo "Generated mesh: $REMESHED_MESH"
    INPUT_FOR_OPT="$REMESHED_MESH"
else
    echo "⚠️ Remesh failed, falling back to cut mesh."
    INPUT_FOR_OPT="$CUTTED_MESH"
fi
popd > /dev/null

# # 5. optimize the mesh
# python mesh_stanko_normal.py "$INPUT_FOR_OPT" "$NORMAL_FILE" -t "$TARGET_EDGE_LENGTH"
# echo "Generated mesh : $OPTIMIZED_MESH"


# At most 20 seconds
if timeout 20s ${PYTHON} mesh_stanko_normal.py "$INPUT_FOR_OPT" "$NORMAL_FILE" -t "$TARGET_EDGE_LENGTH"; then
    echo "Generated mesh : $OPTIMIZED_MESH"
    echo "✅ run 成功"
else
    echo "First run exceeded 10s or failed, retrying without -t"
    # 第二次：不带 -t，同样限时 10s
    if timeout 20s ${PYTHON} mesh_stanko_normal.py "$INPUT_FOR_OPT" ; then
        echo "Generated mesh (fallback) : $OPTIMIZED_MESH"
        echo "✅ run 成功"
    else
        echo "❌ Fallback run also exceeded 2ç0s or failed"
        # OPTIMIZED_MESH="$INPUT_FOR_OPT"
    fi
fi



# 6. Finally viusalize 
${PYTHON} sketch_normal_surface_viewer.py "$NORMAL_FILE" "$OPTIMIZED_MESH" -t "$TARGET_EDGE_LENGTH" 



SURFACE_ISOSURFACE_DIR="${DATA_DIR}/surface_isosurface"
SURFACE_COLLAPSED_DIR="${DATA_DIR}/surface_collapsed"
SURFACE_CUT_DIR="${DATA_DIR}/surface_cut"
SURFACE_REMESHED_DIR="${DATA_DIR}/surface_remeshed"
SURFACE_OPT_DIR="${DATA_DIR}/surface_opt"

# Ensure category directories exist
mkdir -p "$SURFACE_ISOSURFACE_DIR" "$SURFACE_COLLAPSED_DIR" \
         "$SURFACE_CUT_DIR" "$SURFACE_REMESHED_DIR" "$SURFACE_OPT_DIR"


trap '{
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
}' EXIT





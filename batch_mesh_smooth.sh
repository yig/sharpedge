#!/bin/bash

# ================================================
# Simple batch mesh smoothing
# ================================================
# Usage: ./batch_mesh_smooth.sh <normal_folder> <surface_folder> [output_folder]
# ================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 normal_folder surface_folder [output_folder]"
    echo "Example: $0 data/normal data/surface data/smoothed_surface"
    exit 1
fi

NORMAL_FOLDER="$1"
SURFACE_FOLDER="$2"
OUTPUT_FOLDER="${3:-${SURFACE_FOLDER}_smoothed}"

mkdir -p "$OUTPUT_FOLDER"

echo "Using normals: $NORMAL_FOLDER"
echo "Processing meshes: $SURFACE_FOLDER"
echo "Output to: $OUTPUT_FOLDER"
echo ""

for mesh_file in "$SURFACE_FOLDER"/*_isosurface_cleaned.obj; do
    [ ! -f "$mesh_file" ] && continue
    
    filename=$(basename "$mesh_file" _isosurface_cleaned.obj)
    normal_file="$NORMAL_FOLDER/${filename}.normal"
    output_file="$OUTPUT_FOLDER/${filename}_smoothed.obj"
    
    echo "Processing: $filename"
    
    if [ ! -f "$normal_file" ]; then
        echo "  ❌ Normal file not found: ${filename}.normal"
        continue
    fi
    
    python mesh_smooth.py "$normal_file" "$mesh_file" \
        --output "$output_file" --headless
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Success"
    else
        echo "  ❌ Failed"
    fi
    echo ""
done

echo "Done!"
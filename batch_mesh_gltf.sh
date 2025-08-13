#!/bin/bash

# ================================================
# Simple batch mesh gltf export
# ================================================
# Usage: ./batch_mesh_gltf.sh <sketch_folder> <surface_folder> [output_folder]
# ================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 sketch_folder surface_folder [output_folder]"
    echo "Example: $0 data/sketch data/surface gltf/good"
    exit 1
fi

SKETCH_FOLDER="$1"
SURFACE_FOLDER="$2"
OUTPUT_FOLDER="${3:-gltf/export}"

mkdir -p "$OUTPUT_FOLDER"

echo "Using sketches: $SKETCH_FOLDER"
echo "Using surfaces: $SURFACE_FOLDER"
echo "Output to: $OUTPUT_FOLDER"
echo ""

for surface_file in "$SURFACE_FOLDER"/*_smoothed.obj; do
    [ ! -f "$surface_file" ] && continue
    
    filename=$(basename "$surface_file" _smoothed.obj)
    # Remove _2n suffix to get original sketch name
    sketch_name=${filename%_2n}
    sketch_file="$SKETCH_FOLDER/${sketch_name}.obj"
    output_file="$OUTPUT_FOLDER/${sketch_name}_surface_v2.gltf"
    
    echo "Processing: $sketch_name"
    
    if [ -f "$sketch_file" ] && [ -f "$surface_file" ]; then
        echo "  Exporting surface for $sketch_name"
        python export_sketch_surface_gltf.py "$sketch_file" "$surface_file" --output "$output_file"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success"
        else
            echo "  ❌ Failed"
        fi
    else
        echo "  Warning: Missing file(s) for $sketch_name:"
        [ ! -f "$sketch_file" ] && echo "    ❌ $sketch_file sketch not found"
        [ ! -f "$surface_file" ] && echo "    ❌ $surface_file surface not found"
    fi
    echo ""
done

echo "Done!"
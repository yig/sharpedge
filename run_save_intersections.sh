#!/bin/bash

INPUT_DIR="3d-sketches/onshape"
OUTPUT_DIR="3d-sketches-processing/intersections_png"

mkdir -p "$OUTPUT_DIR"

for obj_file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$obj_file" .obj)
    echo "Processing $filename.obj ..."
    
    python sketch_intersections_viewer.py "$obj_file" --no-show \
        && mv intersection_vertices.png "$OUTPUT_DIR/${filename}_intersections.png"
done

echo "All done. Images saved in $OUTPUT_DIR/"

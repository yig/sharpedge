#!/bin/bash
INPUT_DIR="3d-sketches-processing/flowrep_resample"
OUTPUT_DIR="3d-sketches-processing/flowrep_center"
PNG_DIR="$OUTPUT_DIR/visualizations"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PNG_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    base_name=$(basename "$file" .obj)
    echo "Processing $filename"
    python t2f_center_and_scale.py "$file" "$OUTPUT_DIR/$filename" --png "$PNG_DIR/${base_name}.png"
done
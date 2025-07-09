#!/bin/bash

INPUT_DIR="3d-original-objs/Mingzou/SGA"
OUTPUT_DIR="3d-sketches-processing/mingzou"
SCRIPT="curve_to_obj.py"

mkdir -p "$OUTPUT_DIR"

for curve_file in "$INPUT_DIR"/*.curve; do
    filename=$(basename "$curve_file" .curve)
    output_file="$OUTPUT_DIR/$filename.obj"

    echo "Converting $filename.curve to $filename.obj"
    python "$SCRIPT" "$curve_file" "$output_file" 
done
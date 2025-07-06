#!/bin/bash

INPUT_DIR="3d-original-objs/True2Form"
OUTPUT_DIR="3d-sketches-processing/t2f"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_merge_lines.py "$file" "$OUTPUT_DIR/$filename"
done

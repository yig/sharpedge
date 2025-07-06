#!/bin/bash

INPUT_DIR="3d-sketches-processing/t2f"
OUTPUT_DIR="3d-sketches-processing/t2f_no_dup"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_remove_duplicates.py "$file" "$OUTPUT_DIR/$filename"
done

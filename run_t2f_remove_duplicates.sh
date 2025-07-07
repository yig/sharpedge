#!/bin/bash

INPUT_DIR="3d-sketches-processing/flowrep"
OUTPUT_DIR="3d-sketches-processing/flowrep_no_dup"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_remove_duplicates.py "$file" "$OUTPUT_DIR/$filename"
done

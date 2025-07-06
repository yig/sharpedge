#!/bin/bash

INPUT_DIR="3d-sketches-processing/t2f_no_dup"
OUTPUT_DIR="3d-sketches-processing/t2f_intersection"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_split_high_valence.py "$file" "$OUTPUT_DIR/$filename"
done

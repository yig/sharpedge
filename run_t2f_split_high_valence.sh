#!/bin/bash

INPUT_DIR="3d-sketches-processing/edge_split"
OUTPUT_DIR="3d-sketches-processing/haopan_intersection"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_split_high_valence.py "$file" "$OUTPUT_DIR/$filename"
done

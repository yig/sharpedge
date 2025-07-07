#!/bin/bash
INPUT_DIR="3d-sketches-processing/flowrep_center"
OUTPUT_DIR="3d-sketches-processing/edge_split"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Splitting high-valence edges in: $filename"
    python t2f_split_high_valence_edges.py "$file" "$OUTPUT_DIR/$filename"
done

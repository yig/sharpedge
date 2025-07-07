#!/bin/bash

INPUT_DIR="3d-sketches-processing/flowrep_no_dup"
OUTPUT_DIR="3d-sketches-processing/flowrep_resample"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file")
    echo "Processing $filename"
    python t2f_resample_rdp.py "$file" "$OUTPUT_DIR/$filename"
done

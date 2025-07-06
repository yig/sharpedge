#!/bin/bash

INPUT_DIR="3d-sketches-processing/t2f_align"

for file in "$INPUT_DIR"/*.obj; do
    echo "Processing $file"
    python t2f_rotate_by_index.py "$file" "$file"
done

echo "All files processed and overwritten!"
#!/bin/bash
INPUT_DIR="3d-sketches-processing/t2f_resample"
OUTPUT_BASE="3d-sketches-processing/t2f_align"
mkdir -p "$OUTPUT_BASE"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file" .obj)  # Remove .obj extension
    output_folder="$OUTPUT_BASE/$filename"
    echo "Processing $file -> $output_folder"
    python t2f_rotate_align_polyline.py "$file" "$output_folder" --visualize 
done

echo "All files processed!"
echo "Results saved in: $OUTPUT_BASE"
#!/bin/bash

# Check that the user provides a subfolder name
if [ $# -lt 1 ]; then
    echo "Usage: $0 subfolder_name"
    exit 1
fi

# Read the subfolder name
SUBFOLDER="$1"

# Define input and output directories
INPUT_DIR="3d-sketches/$SUBFOLDER"
OUTPUT_DIR="3d-sketches-processing/intersections_png"
mkdir -p "$OUTPUT_DIR"

# Step 1: Process each OBJ file in the input directory
for obj_file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$obj_file" .obj)
    echo "Processing $filename.obj ..."

    # Run intersection viewer script (no GUI)
    python sketch_intersections_viewer.py "$obj_file" --no-show

    # Move the output PNG to the output directory and rename
    mv intersection_vertices.png "$OUTPUT_DIR/${filename}_intersections.png"
done

echo "All .obj files processed."
echo "Images saved in $OUTPUT_DIR/"

# Step 2: Trim white space from all images in the output folder
echo "Trimming white space from images..."
python image_trim_white_space.py "$OUTPUT_DIR"

# Step 3: Merge trimmed images into a single grid image
echo "Merging images into final grid..."
python image_merge.py "$OUTPUT_DIR" --width 2100 --height 1500 --output sketches_latex.png

echo "Done! Final image saved as sketches_latex.png"

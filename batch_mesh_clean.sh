INPUT_DIR="signed-heat-3d/export"
OUTPUT_DIR="data/surface_cleaned"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$file" .obj)
    echo "Processing $filename.obj"
    python mesh_clean.py "$file" "$OUTPUT_DIR/${filename}_cleaned.obj"
done
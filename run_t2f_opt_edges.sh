#!/bin/bash

INPUT_DIR="3d-sketches/onshape"
OUTPUT_DIR="debug_normals/onshape"

mkdir -p "$OUTPUT_DIR"

for obj_file in "$INPUT_DIR"/*.obj; do
    filename=$(basename "$obj_file" .obj)
    echo "Processing $filename.obj ..."

    python opt_edges.py "$obj_file"

    # 移动与当前文件相关的所有 debug_normals 输出
    mv "debug_normals/${filename}_2n.normal" \
       "debug_normals/${filename}.gltf" \
       "debug_normals/${filename}_n0.gltf" \
       "debug_normals/${filename}_n1.gltf" \
       "$OUTPUT_DIR" 2>/dev/null
done

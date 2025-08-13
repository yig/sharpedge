#!/bin/bash

NORMAL_DIR="data/normal"
SURFACE_DIR="data/surface_cleaned"

for normal_file in "$NORMAL_DIR"/*.normal; do
    # 提取不带扩展名的文件名
    base=$(basename "$normal_file" .normal)

    # 对应的 surface 文件
    surface_file="$SURFACE_DIR/${base}_isosurface_cleaned.obj"

    if [[ -f "$surface_file" ]]; then
        echo "Processing $normal_file  +  $surface_file"
        ./myviewer "$normal_file" "$surface_file"
    else
        echo "Warning: $surface_file not found, skipping..."
    fi
done
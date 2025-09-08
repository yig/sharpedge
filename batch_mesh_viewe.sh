#!/bin/bash

NORMAL_DIR="data/normal"
# 如果用户传了第 1 个参数就用它，否则用默认值 data/surface_mvc
MESH_DIR="${1:-data/surface_mvc}"

shopt -s nullglob  # 确保找不到文件时不会返回字面字符串

for normal_file in "$NORMAL_DIR"/*.normal; do
    base=$(basename "$normal_file" .normal)

    matches=("$MESH_DIR/${base}_"*.obj)

    if [[ ${#matches[@]} -gt 0 ]]; then
        for surface_file in "${matches[@]}"; do
            echo "python sketch_normal_surface_viewer.py \"$normal_file\" \"$surface_file\""
            python sketch_normal_surface_viewer.py "$normal_file" "$surface_file"
        done
    else
        echo "Warning: no match for $base in $MESH_DIR, skipping..."
    fi
done

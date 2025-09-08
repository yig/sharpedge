NORMAL_DIR="data/normal"
MESH_DIR="data/surface_cut"
OUT_DIR="data/surface_mvc"

mkdir -p "$OUT_DIR"

shopt -s nullglob  # 确保找不到文件时不会返回字面字符串

for normal_file in "$NORMAL_DIR"/*.normal; do
    base=$(basename "$normal_file" .normal)

    matches=("$MESH_DIR/${base}_"*.obj)

    if [[ ${#matches[@]} -gt 0 ]]; then
        for surface_file in "${matches[@]}"; do
            # 提取原始 mesh 文件名（不带路径）
            surface_base=$(basename "$surface_file" .obj)
            # 构造输出文件名
            out_file="$OUT_DIR/${surface_base}_mvc.obj"

            # echo "python mesh_stanko_normal.py \"$surface_file\" \"$normal_file\" --add-boundary-flaps -o \"$out_file\""
            python mesh_stanko_normal.py "$surface_file" "$normal_file" --add-boundary-flaps -o "$out_file"
            # python mesh_stanko_normal.py "$surface_file" --add-boundary-flaps -o "$out_file"
        done
    else
        echo "Warning: no match for $base in $MESH_DIR, skipping..."
    fi
done
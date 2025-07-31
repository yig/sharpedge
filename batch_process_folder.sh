#!/bin/bash

# ================================================
# Batch processing pipeline for sketch-to-GLTF
# ================================================
# Usage:
#   ./this_script.sh <subfolder_name> [step1|step2|step3|step4|all]
#
# Arguments:
#   <subfolder_name>   Subfolder inside '3d-sketches/' to process
#   [stepX|all]        Optional step to run (default: all)
#
# Steps:
#   STEP 1 - Export .obj sketches to GLTF format
#   STEP 2 - Run opt_edges.py to compute normal directions and move GLTFs
#   STEP 3 - Compute signed heat SDF from normals using C++ binary
#   STEP 4 - Generate surface mesh from SDF and export combined GLTF
#
# Output files are saved to:
#   gltf/<subfolder_name>/...
# ================================================


if [ $# -lt 1 ]; then
    echo "Usage: $0 subfolder_name [step1|step2|step3|all] (default: all)"
    exit 1
fi

SUBFOLDER="$1"
STEP=${2:-all}  # Default is 'all' if not provided

INPUT_DIR="3d-sketches/$SUBFOLDER"
OUTPUT_DIR="gltf/$SUBFOLDER"
mkdir -p "$OUTPUT_DIR"

# ---- STEP 1: Export sketch polylines to GLTF ----
if [ "$STEP" = "step1" ] || [ "$STEP" = "all" ]; then
    echo "=== Running STEP 1: export_sketch_gltf ==="
    for filepath in "$INPUT_DIR"/*.obj; do
        filename=$(basename "$filepath" .obj)
        output_file="${OUTPUT_DIR}/${filename}_sketch.gltf"
        echo "Exporting: $filepath -> $output_file"
        python export_sketch_gltf.py "$filepath" -o "$output_file"
    done
fi

# ---- STEP 2: Run opt_edges.py and move GLTFs + .normal ----
if [ "$STEP" = "step2" ] || [ "$STEP" = "all" ]; then
    echo "=== Running STEP 2: opt_edges + move normal gltfs ==="
    mkdir -p debug_normals
    for filepath in "$INPUT_DIR"/*.obj; do
        filename=$(basename "$filepath" .obj)
        echo "Running opt_edges on: $filepath"
        python opt_edges.py "$filepath" "-i" "false"

        # Move normal gltfs
        for suffix in n0 n1 2n; do
            src_file="debug_normals/${filename}_${suffix}.gltf"
            if [ -f "$src_file" ]; then
                mv "$src_file" "$OUTPUT_DIR/"
                echo "Moved $src_file -> $OUTPUT_DIR/"
            else
                echo "Warning: $src_file not found"
            fi
        done

        # Move .normal file to signed-heat-3d/data
        normal_file="debug_normals/${filename}_2n.normal"
        if [ -f "$normal_file" ]; then
            mv "$normal_file" "signed-heat-3d/data/${filename}_2n.normal"
            echo "Moved $normal_file -> signed-heat-3d/data/"
        else
            echo "Warning: $normal_file not found"
        fi
    done
fi

# ---- STEP 3: Run signed-heat computation ----
if [ "$STEP" = "step3" ] || [ "$STEP" = "all" ]; then
    echo "=== Running STEP 3: signed-heat computation ==="
    for filepath in "$INPUT_DIR"/*.obj; do
        filename=$(basename "$filepath" .obj)
        normal_input="signed-heat-3d/data/${filename}_2n.normal"

        if [ -f "$normal_input" ]; then
            echo "Running signed heat on: $normal_input"
            (
                cd signed-heat-3d/build || exit 1
                ./bin/main "../data/${filename}_2n.normal" --headless
            )
        else
            echo "Warning: $normal_input not found"
        fi
    done
fi


# ---- STEP 4: Export surface + sketch as GLTF ----
if [ "$STEP" = "step4" ] || [ "$STEP" = "all" ]; then
    echo "=== Running STEP 4: Export sketch + surface GLTF ==="

    for filepath in "$INPUT_DIR"/*.obj; do
        filename=$(basename "$filepath" .obj)

        obj_file="3d-sketches/$SUBFOLDER/${filename}.obj"
        surface_file="signed-heat-3d/export/${filename}_2n_isosurface.obj"
        output_file="${OUTPUT_DIR}/${filename}_surface.gltf"

        if [ -f "$obj_file" ] && [ -f "$surface_file" ]; then
            echo "Exporting surface for $filename"
            python export_sketch_surface_gltf.py "$obj_file" "$surface_file" --output "$output_file"
        else
            echo "Warning: Missing file(s) for $filename:"
            [ ! -f "$obj_file" ] && echo "  ❌ $obj_file obj not found"
            [ ! -f "$sdf_file" ] && echo "  ❌ $surface_file  surface not found"
        fi
    done
fi


#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/data"

mkdir -p "$OUT"

echo "Installing tools to: $OUT"
echo

# ------------------------------------------------------------
# cut_mesh
# ------------------------------------------------------------
echo "Building cut_mesh..."
cmake -S "$ROOT/cut_edges" -B "$ROOT/cut_edges/build"
cmake --build "$ROOT/cut_edges/build"
cmake --install "$ROOT/cut_edges/build" --prefix "$OUT"
echo

# ------------------------------------------------------------
# cgal_remesh
# ------------------------------------------------------------
echo "Building cgal_remesh..."
cmake -S "$ROOT/cgal_remesh" -B "$ROOT/cgal_remesh/build"
cmake --build "$ROOT/cgal_remesh/build"
cmake --install "$ROOT/cgal_remesh/build" --prefix "$OUT"
echo

# ------------------------------------------------------------
# signed-heat-3d (submodule)
# ------------------------------------------------------------
echo "Building signed_heat_3d..."
cmake -S "$ROOT/signed-heat-3d" -B "$ROOT/signed-heat-3d/build"
cmake --build "$ROOT/signed-heat-3d/build"
cmake --install "$ROOT/signed-heat-3d/build" --prefix "$OUT"
echo

echo "✅ All tools installed to:"
ls -1 "$OUT"

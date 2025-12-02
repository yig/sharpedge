# README

## Optimize Normals

Run `opt_edges.py` and it will show the usage.

Example:

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj
```

With `uv`:
```
uv run --with-requirements requirements.freeze.txt --python 3.12 opt_edges.py sketch/onshape_simple_mouse.obj
```

The optimized normals will be saved in:

```bash
data/normal/
```

---

## Generate the Surface

Make sure the following binaries are compiled and placed in the `data` directory:

-   `signed-heat-3d`: `cd signed-heat-3d && cmake -B build-dir -G Ninja && cmake --build build-dir; cd ../data; ln -s ../signed-heat-3d/build-dir/bin/main`
-   `cgal_remesh`: `cd cgal_remesh && cmake -B build-dir -G Ninja && cmake --build build-dir; cd ../data; ln -s ../cgal_remesh/build-dir/cgal_remesh`
-   `cut_edges`: `cd cut_edges && cmake -B build-dir -G Ninja && cmake --build build-dir; cd ../data; ln -s ../cut_edges/build-dir/bin/cut_mesh`

Then run:

```bash
sh normal_to_mesh.sh data/normal/onshape_simple_mouse_2n.normal
```

All generated surface results will appear in:

```bash
data/onshape_simple_mouse/
```

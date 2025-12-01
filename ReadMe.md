# README

## Optimize Normals

Run `opt_edges.py` and it will show the usage.  

Example:

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj
```

The optimized normals will be saved in:

```bash
data/normal/
```

---

## Generate the Surface

Make sure the following binaries are compiled and placed in the `data` directory:

-   `signed-heat-3d`
    
-   `cgal_remesh`
    
-   `cut_edges`
    

Then run:

```bash
sh normal_to_mesh.sh data/normal/onshape_simple_mouse_2n.normal
```

All generated surface results will appear in:

```bash
data/onshape_simple_mouse/
```

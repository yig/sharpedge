# README

## Optimize Normals

Run `opt_edges.py` and it will show the usage.  

Example:

```
python opt_edges.py sketch/onshape_simple_mouse.obj
```

Or run with visualization

```
python opt_edges.py sketch/onshape_simple_mouse.obj --show-plot
```

The optimized normals will be saved in:

```
data/normal/
```

(The directory will be created automatically if it does not exist.)


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

```
data/onshape_simple_mouse/
```

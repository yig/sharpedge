# README

### Clone (private)

```bash
git clone --recurse-submodules git@github.com:KrisYu/marching_cube.git
```


## Optimize Normals

Run `opt_edges.py` and it will show the usage.  

Example:

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj
```

Or run with visualization

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj --show-plot
```

The optimized normals will be saved in:

```bash
data/normal/
```

(The directory will be created automatically if it does not exist.)


---

## Generate the Surface

Build tools:

```bash
sh build_tools.sh
```

The following binaries will be compiled and placed in the `data` directory:

-   `signed_heat_3d`
    
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

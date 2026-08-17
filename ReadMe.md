# README

### Clone 

```bash
git clone --recurse-submodules git@github.com:KrisYu/SharpEdge.git
```


## Optimize Normals

Run `opt_edges.py` and it will show the usage.  

Example:

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj
```

With `uv`:
```bash
uv run --with-requirements requirements.freeze.minimal.txt --python 3.12 opt_edges.py sketch/onshape_simple_mouse.obj
```

Or run with visualization:

```bash
python opt_edges.py sketch/onshape_simple_mouse.obj --show-plot
```

With `uv`:
```
uv run --with-requirements requirements.freeze.txt --python 3.12 opt_edges.py sketch/onshape_simple_mouse.obj --show-plot
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


Specify target edge lengths for surface generation and target length for remesh:


```bash
sh normal_to_mesh.sh data/normal/bowl_curve_2n.normal 0.05
sh normal_to_mesh.sh data/normal/bowl_curve_2n.normal 0.06 0.02
```

- 2nd argument: `TARGET_EDGE_LENGTH` (default: `0.04`)
- 3rd argument: `TARGET_REMESH_LENGTH` (default: `0.015`)


All generated surface results will appear in:

```bash
data/onshape_simple_mouse/
```

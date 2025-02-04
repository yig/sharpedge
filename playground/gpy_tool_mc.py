#!/usr/bin/env python3

import gpytoolbox
import numpy as np 
import polyscope as ps 

# Some scalar function fun
def fun(V):
    return np.sum(V**2,axis=1)

def fun2(V):
    x, y, z = V[:,0], V[:,1], V[:,2]
    return np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(x)

def fun3(V):
    x, y, z = V[:,0], V[:,1], V[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r * (1 + 0.5 * np.sin(3*theta) * np.cos(4*phi))

def fun4(V):
    centers = np.array([
        [0.3, 0.3, 0.3],
        [-0.3, -0.3, -0.3],
        [0.3, -0.3, 0.3]
    ])
    field = np.zeros(len(V))
    for center in centers:
        dist = np.sum((V - center)**2, axis=1)
        field += 1.0 / (dist + 1e-6)
    return field

def fun5(V):
    x, y, z = V[:,0], V[:,1], V[:,2]
    sources = np.array([
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5],
    ])
    field = np.zeros(len(V))
    for source in sources:
        dist = np.sqrt(np.sum((V - source)**2, axis=1))
        field += np.sin(10 * dist) / (dist + 1e-6)
    return field

def fun6(V):
    x, y, z = V[:,0], V[:,1], V[:,2]
    r = np.sqrt(x**2 + y**2)
    major_r = 0.5
    minor_r = 0.2
    theta = np.arctan2(y, x)
    return (r - major_r)**2 + (z - minor_r*np.sin(3*theta))**2 - minor_r**2

def fun7(V):
    x, y, z = V[:,0], V[:,1], V[:,2]
    return np.sin(x*5)*np.cos(y*5)*np.sin(z*5) + \
           0.5*np.sin(x*10)*np.cos(y*10)*np.sin(z*10) + \
           0.25*np.sin(x*20)*np.cos(y*20)*np.sin(z*20)

def scale_grid(V, scale=2.0, center=True):
    """
    Scale the grid vertices and optionally center them
    V: vertices [N,3]
    scale: how much to scale the domain
    center: whether to center the domain around origin
    """
    V_scaled = V * scale
    if center:
        V_scaled = V_scaled - scale/2
    return V_scaled

# # Now your main code becomes:
# GV,_ = gpytoolbox.regular_cube_mesh(100)
# GV = scale_grid(GV, scale=4.0)  # Makes domain go from -2 to 2 instead of 0 to 1

# # Generate a grid
# # GV,_ = gpytoolbox.regular_cube_mesh(100)
# # # Evaluate scalar function on grid
# S = fun3(GV)
# # Compute isosurface
# V,F = gpytoolbox.marching_cubes(S,GV,100,100,100,0.5)

# ps.init()
# ps_mesh = ps.register_surface_mesh("my mesh", V, F)
# ps.set_ground_plane_mode("none")
# ps.show()


def show_all_functions():
    # List of all functions and their recommended isovalues
    function_list = [
        (fun, "Simple Sphere", 0.5),
        (fun2, "Gyroid", 0.0),
        (fun3, "Spherical Harmonics", 1.0),
        (fun4, "Metaballs", 2.0),
        (fun5, "Wave Interference", 0.0),
        (fun6, "Twisted Torus", 0.0),
        (fun7, "Fractal Surface", 0.0)
    ]
    
    # Initialize grid once
    GV,_ = gpytoolbox.regular_cube_mesh(100)
    GV = scale_grid(GV, scale=4.0)
    
    # Initialize polyscope
    ps.init()
    
    # Process each function
    for func, name, isovalue in function_list:
        # Evaluate function
        S = func(GV)
        
        # Compute isosurface
        V,F = gpytoolbox.marching_cubes(S,GV,100,100,100,isovalue)
        
        # Register mesh with unique name
        ps_mesh = ps.register_surface_mesh(name, V, F)
        
        # Set visualization options
        ps.set_ground_plane_mode("none")
        
        # Show the mesh
        print(f"\nShowing: {name}")
        print("Close the window to see the next shape")
        ps.show()
        
        # Remove the current mesh before showing the next one
        ps.remove_all_structures()

# Run the visualization
if __name__ == "__main__":
    show_all_functions()

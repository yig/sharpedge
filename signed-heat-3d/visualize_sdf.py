import pandas as pd
import numpy as np
import polyscope as ps
from skimage import measure
import argparse

def visualize_sdf_with_polyscope(csv_file, iso_value=0.0, show_grid=True, show_isosurface=True):
    """
    Visualize SDF data from CSV using Polyscope
    
    Args:
        csv_file: Path to CSV file with columns xCoord, yCoord, zCoord, SDF
        iso_value: Isosurface level to extract (default: 0.0)
        show_grid: Whether to show the volume grid with SDF values
        show_isosurface: Whether to show the extracted isosurface
    """
    
    # Read CSV data
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Extract coordinates and SDF values
    x_coords = df['xCoord'].values
    y_coords = df['yCoord'].values
    z_coords = df['zCoord'].values
    sdf_values = df['SDF'].values
    
    print(f"Loaded {len(df)} grid points")
    print(f"SDF range: [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")
    
    # Determine grid dimensions
    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    z_unique = np.sort(np.unique(z_coords))
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Initialize Polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    
    # Show volume grid with SDF values
    if show_grid:
        print("Creating volume grid...")
        
        # Calculate grid bounds
        bound_min = [x_unique[0], y_unique[0], z_unique[0]]
        bound_max = [x_unique[-1], y_unique[-1], z_unique[-1]]
        
        # Reshape SDF values into 3D grid
        try:
            # Try standard ordering: for k in nz: for j in ny: for i in nx
            sdf_grid = sdf_values.reshape((nz, ny, nx))
        except ValueError:
            try:
                # Try alternative ordering: for i in nx: for j in ny: for k in nz
                sdf_grid = sdf_values.reshape((nx, ny, nz))
                sdf_grid = np.transpose(sdf_grid, (2, 1, 0))  # Reorder to (nz, ny, nx)
            except ValueError:
                print("Error: Cannot reshape data into regular grid")
                return
        
        # Register volume grid with Polyscope
        ps_grid = ps.register_volume_grid("SDF Volume", 
                                        [nx, ny, nz],
                                        bound_min, 
                                        bound_max,
                                        enabled=False)
        
        # Add SDF as scalar quantity
        ps_grid.add_scalar_quantity("SDF", sdf_grid, 
                                  defined_on='nodes', 
                                  enabled=True,
                                  cmap='coolwarm')
        
        print(f"Added volume grid with SDF values")
    
    # Extract and show isosurface
    if show_isosurface:
        print(f"Extracting isosurface at level {iso_value}...")
        
        try:
            # Grid spacing
            dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
            dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
            dz = z_unique[1] - z_unique[0] if len(z_unique) > 1 else 1.0
            spacing = (dx, dy, dz)
            
            # Extract isosurface using marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                sdf_grid, 
                level=iso_value, 
                spacing=spacing
            )
            
            # Translate vertices to correct world coordinates
            vertices[:, 0] += x_unique[0]
            vertices[:, 1] += y_unique[0]
            vertices[:, 2] += z_unique[0]
            
            print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Register isosurface mesh with Polyscope
            ps_mesh = ps.register_surface_mesh(f"Isosurface (level={iso_value})", 
                                             vertices, faces)
            
            # Add vertex normals
            ps_mesh.add_vector_quantity("normals", normals, 
                                      defined_on='vertices', 
                                      enabled=False)
            
            # Color by distance from iso-level
            ps_mesh.add_scalar_quantity("distance_from_iso", values,
                                      defined_on='vertices',
                                      enabled=False)
            
            # Set some nice visual properties
            ps_mesh.set_color([0.8, 0.2, 0.2])  # Red color
            ps_mesh.set_transparency(0.8)
            
        except ValueError as e:
            print(f"Could not extract isosurface: {e}")
    
    # Show interactive viewer
    print("\nStarting Polyscope viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Click on volume grid to see SDF slice views")
    print("  - Use GUI panels to adjust visualization settings")
    
    ps.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize SDF data from CSV using Polyscope')
    parser.add_argument('csv_file', help='Input CSV file path')
    parser.add_argument('-i', '--iso', type=float, default=0.0, 
                       help='Iso-value for surface extraction (default: 0.0)')
    parser.add_argument('--no-grid', action='store_true', 
                       help='Don\'t show volume grid')
    parser.add_argument('--no-surface', action='store_true', 
                       help='Don\'t show isosurface')
    
    args = parser.parse_args()
    
    visualize_sdf_with_polyscope(
        csv_file=args.csv_file,
        iso_value=args.iso,
        show_grid=not args.no_grid,
        show_isosurface=not args.no_surface
    )

if __name__ == "__main__":
    main()

# Example usage:
# python visualize_sdf.py bowl_curve_2n.csv
# python visualize_sdf.py bowl_curve_2n.csv -i 0.1
# python visualize_sdf.py bowl_curve_2n.csv --no-grid  # Only show isosurface
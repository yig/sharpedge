import polyscope as ps 
import numpy as np 
import matplotlib.pyplot as plt
import colorsys


def plot_tet_mesh(vertices, tets):
    """
    Plot the tetrahedral mesh using Polyscope, showing only the edges.
    
    Args:
        vertices: Nx3 array of vertex coordinates
        tets: Mx4 array of tetrahedral connectivity
    """

    vertices = np.asarray(vertices)
    tets = np.asarray(tets)
    
    # Clear any existing structures
    # ps.remove_all_structures()
    
    # Initialize polyscope
    ps.init()
    
    # Register the tetrahedral mesh
    ps_mesh = ps.register_volume_mesh("tet mesh", vertices, tets)
    
    # Hide the tet faces
    ps_mesh.set_enabled(False)
    
    # Create rainbow colors for tets
    tet_colors = plt.cm.rainbow(np.linspace(0, 1, len(tets)))[:, :3]  # Get RGB values
    
    # Add colors to the tets
    ps_mesh.add_color_quantity("tet colors", tet_colors, defined_on='cells', enabled=True)
    
    # Make tets fully transparent
    ps_mesh.set_transparency(0.08)  # 1.0 is fully transparent
    
    # Create edges array
    edges = get_tet_edges(tets)
    
    # Register the edges as a curve network
    ps_edges = ps.register_curve_network("tet edges", vertices, edges)
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # You can customize the edge appearance
    ps_edges.set_radius(0.001)  # Make edges thinner
    ps_edges.set_color((0.1, 0.1, 0.8))  # Set edge color (RGB)
    
    # Set the camera angle
    # ps.reset_camera_to_home_view()
    
    # Show the mesh
    ps.show()

def get_tet_edges(tets):
    """Helper function to extract unique edges from tetrahedral mesh"""
    edges = set()
    for tet in tets:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([tet[i], tet[j]]))
                edges.add(edge)
    return np.array(list(edges))


def plot_normal_data(V, E, N):
    """
    Plot edges and their normal vectors using Polyscope:
    - Edges shown as curves
    - Normal vectors shown as vectors at edge midpoints
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        N: (m,3) array of normal vectors for edges
    """
    ps.init()


    # Register the point cloud
    ps_points = ps.register_point_cloud("vertices", V)
    ps_points.set_color((0.0, 0.0, 1.0))  # Blue color for vertices
    ps_points.set_radius(0.002)  # Small point size
    
    # Create curve network for edges
    ps_edges = ps.register_curve_network("edges", V, E)
    ps_edges.set_color((0.0, 0.0, 0.0))  # Black color for edges
    ps_edges.set_radius(0.001)  # Edge thickness
    
    # Calculate edge midpoints
    midpoints = (V[E[:, 0]] + V[E[:, 1]]) / 2
    
    # Register vectors at midpoints
    ps_vectors = ps.register_point_cloud("normals", midpoints)
    ps_vectors.set_radius(0.0005)  # Much smaller radius for midpoints
    ps_vectors.set_color((0.0, 1.0, 0.0))  # Match color to normal vectors
    # ps_vectors.set_enabled(False)  # Hide the actual midpoints
    ps_vectors.add_vector_quantity("normal_vectors", N, 
                                 enabled=True,
                                 color=(0.0, 1.0, 0.0),
                                 length=0.10)  # Green color for normals
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # Show the window
    ps.show()

def plot_two_normals(V, E, N1, N2):
    """
    Plot edges and two sets of normal vectors using Polyscope:
    - Edges shown as curves
    - Two different normal vectors shown at edge midpoints with different colors
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        N1: (m,3) array of first set of normal vectors for edges
        N2: (m,3) array of second set of normal vectors for edges
    """
    ps.init()
    
    # Register the point cloud
    ps_points = ps.register_point_cloud("vertices", V)
    ps_points.set_color((0.0, 0.0, 1.0))  # Blue color for vertices
    ps_points.set_radius(0.002)  # Small point size
    
    # Create curve network for edges
    ps_edges = ps.register_curve_network("edges", V, E)
    ps_edges.set_color((0.0, 0.0, 0.0))  # Black color for edges
    ps_edges.set_radius(0.001)  # Edge thickness
    
    # Calculate edge midpoints
    midpoints = (V[E[:, 0]] + V[E[:, 1]]) / 2
    
    # Register vectors at midpoints for first normal set
    ps_vectors1 = ps.register_point_cloud("normals1", midpoints)
    ps_vectors1.set_radius(0.0005)  # Much smaller radius for midpoints
    ps_vectors1.set_color((0.0, 1.0, 0.0))  # Green for first set
    ps_vectors1.add_vector_quantity("normal_vectors1", N1,
                                   enabled=True,
                                   color=(0.0, 1.0, 0.0),
                                   length=0.10)  # Green color for first normals
    
    # Register vectors at midpoints for second normal set
    ps_vectors2 = ps.register_point_cloud("normals2", midpoints)
    ps_vectors2.set_radius(0.0005)  # Much smaller radius for midpoints
    ps_vectors2.set_color((1.0, 0.0, 0.0))  # Red for second set
    ps_vectors2.add_vector_quantity("normal_vectors2", N2,
                                   enabled=True,
                                   color=(1.0, 0.0, 0.0),
                                   length=0.10)  # Red color for second normals
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # Show the window
    ps.show()

def plot_different_normals(V, E, N1, N2, angle_threshold_degrees=20):
    """
    Plot edges and only the normal vectors that differ by more than the specified angle threshold
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex pairs
        N1: (m,3) array of first set of normal vectors for edges
        N2: (m,3) array of second set of normal vectors for edges
        angle_threshold_degrees: Minimum angle difference to consider normals "very different"
    """
    # Convert angle threshold to radians
    angle_threshold = np.deg2rad(angle_threshold_degrees)
    
    # Calculate dot products between normalized vectors to get cosines of angles
    N1_normalized = N1 / np.linalg.norm(N1, axis=1, keepdims=True)
    N2_normalized = N2 / np.linalg.norm(N2, axis=1, keepdims=True)
    
    dot_products = np.sum(N1_normalized * N2_normalized, axis=1)
    # Clip to valid range for arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    # Calculate angles
    angles = np.arccos(dot_products)
    
    # Find indices where angle difference exceeds threshold
    different_indices = np.where(angles > angle_threshold)[0]
    
    # Calculate edge midpoints
    midpoints = (V[E[:, 0]] + V[E[:, 1]]) / 2
    
    # Only select midpoints and normals at the different indices
    different_midpoints = midpoints[different_indices]
    different_N1 = N1[different_indices]
    different_N2 = N2[different_indices]
    
    ps.init()
    # Register the point cloud
    ps_points = ps.register_point_cloud("vertices", V)
    ps_points.set_color((0.0, 0.0, 1.0))  # Blue color for vertices
    ps_points.set_radius(0.002)  # Small point size
    
    # Create curve network for edges
    ps_edges = ps.register_curve_network("edges", V, E)
    ps_edges.set_color((0.0, 0.0, 0.0))  # Black color for edges
    ps_edges.set_radius(0.001)  # Edge thickness
    
    if len(different_indices) > 0:
        # Register vectors at midpoints for first normal set (at different locations only)
        ps_vectors1 = ps.register_point_cloud("different_normals1", different_midpoints)
        ps_vectors1.set_radius(0.0005)  # Much smaller radius for midpoints
        ps_vectors1.set_color((0.8, 0.2, 0.2))  # Red for first set
        ps_vectors1.add_vector_quantity("normal_vectors1", different_N1,
                                       enabled=True,
                                       color=(1.0, 0.0, 0.0),
                                       length=0.10)  # Red color for first normals
        
        # Register vectors at midpoints for second normal set (at different locations only)
        ps_vectors2 = ps.register_point_cloud("different_normals2", different_midpoints)
        ps_vectors2.set_radius(0.0005)  # Much smaller radius for midpoints
        ps_vectors2.set_color((0.2, 0.2, 0.8))  # Blue for second set
        ps_vectors2.add_vector_quantity("normal_vectors2", different_N2,
                                       enabled=True,
                                       color=(0.0, 0.0, 1.0),
                                       length=0.10)  # Blue color for second normals
    
    # Set visualization options
    ps.set_ground_plane_mode("none")
    
    # Print some statistics
    print(f"Total number of normal pairs: {len(N1)}")
    print(f"Number of significantly different normal pairs: {len(different_indices)}")
    print(f"Percentage of different normals: {100 * len(different_indices) / len(N1):.2f}%")
    
    # Show the window
    ps.show()

    
def plot_cdt_skecth(vertices, lines):
    """
    Create a 3D visualization of the vertices and lines using Polyscope,
    with each line in a different color.
    
    Args:
        vertices: nx3 array of vertex coordinates
        lines: mx2 array of line vertex indices
    """
    # Initialize polyscope
    ps.init()
    
    # Register the point cloud
    points = ps.register_point_cloud("vertices", vertices)
    points.set_color((0, 0, 0))  # Set points to black
    points.set_radius(0.0055)  # Adjust point size
    
    # Convert lines to the format Polyscope expects (Nx2 array of indices)
    edges = np.array(lines)
    
    # Generate unique colors for each line using HSV color space
    num_lines = len(edges)
    colors = []
    for i in range(num_lines):
        # Use HSV color space for better color distribution
        # H: varies from 0 to 1 (hue)
        # S: constant at 0.8 (saturation)
        # V: constant at 0.8 (value)
        hue = i / num_lines
        
        # Convert HSV to RGB
        rgb_color = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append(rgb_color)
    
    # Convert colors to numpy array
    colors = np.array(colors)
    
    # Register individual curve networks for each line
    for i, (edge, color) in enumerate(zip(edges, colors)):
        # Create a single-edge curve network
        single_edge = np.array([edge])
        network = ps.register_curve_network(f"line_{i}", vertices, single_edge)
        network.set_color(color)
        network.set_radius(0.005)  # Adjust line thickness
    
    # Set some reasonable camera parameters
    ps.set_ground_plane_mode("none")  # Remove ground plane
    ps.set_navigation_style("turntable")
    
    # Show the visualization
    ps.show()

def plot_cdt_skecth_with_polylines(vertices, edges, polylines=None):
    """
    Create a 3D visualization with different colors for polylines.
    
    Args:
        vertices: nx3 array of vertex coordinates
        edges: mx2 array of edge indices
        polylines: list of arrays containing vertex indices for each polyline
    """    
    # Initialize polyscope
    ps.init()
    
    # Register the point cloud
    points = ps.register_point_cloud("vertices", vertices)
    points.set_color((0, 0, 0))  # Set points to black
    points.set_radius(0.0055)  # Adjust point size
    
    if polylines is not None:
        # Plot each polyline in a different color
        for i, polyline in enumerate(polylines):
            # Generate edges from polyline
            poly_edges = np.array([[polyline[j], polyline[j+1]] 
                                 for j in range(len(polyline)-1)])
            
            # Generate color using HSV color space
            hue = i / len(polylines)
            rgb_color = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            
            # Register curve network for this polyline
            network = ps.register_curve_network(f"polyline_{i}", 
                                             vertices, 
                                             poly_edges)
            network.set_color(rgb_color)
            network.set_radius(0.005)
    else:
        # If no polylines provided, plot edges with different colors
        for i, edge in enumerate(edges):
            # Generate color
            hue = i / len(edges)
            rgb_color = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            
            # Create single-edge curve network
            single_edge = np.array([edge])
            network = ps.register_curve_network(f"line_{i}", 
                                             vertices, 
                                             single_edge)
            network.set_color(rgb_color)
            network.set_radius(0.0005)
    
    # Set visualization parameters
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("turntable")
    
    # Show the visualization
    ps.show()
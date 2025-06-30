import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_edge_circulation(edge_indices, vertex_idx, E, V):
    """
    Compute the counter-clockwise circulation ordering of edges around a vertex.
    """
    vertex_pos = V[vertex_idx]
    # Filter for incident edges
    incident_edges = [ei for ei in edge_indices if vertex_idx in E[ei]]
    
    if len(incident_edges) <= 2:
        return incident_edges
    
    vectors = []
    vector_array = np.zeros((len(incident_edges), 3))
    
    for i, ei in enumerate(incident_edges):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        vec = V[other_idx] - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, ei))
        vector_array[i] = unit_vec
    
    # Use SVD to find the best 2D plane
    centered_data = vector_array
    _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
    basis_1, basis_2 = Vt[0], Vt[1]
    
    projected_vectors = []
    for unit_vec, ei in vectors:
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, ei))
    
    # Normalize to [0, 2π) range first
    normalized_vectors = []
    for angle, ei in projected_vectors:
        normalized_angle = (angle + 2*np.pi) % (2*np.pi)
        print('normalized_angle', normalized_angle)
        normalized_vectors.append((normalized_angle, ei))

    normalized_vectors.sort(key=lambda x: x[0])  # Correct!
    ordered_edges = [ei for _, ei in projected_vectors]
    
    return ordered_edges

def visualize_edge_circulation(V, E, central_vertex=0):
    """
    Create 3D and 2D visualizations of the edge circulation
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D visualization
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot all vertices
    for i, vertex in enumerate(V):
        color = 'red' if i == central_vertex else 'blue'
        size = 100 if i == central_vertex else 60
        ax1.scatter(vertex[0], vertex[1], vertex[2], 
                   c=color, s=size, alpha=0.8)
        ax1.text(vertex[0], vertex[1], vertex[2] + 0.1, 
                f'V{i}', fontsize=10)
    
    # Plot edges
    for i, (v1, v2) in enumerate(E):
        start, end = V[v1], V[v2]
        ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                'k-', linewidth=2, alpha=0.7)
        
        # Label edges at midpoint
        mid = (start + end) / 2
        ax1.text(mid[0], mid[1], mid[2], f'E{i}', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Compute SVD to find the projection plane
    vertex_pos = V[central_vertex]
    edge_indices = list(range(len(E)))
    incident_edges = [ei for ei in edge_indices if central_vertex in E[ei]]
    
    if len(incident_edges) > 2:
        # Create direction vectors for SVD
        vector_array = np.zeros((len(incident_edges), 3))
        for i, ei in enumerate(incident_edges):
            v1, v2 = E[ei]
            other_idx = v2 if v1 == central_vertex else v1
            vec = V[other_idx] - vertex_pos
            unit_vec = vec / np.linalg.norm(vec)
            vector_array[i] = unit_vec
        
        # Perform SVD to get the projection plane
        _, _, Vt = np.linalg.svd(vector_array, full_matrices=False)
        basis_1, basis_2 = Vt[0], Vt[1]  # First two principal components
        normal = Vt[2] if len(Vt) > 2 else np.cross(basis_1, basis_2)  # Plane normal
        
        # Create plane mesh centered at the central vertex
        plane_size = 2.0  # Size of the plane to draw
        u = np.linspace(-plane_size, plane_size, 10)
        v = np.linspace(-plane_size, plane_size, 10)
        U, V_mesh = np.meshgrid(u, v)
        
        # Plane equation: point = center + u*basis_1 + v*basis_2
        plane_points = vertex_pos[np.newaxis, np.newaxis, :] + \
                      U[:, :, np.newaxis] * basis_1[np.newaxis, np.newaxis, :] + \
                      V_mesh[:, :, np.newaxis] * basis_2[np.newaxis, np.newaxis, :]
        
        X_plane = plane_points[:, :, 0]
        Y_plane = plane_points[:, :, 1]
        Z_plane = plane_points[:, :, 2]
        
        # Draw the projection plane
        ax1.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='cyan', 
                        label='SVD Projection Plane')
        
        # Draw the basis vectors on the plane
        scale = 1.5
        # Basis 1 vector (red)
        ax1.quiver(vertex_pos[0], vertex_pos[1], vertex_pos[2],
                  basis_1[0]*scale, basis_1[1]*scale, basis_1[2]*scale,
                  color='red', arrow_length_ratio=0.1, linewidth=3, 
                  label='Basis 1')
        
        # Basis 2 vector (green)
        ax1.quiver(vertex_pos[0], vertex_pos[1], vertex_pos[2],
                  basis_2[0]*scale, basis_2[1]*scale, basis_2[2]*scale,
                  color='green', arrow_length_ratio=0.1, linewidth=3,
                  label='Basis 2')
        
        # Normal vector (purple)
        ax1.quiver(vertex_pos[0], vertex_pos[1], vertex_pos[2],
                  normal[0]*scale, normal[1]*scale, normal[2]*scale,
                  color='purple', arrow_length_ratio=0.1, linewidth=3,
                  label='Normal')
        
        # Project direction vectors onto the plane and draw them
        for i, ei in enumerate(incident_edges):
            v1, v2 = E[ei]
            other_idx = v2 if v1 == central_vertex else v1
            vec = V[other_idx] - vertex_pos
            unit_vec = vec / np.linalg.norm(vec)
            
            # Project onto the plane
            proj_1 = np.dot(unit_vec, basis_1)
            proj_2 = np.dot(unit_vec, basis_2)
            projected_vec = proj_1 * basis_1 + proj_2 * basis_2
            
            # Draw projection on the plane
            end_point = vertex_pos + projected_vec * 1.2
            ax1.plot([vertex_pos[0], end_point[0]], 
                    [vertex_pos[1], end_point[1]], 
                    [vertex_pos[2], end_point[2]], 
                    'orange', linewidth=2, linestyle='--', alpha=0.8)
        
        # Add legend
        ax1.legend(loc='upper right', fontsize=8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: SVD Projection Plane')
    
    # Compute circulation
    edge_indices = list(range(len(E)))
    ordered_edges = compute_edge_circulation(edge_indices, central_vertex, E, V)
    
    # 2D projection visualization
    ax2 = fig.add_subplot(132)
    
    # Get direction vectors and project to 2D
    vertex_pos = V[central_vertex]
    incident_edges = [ei for ei in edge_indices if central_vertex in E[ei]]
    
    vectors = []
    for ei in incident_edges:
        v1, v2 = E[ei]
        other_idx = v2 if v1 == central_vertex else v1
        vec = V[other_idx] - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, ei))
    
    # Project to XY plane for simple visualization
    ax2.scatter(0, 0, c='red', s=100, label='Central Vertex')
    
    for i, (unit_vec, ei) in enumerate(vectors):
        # Project to XY plane
        x, y = unit_vec[0], unit_vec[1]
        ax2.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                 fc='blue', ec='blue', alpha=0.7)
        ax2.text(x*1.1, y*1.1, f'E{ei}', fontsize=10, ha='center')
    
    # Show circulation order
    for i, ei in enumerate(ordered_edges):
        # Find the vector for this edge
        for unit_vec, edge_id in vectors:
            if edge_id == ei:
                x, y = unit_vec[0], unit_vec[1]
                circle = plt.Circle((x*0.8, y*0.8), 0.08, 
                                  fill=False, color='red', linewidth=2)
                ax2.add_patch(circle)
                ax2.text(x*0.8, y*0.8, str(i+1), 
                        ha='center', va='center', fontweight='bold', color='red')
                break
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('2D Projection: Circulation Order')
    ax2.set_xlabel('X direction')
    ax2.set_ylabel('Y direction')
    
    # Results table
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    # Create results text
    results_text = "Edge Circulation Results:\n\n"
    results_text += f"Central Vertex: V{central_vertex}\n"
    results_text += f"Position: {V[central_vertex]}\n\n"
    results_text += "Counter-clockwise order:\n"
    
    for i, ei in enumerate(ordered_edges):
        v1, v2 = E[ei]
        other_vertex = v2 if v1 == central_vertex else v1
        other_pos = V[other_vertex]
        
        # Calculate angle for reference
        vec = other_pos - vertex_pos
        angle_deg = np.degrees(np.arctan2(vec[1], vec[0]))
        
        results_text += f"{i+1}. Edge E{ei}: V{central_vertex} → V{other_vertex}\n"
        results_text += f"   Position: {other_pos}\n"
        results_text += f"   XY Angle: {angle_deg:.1f}°\n\n"
    
    ax3.text(0.05, 0.95, results_text, transform=ax3.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig, ordered_edges

# Example 1: Simple cross pattern
def example_1():
    print("=== Example 1: Simple Cross Pattern ===")
    
    # Vertices: central vertex at origin, 4 others in cardinal directions
    V = np.array([
        [0, 0, 0],      # V0: center
        [1, 0, 0],      # V1: east
        [0, 1, 0],      # V2: north  
        [-1, 0, 0],     # V3: west
        # [0, -1, 0]      # V4: south
        [0, 0, 1]      # V4: south

    ])
    
    # Edges connecting center to all others
    E = [
        [0, 1],  # E0: center to east
        [0, 2],  # E1: center to north
        [0, 3],  # E2: center to west
        [0, 4]   # E3: center to south
    ]
    
    fig, ordered_edges = visualize_edge_circulation(V, E, central_vertex=0)
    fig.suptitle('Example 1: Cross Pattern', fontsize=14, fontweight='bold')
    
    print(f"Ordered edges: {ordered_edges}")
    print("Expected: counter-clockwise from east: [0, 1, 2, 3] or similar rotation")
    print()
    
    return fig

# Example 2: 3D pyramid structure
def example_2():
    print("=== Example 2: 3D Pyramid Structure ===")
    
    # Vertices: central vertex slightly above origin, others form base
    V = np.array([
        [0, 0, 0.5],    # V0: center (apex)
        [1, 1, 0],      # V1: northeast corner
        [-1, 1, 0],     # V2: northwest corner
        [-1, -1, 0],    # V3: southwest corner
        [1, -1, 0],     # V4: southeast corner
        [0, 0, -1]      # V5: bottom point
    ])
    
    # Edges from apex to all base vertices and bottom
    E = [
        [0, 1],  # E0: apex to northeast
        [0, 2],  # E1: apex to northwest
        [0, 3],  # E2: apex to southwest
        [0, 4],  # E3: apex to southeast
        [0, 5]   # E4: apex to bottom
    ]
    
    fig, ordered_edges = visualize_edge_circulation(V, E, central_vertex=0)
    fig.suptitle('Example 2: 3D Pyramid Structure', fontsize=14, fontweight='bold')
    
    print(f"Ordered edges: {ordered_edges}")
    print("Expected: circulation around the pyramid base, with bottom edge positioned by SVD")
    print()
    
    return fig

# Example 3: Random 3D configuration
def example_3():
    print("=== Example 3: Random 3D Configuration ===")
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Central vertex at origin
    V = [np.array([0, 0, 0])]
    
    # Generate 6 random vertices around the center
    for i in range(4):
        # Random point on unit sphere, then scale randomly
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.8, 2.0)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        V.append(np.array([x, y, z]))
    
    V = np.array(V)
    
    # Connect center to all other vertices
    E = [[0, i+1] for i in range(4)]
    
    fig, ordered_edges = visualize_edge_circulation(V, E, central_vertex=0)
    fig.suptitle('Example 3: Random 3D Configuration', fontsize=14, fontweight='bold')
    
    print(f"Ordered edges: {ordered_edges}")
    print("Random configuration - order determined by SVD projection")
    print()
    
    return fig

# Example 4: Testing the algorithm step by step
def detailed_algorithm_demo():
    print("=== Detailed Algorithm Demonstration ===")
    
    # Simple case for clear demonstration
    V = np.array([
        [0, 0, 0],      # V0: center
        [2, 1, 0.5],    # V1
        [-1, 2, -0.3],  # V2
        [-2, -1, 0.8],  # V3
        [1, -2, -0.5]   # V4
    ])
    
    E = [[0, 1], [0, 2], [0, 3], [0, 4]]
    
    print("Step-by-step algorithm execution:")
    print(f"Vertices: {V}")
    print(f"Edges: {E}")
    print()
    
    # Manual algorithm execution with prints
    vertex_idx = 0
    edge_indices = [0, 1, 2, 3]
    vertex_pos = V[vertex_idx]
    
    print(f"1. Central vertex V{vertex_idx} at position: {vertex_pos}")
    
    # Filter incident edges
    incident_edges = [ei for ei in edge_indices if vertex_idx in E[ei]]
    print(f"2. Incident edges: {incident_edges}")
    
    # Create direction vectors
    print("3. Direction vectors:")
    vectors = []
    vector_array = np.zeros((len(incident_edges), 3))
    
    for i, ei in enumerate(incident_edges):
        v1, v2 = E[ei]
        other_idx = v2 if v1 == vertex_idx else v1
        vec = V[other_idx] - vertex_pos
        unit_vec = vec / np.linalg.norm(vec)
        vectors.append((unit_vec, ei))
        vector_array[i] = unit_vec
        print(f"   E{ei} → V{other_idx}: {vec} → {unit_vec}")
    
    # SVD projection
    print("4. SVD projection to 2D plane:")
    _, _, Vt = np.linalg.svd(vector_array, full_matrices=False)
    basis_1, basis_2 = Vt[0], Vt[1]
    print(f"   Basis 1: {basis_1}")
    print(f"   Basis 2: {basis_2}")
    
    # Project and compute angles
    print("5. Projected angles:")
    projected_vectors = []
    for unit_vec, ei in vectors:
        proj_1 = np.dot(unit_vec, basis_1)
        proj_2 = np.dot(unit_vec, basis_2)
        angle = np.arctan2(proj_2, proj_1)
        projected_vectors.append((angle, ei))
        print(f"   E{ei}: proj=({proj_1:.3f}, {proj_2:.3f}) → angle={np.degrees(angle):.1f}°")
    
    # Sort by angle
    projected_vectors.sort(key=lambda x: x[0])
    ordered_edges = [ei for _, ei in projected_vectors]
    
    print(f"6. Final counter-clockwise order: {ordered_edges}")
    
    # Create visualization
    fig, _ = visualize_edge_circulation(V, E, central_vertex=0)
    fig.suptitle('Detailed Algorithm Demo', fontsize=14, fontweight='bold')
    
    return fig

if __name__ == "__main__":
    # Run all examples
    print("Edge Circulation Algorithm Examples")
    print("=" * 50)
    
    fig1 = example_1()
    plt.show()
    
    fig2 = example_2()
    plt.show()
    
    fig3 = example_3()
    plt.show()
    
    fig4 = detailed_algorithm_demo()
    plt.show()
    
    print("All examples completed!")
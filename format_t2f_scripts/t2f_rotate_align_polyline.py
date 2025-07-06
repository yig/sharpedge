"""
t2f_rotate_align_polyline.py

Rotate a 3D sketch to align its straight polylines with XYZ axes.

Steps:
1. Load sketch from .obj file (vertices + polylines).
2. Detect straight polylines via PCA.
3. Optimize a global rotation aligning them to axes.
4. Apply rotation and save the result (.obj and optionally .png).

Usage:
    python t2f_rotate_align_polyline.py input.obj output.obj [--visualize] [--interactive]
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_sketch_polyline_data(filename):
    """Parse OBJ file to extract vertex coordinates and polyline data."""
    vertices = []
    polylines = []
    
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'l':
                polyline = [int(idx) - 1 for idx in parts[1:]]
                polylines.append(np.array(polyline))
    
    V = np.array(vertices)
    P = polylines
    
    print(f"Read from {filename}:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(polylines)} polylines")
    
    return V, P

def is_polyline_straight(vertices, polyline, straightness_threshold=0.95):
    """
    Check if a polyline is straight using PCA.
    Returns (is_straight, straightness_score, direction_vector)
    """
    if len(polyline) < 3:
        start_point = vertices[polyline[0]]
        end_point = vertices[polyline[-1]]
        direction = end_point - start_point
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            return True, 1.0, direction / norm
        else:
            return False, 0.0, None
    
    points = vertices[polyline]
    points_centered = points - np.mean(points, axis=0)
    
    # Compute covariance matrix and eigenvalues
    cov = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Handle case where all points are the same
    if np.sum(eigenvalues) < 1e-10:
        return False, 0.0, None
    
    # Straightness is the ratio of largest eigenvalue to sum of all eigenvalues
    straightness = eigenvalues[0] / np.sum(eigenvalues)
    
    # Direction is the first principal component
    direction = eigenvectors[:, 0]
    
    # Ensure direction points from start to end roughly
    start_to_end = vertices[polyline[-1]] - vertices[polyline[0]]
    if np.dot(direction, start_to_end) < 0:
        direction = -direction
    
    is_straight = straightness >= straightness_threshold
    
    return is_straight, straightness, direction

def find_straight_polylines(vertices, polylines, straightness_threshold=0.95):
    """Find all straight polylines and their properties."""
    straight_polylines = []
    
    for i, polyline in enumerate(polylines):
        is_straight, straightness, direction = is_polyline_straight(vertices, polyline, straightness_threshold)
        
        if is_straight and direction is not None:
            # Calculate length
            total_length = 0
            for j in range(len(polyline) - 1):
                v1, v2 = vertices[polyline[j]], vertices[polyline[j + 1]]
                total_length += np.linalg.norm(v2 - v1)
            
            straight_polylines.append({
                'index': i,
                'polyline': polyline,
                'direction': direction,
                'straightness': straightness,
                'length': total_length,
                'num_points': len(polyline)
            })
            
            print(f"Straight polyline {i}: {len(polyline)} points, "
                  f"length={total_length:.3f}, straightness={straightness:.3f}")
    
    return straight_polylines

def rotation_matrix_from_euler(rx, ry, rz):
    """Create rotation matrix from Euler angles (in radians)."""
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def alignment_cost(euler_angles, straight_polylines):
    """
    Cost function to minimize: measures how well straight polylines align with X, Y, Z axes.
    Lower cost = better alignment.
    """
    rx, ry, rz = euler_angles
    R = rotation_matrix_from_euler(rx, ry, rz)
    
    # Target directions (X, Y, Z axes)
    targets = np.array([
        [1, 0, 0],  # X axis
        [0, 1, 0],  # Y axis
        [0, 0, 1]   # Z axis
    ])
    
    total_cost = 0
    
    for poly_info in straight_polylines:
        # Rotate the polyline direction
        rotated_direction = R @ poly_info['direction']
        
        # Find best alignment with any of the three axes
        alignment_scores = []
        for target in targets:
            # Consider both positive and negative directions
            score1 = abs(np.dot(rotated_direction, target))
            score2 = abs(np.dot(rotated_direction, -target))
            alignment_scores.append(max(score1, score2))
        
        # Best alignment score (closer to 1 is better)
        best_alignment = max(alignment_scores)
        
        # Convert to cost (closer to 0 is better) and weight by length
        cost = (1 - best_alignment) * poly_info['length']
        total_cost += cost
    
    return total_cost

def find_optimal_alignment(straight_polylines):
    """Find optimal rotation to align straight polylines with axes."""
    if not straight_polylines:
        return np.eye(3), float('inf')
    
    print(f"\nOptimizing alignment for {len(straight_polylines)} straight polylines...")
    
    # Initial guess: no rotation
    initial_guess = [0, 0, 0]
    
    # Optimize
    result = minimize(
        alignment_cost,
        initial_guess,
        args=(straight_polylines,),
        method='BFGS',
        options={'disp': True, 'maxiter': 1000}
    )
    
    if result.success:
        optimal_angles = result.x
        optimal_rotation = rotation_matrix_from_euler(*optimal_angles)
        final_cost = result.fun
        
        print(f"Optimization successful!")
        print(f"Optimal Euler angles (deg): {np.degrees(optimal_angles)}")
        print(f"Final cost: {final_cost:.6f}")
        
        return optimal_rotation, final_cost
    else:
        print(f"Optimization failed: {result.message}")
        return np.eye(3), float('inf')

def analyze_alignment_quality(vertices, straight_polylines, rotation_matrix):
    """Analyze how well the polylines align with axes after rotation."""
    targets = {
        'X+': np.array([1, 0, 0]),
        'X-': np.array([-1, 0, 0]),
        'Y+': np.array([0, 1, 0]),
        'Y-': np.array([0, -1, 0]),
        'Z+': np.array([0, 0, 1]),
        'Z-': np.array([0, 0, -1])
    }
    
    print("\nAlignment analysis:")
    print("-" * 60)
    
    for poly_info in straight_polylines:
        rotated_direction = rotation_matrix @ poly_info['direction']
        
        # Find best alignment
        best_axis = None
        best_score = 0
        
        for axis_name, target in targets.items():
            score = abs(np.dot(rotated_direction, target))
            if score > best_score:
                best_score = score
                best_axis = axis_name
        
        print(f"Polyline {poly_info['index']:2d}: "
              f"{poly_info['num_points']:3d} pts, "
              f"len={poly_info['length']:6.3f}, "
              f"→ {best_axis} axis (score: {best_score:.3f})")

def write_obj_file(vertices, polylines, output_filename):
    """Write vertices and polylines to OBJ file."""
    with open(output_filename, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for polyline in polylines:
            indices_1based = [str(i + 1) for i in polyline]
            f.write(f"l {' '.join(indices_1based)}\n")

def visualize_sketch(vertices, polylines, straight_polylines, output_path=None, title="Sketch"):
    """Create 3D visualization highlighting straight polylines."""
    try:
        if output_path:
            plt.switch_backend('Agg')
        
        fig = plt.figure(figsize=(8,8))
        
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(vertical_axis='y', elev=30, azim=45)
        ax.set_aspect('equal')
        
        # Get indices of straight polylines
        straight_indices = {poly['index'] for poly in straight_polylines}
        
        # Plot all polylines
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'pink']
        straight_count = 0
        
        for i, poly in enumerate(polylines):
            if len(poly) < 2:
                continue
            
            pts = vertices[poly]
            
            if i in straight_indices:
                # Highlight straight polylines
                color = colors[straight_count % len(colors)]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                       linewidth=3, color=color, alpha=0.9,
                       label=f'Straight {i} ({len(poly)} pts)')
                straight_count += 1
            else:
                # Regular polylines
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                       linewidth=1, color='gray', alpha=0.5)
        
        # Draw coordinate axes
        max_range = np.max(np.ptp(vertices, axis=0)) * 0.3
        origin = np.mean(vertices, axis=0)
        
        ax.quiver(origin[0], origin[1], origin[2], max_range, 0, 0, 
                 color='red', arrow_length_ratio=0.1, linewidth=2, label='X axis')
        ax.quiver(origin[0], origin[1], origin[2], 0, max_range, 0, 
                 color='green', arrow_length_ratio=0.1, linewidth=2, label='Y axis')
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, max_range, 
                 color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z axis')
        
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title(title)

            
        plt.axis('off')
        plt.axis('equal')
        
        # Set equal aspect ratio
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                             vertices[:,1].max()-vertices[:,1].min(),
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Visualization saved: {output_path}")
        else:
            plt.show()
            
        return True
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Align straight polylines with X, Y, Z axes optimally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_file", help="Input .obj file")
    parser.add_argument("output_file", help="Output .obj file")
    parser.add_argument("--straightness-threshold", type=float, default=0.95,
                       help="Minimum straightness score (0-1) to consider a polyline straight")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate before/after visualizations")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive plots")
    
    args = parser.parse_args()
    
    # Load data
    V, P = load_sketch_polyline_data(args.input_file)
    
    # Find straight polylines
    print(f"\nFinding straight polylines (threshold: {args.straightness_threshold})...")
    straight_polylines = find_straight_polylines(V, P, args.straightness_threshold)
    
    if not straight_polylines:
        print("No straight polylines found! Try lowering the --straightness-threshold")
        return
    
    print(f"\nFound {len(straight_polylines)} straight polylines")
    
    # Show before visualization
    if args.visualize or args.interactive:
        if args.interactive:
            visualize_sketch(V, P, straight_polylines, title="Before Alignment")
        if args.visualize:
            before_path = args.output_file.replace('.obj', '_before.png')
            visualize_sketch(V, P, straight_polylines, before_path, "Before Alignment")
    
    # Find optimal alignment
    optimal_rotation, cost = find_optimal_alignment(straight_polylines)
    
    # Apply rotation
    V_aligned = (optimal_rotation @ V.T).T
    
    # Update straight polylines with new directions
    aligned_straight_polylines = []
    for poly_info in straight_polylines:
        new_direction = optimal_rotation @ poly_info['direction']
        aligned_poly_info = poly_info.copy()
        aligned_poly_info['direction'] = new_direction
        aligned_straight_polylines.append(aligned_poly_info)
    
    # Analyze alignment quality
    analyze_alignment_quality(V, straight_polylines, optimal_rotation)
    
    # Save result
    write_obj_file(V_aligned, P, args.output_file if args.output_file.endswith('.obj') else args.output_file + '.obj')
    print(f"\nAligned model saved: {args.output_file}")
    
    # Show after visualization
    if args.visualize or args.interactive:
        if args.interactive:
            visualize_sketch(V_aligned, P, aligned_straight_polylines, title="After Alignment")
        if args.visualize:
            after_path = args.output_file.replace('.obj', '_after.png')
            visualize_sketch(V_aligned, P, aligned_straight_polylines, after_path, "After Alignment")

if __name__ == "__main__":
    main()
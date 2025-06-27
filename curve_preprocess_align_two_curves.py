'''
curve_preprocess_align_two_curves.py
Load two curve files and align them.
This can help to make the curve in the bounding box.
'''

import argparse
import numpy as np
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_sketch_data

def center_vertices(V):
    """Center vertices around their centroid"""
    centroid = np.mean(V, axis=0)
    return V - centroid, centroid

def scale_to_match_size(V_source, V_target):
    """
    Scale V_source to match the overall size of V_target
    
    Args:
        V_source: vertices to be scaled
        V_target: reference vertices for size
    
    Returns:
        scaled vertices, scale factor
    """
    # Calculate bounding box sizes
    source_size = np.max(V_source, axis=0) - np.min(V_source, axis=0)
    target_size = np.max(V_target, axis=0) - np.min(V_target, axis=0)
    
    # Use the maximum dimension for uniform scaling
    source_max_dim = np.max(source_size)
    target_max_dim = np.max(target_size)
    
    if source_max_dim == 0:
        return V_source, 1.0
    
    scale_factor = target_max_dim / source_max_dim
    return V_source * scale_factor, scale_factor

def compute_rotation_matrix_procrustes(V_source, V_target):
    """
    Compute optimal rotation matrix using Procrustes analysis
    Assumes both point sets are already centered and scaled
    """
    # Compute cross-covariance matrix
    H = np.dot(V_source.T, V_target)
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R

def try_different_orientations_procrustes(V_source, V_target):
    """
    Try different orientations with Procrustes to avoid upside-down results
    """
    # Get the basic Procrustes rotation
    R_base = compute_rotation_matrix_procrustes(V_source, V_target)
    
    best_R = R_base
    best_error = float('inf')
    
    # Test the original rotation
    V_test = np.dot(V_source, R_base.T)
    if len(V_target) == len(V_test):
        error = np.mean(np.linalg.norm(V_test - V_target, axis=1))
    else:
        distances = np.linalg.norm(V_test[:, np.newaxis] - V_target[np.newaxis, :], axis=2)
        error = np.mean(np.min(distances, axis=1))
    
    if error < best_error:
        best_error = error
        best_R = R_base
    
    # Try flipping different axes
    flip_matrices = [
        np.diag([1, 1, -1]),   # flip Z
        np.diag([1, -1, 1]),   # flip Y  
        np.diag([-1, 1, 1]),   # flip X
        np.diag([1, -1, -1]),  # flip Y and Z
        np.diag([-1, 1, -1]),  # flip X and Z
        np.diag([-1, -1, 1]),  # flip X and Y
        np.diag([-1, -1, -1])  # flip all axes
    ]
    
    for i, flip_matrix in enumerate(flip_matrices):
        R_flipped = np.dot(R_base, flip_matrix)
        V_test = np.dot(V_source, R_flipped.T)
        
        if len(V_target) == len(V_test):
            error = np.mean(np.linalg.norm(V_test - V_target, axis=1))
        else:
            distances = np.linalg.norm(V_test[:, np.newaxis] - V_target[np.newaxis, :], axis=2)
            error = np.mean(np.min(distances, axis=1))
        
        if error < best_error:
            best_error = error
            best_R = R_flipped
            print(f"Better orientation found with flip pattern {i+1}, error: {best_error:.6f}")
    
    print(f"Final Procrustes orientation error: {best_error:.6f}")
    return best_R

def align_principal_axes(V_source, V_target):
    """
    Align using Principal Component Analysis (PCA) with orientation correction
    """
    def compute_pca(V):
        # Center the data
        V_centered = V - np.mean(V, axis=0)
        # Compute covariance matrix
        cov_matrix = np.cov(V_centered.T)
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        return eigenvecs[:, idx], eigenvals[idx]
    
    # Get principal axes for both point clouds
    axes_source, vals_source = compute_pca(V_source)
    axes_target, vals_target = compute_pca(V_target)
    
    # Try all 8 possible orientations (each axis can be flipped)
    best_R = None
    best_error = float('inf')
    
    for flip_x in [1, -1]:
        for flip_y in [1, -1]:
            for flip_z in [1, -1]:
                # Create flipped target axes
                axes_target_flipped = axes_target.copy()
                axes_target_flipped[:, 0] *= flip_x
                axes_target_flipped[:, 1] *= flip_y
                axes_target_flipped[:, 2] *= flip_z
                
                # Ensure right-handed coordinate system
                if np.linalg.det(axes_target_flipped) < 0:
                    axes_target_flipped[:, -1] *= -1
                
                # Compute rotation matrix
                R = np.dot(axes_target_flipped, axes_source.T)
                
                # Test this rotation
                V_rotated = np.dot(V_source, R.T)
                
                # Compute alignment error
                if len(V_target) == len(V_rotated):
                    error = np.mean(np.linalg.norm(V_rotated - V_target, axis=1))
                else:
                    distances = np.linalg.norm(V_rotated[:, np.newaxis] - V_target[np.newaxis, :], axis=2)
                    error = np.mean(np.min(distances, axis=1))
                
                if error < best_error:
                    best_error = error
                    best_R = R
    
    print(f"Best PCA orientation found with error: {best_error:.6f}")
    return best_R

def iterative_closest_point_simple(V_source, V_target, max_iterations=50, tolerance=1e-6):
    """
    Simple ICP algorithm for fine alignment
    """
    V_aligned = V_source.copy()
    
    for iteration in range(max_iterations):
        # Find closest points
        distances = np.linalg.norm(V_aligned[:, np.newaxis] - V_target[np.newaxis, :], axis=2)
        closest_indices = np.argmin(distances, axis=1)
        closest_points = V_target[closest_indices]
        
        # Center both point sets
        centroid_source = np.mean(V_aligned, axis=0)
        centroid_target = np.mean(closest_points, axis=0)
        
        V_centered_source = V_aligned - centroid_source
        V_centered_target = closest_points - centroid_target
        
        # Compute rotation
        H = np.dot(V_centered_source.T, V_centered_target)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Apply transformation
        V_new = np.dot(V_centered_source, R.T) + centroid_target
        
        # Check convergence
        change = np.mean(np.linalg.norm(V_new - V_aligned, axis=1))
        V_aligned = V_new
        
        if change < tolerance:
            print(f"ICP converged after {iteration + 1} iterations")
            break
    
    return V_aligned, R, centroid_target - np.dot(centroid_source, R.T)

def align_point_clouds(V_source, V_target, method='pca', use_icp=False):
    """
    Align V_source to V_target using various methods
    
    Args:
        V_source: source point cloud to be aligned
        V_target: target point cloud to align to
        method: 'pca', 'procrustes', or 'center_only'
        use_icp: whether to use ICP for fine alignment
    
    Returns:
        aligned vertices, transformation info
    """
    print(f"Aligning using method: {method}")
    
    # Step 1: Center both point clouds
    V_source_centered, centroid_source = center_vertices(V_source)
    V_target_centered, centroid_target = center_vertices(V_target)
    
    # Step 2: Scale source to match target size
    V_source_scaled, scale_factor = scale_to_match_size(V_source_centered, V_target_centered)
    
    print(f"Scale factor: {scale_factor:.4f}")
    
    # Step 3: Find optimal rotation with orientation correction
    if method == 'pca':
        R = align_principal_axes(V_source_scaled, V_target_centered)
    elif method == 'procrustes':
        R = try_different_orientations_procrustes(V_source_scaled, V_target_centered)
    elif method == 'center_only':
        R = np.eye(3)  # No rotation
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 4: Apply rotation
    V_rotated = np.dot(V_source_scaled, R.T)
    
    # Step 5: Translate to target centroid
    V_aligned = V_rotated + centroid_target
    
    # Step 6: Optional ICP refinement
    if use_icp and method != 'center_only':
        print("Applying ICP refinement...")
        V_aligned, R_icp, t_icp = iterative_closest_point_simple(V_aligned, V_target)
        R = np.dot(R_icp, R)  # Combine rotations
    
    # Compute final alignment error
    if len(V_target) == len(V_aligned):
        alignment_error = np.mean(np.linalg.norm(V_aligned - V_target, axis=1))
    else:
        # Find closest point distances for different sized point clouds
        distances = np.linalg.norm(V_aligned[:, np.newaxis] - V_target[np.newaxis, :], axis=2)
        closest_distances = np.min(distances, axis=1)
        alignment_error = np.mean(closest_distances)
    
    transform_info = {
        'centroid_source': centroid_source,
        'centroid_target': centroid_target,
        'scale_factor': scale_factor,
        'rotation_matrix': R,
        'alignment_error': alignment_error,
        'method': method
    }
    
    return V_aligned, transform_info

def save_sketch_polyline_data(filename, V, E, P):
    """
    Save sketch polyline data to file in simple v and ls format.
    
    Args:
        filename (str): Output file path
        V (ndarray): nx3 array of vertex coordinates
        E (list/array): Edge data (not used in this format)
        P (list): List of polyline indices
    """
    with open(filename, 'w') as f:
        # Write vertices
        for i, vertex in enumerate(V):
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        # Write polylines as line segments (ls)
        for i, polyline in enumerate(P):
            polyline_str = " ".join(map(str, polyline+1))
            f.write(f"l {polyline_str}\n")
            
def print_alignment_info(info):
    """Print alignment transformation details"""
    print("="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    print(f"Method: {info['method']}")
    print(f"Scale factor: {info['scale_factor']:.6f}")
    print(f"Source centroid: [{info['centroid_source'][0]:.4f}, {info['centroid_source'][1]:.4f}, {info['centroid_source'][2]:.4f}]")
    print(f"Target centroid: [{info['centroid_target'][0]:.4f}, {info['centroid_target'][1]:.4f}, {info['centroid_target'][2]:.4f}]")
    print(f"Final alignment error: {info['alignment_error']:.6f}")
    print(f"Rotation matrix:")
    for i, row in enumerate(info['rotation_matrix']):
        print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")

def visualize_alignment(V1, V2_original, V2_aligned, title="Alignment Result"):
    """Visualize original and aligned point clouds"""
    print(f"\n{title}:")
    print("V1 (target) - Blue")
    print("V2 original - Red") 
    print("V2 aligned - Green")
    
    # You can create a combined visualization here if needed
    # For now, we'll plot them separately
    plot_sketch_data(V1, [[i] for i in range(len(V1))])  # V1 as individual points
    plot_sketch_data(V2_original, [[i] for i in range(len(V2_original))])  # V2 original
    plot_sketch_data(V2_aligned, [[i] for i in range(len(V2_aligned))])  # V2 aligned


# Main execution
parser = argparse.ArgumentParser(description='Align V2 to V1 with scale and rotation')
parser.add_argument('curve_1', nargs='?', help='Target curve sketch (V1)')
parser.add_argument('curve_2', nargs='?', help='Source curve sketch (V2) to be aligned')
parser.add_argument('--method', default='pca', choices=['pca', 'procrustes', 'center_only'],
                   help='Alignment method')
parser.add_argument('--use-icp', action='store_true', help='Use ICP for fine alignment')
parser.add_argument('--output', help='Output file for aligned V2')

args = parser.parse_args()

curve_1 = args.curve_1
curve_2 = args.curve_2

# Load both datasets
print(f"Loading target (V1): {curve_1}")
V1, E1, P1 = load_sketch_polyline_data(curve_1)
print(f"V1: {len(V1)} vertices")

print(f"Loading source (V2): {curve_2}")
V2, E2, P2 = load_sketch_polyline_data(curve_2)
print(f"V2: {len(V2)} vertices")

# Perform alignment
V2_aligned, transform_info = align_point_clouds(V2, V1, method=args.method, use_icp=args.use_icp)

# Print results
print_alignment_info(transform_info)

# Visualize results
print("\nTarget point cloud (V1):")
plot_sketch_data(V1, P1)

print("Source point cloud (V2) - Original:")
plot_sketch_data(V2, P2)

print("Source point cloud (V2) - Aligned to V1:")
plot_sketch_data(V2_aligned, P2)

# Save aligned result if requested
if args.output:
    print(f"\nSaving aligned V2 to: {args.output}")
    save_sketch_polyline_data(args.output, V2_aligned, E2, P2)

print(f"\nAlignment complete! Final error: {transform_info['alignment_error']:.6f}")

# Optional: Compute and display alignment statistics
print("\nAlignment Statistics:")
print(f"Original V1 bounds: {np.min(V1, axis=0)} to {np.max(V1, axis=0)}")
print(f"Original V2 bounds: {np.min(V2, axis=0)} to {np.max(V2, axis=0)}")
print(f"Aligned V2 bounds: {np.min(V2_aligned, axis=0)} to {np.max(V2_aligned, axis=0)}")
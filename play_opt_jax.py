import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import time
import numpy as np
import scipy.optimize as opt

from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_edge_info, plot_edge_constraints, plot_edge_constraints_two_normals
from opt_edges import create_frames_for_each_polyline, edge_distance_matrix, extract_pairwise_weight, build_vertex_to_edges_map, estimate_initial_thetas
from collections import deque
from utility_rotate_vector import rotation_matrix_from
from play_opt_parameters import propagate_normals_from_constraints, edge_to_edge_normal_transport



import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import time
import numpy as np
import scipy.optimize as opt

def create_edge_rotation_map_jax_improved(V, E):
    '''
    Create an optimized JAX-friendly edge rotation map.
    
    This version pre-computes rotation matrices for all edge pairs and returns
    a structure optimized for the pairwise energy computation.
    
    Parameters:
    - V: (n, 3) array of vertex coordinates
    - E: (m, 2) array of edge vertex index pairs
    
    Returns:
    - Dictionary with rotation data in a format optimized for the pairwise energy computation
    '''
    n_edges = len(E)
    
    # Pre-allocate arrays for all possible edge pairs
    rotation_matrices = np.zeros((n_edges, n_edges, 3, 3))
    has_rotation = np.zeros((n_edges, n_edges), dtype=bool)
    
    # Compute rotations for all pairs
    for i in range(n_edges):
        for j in range(n_edges):
            if i == j:
                continue
                
            ei = E[i]
            ej = E[j]
            
            # Check if they share an endpoint
            shared_indices = set(ei) & set(ej)
            if len(shared_indices) != 0:
                assert len(shared_indices) == 1
                
                # Get the shared index
                shared_index = next(iter(shared_indices))
                
                # Get the non-shared index from ei
                ei_other_index = next(iter(set(ei) - shared_indices))
                
                # Get the non-shared index from ej
                ej_other_index = next(iter(set(ej) - shared_indices))
                
                # Compute vectors
                vector_ei = V[shared_index] - V[ei_other_index]
                vector_ej = V[ej_other_index] - V[shared_index]
                
                # Compute rotation matrix
                rotation_matrices[i, j] = rotation_matrix_from(vector_ei, vector_ej)
                has_rotation[i, j] = True
    
    return {
        'matrices': rotation_matrices,
        'has_rotation': has_rotation
    }

def rotation_matrix_from(v1, v2):
    """
    Compute rotation matrix that rotates v1 to v2.
    
    Parameters:
    - v1: First vector
    - v2: Second vector
    
    Returns:
    - 3x3 rotation matrix
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Compute axis of rotation (cross product)
    axis = np.cross(v1_norm, v2_norm)
    axis_norm = np.linalg.norm(axis)
    
    # If vectors are parallel, return identity
    if axis_norm < 1e-10:
        return np.eye(3)
    
    # Normalize axis
    axis = axis / axis_norm
    
    # Compute angle between vectors
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Compute rotation matrix using Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R

def preprocess_data_for_jax_with_improved_rotations(E, pairwise, rotations_data, vertex_to_edges_map):
    """
    Pre-process data structures for JAX optimization with improved rotation handling
    
    Parameters:
    - E: List of edge vertex indices
    - pairwise: List of tuples (edge1, edge2, weight)
    - rotations_data: Dictionary with rotation data from create_edge_rotation_map_jax_improved
    - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices
    
    Returns:
    - Dictionary with pre-processed data for JAX optimization
    """
    # Convert E to numpy array if it isn't already
    E_array = np.array(E)
    
    # Extract pairwise data
    e1_indices = np.array([p[0] for p in pairwise])
    e2_indices = np.array([p[1] for p in pairwise])
    weights = np.array([p[2] for p in pairwise])
    
    # Pre-compute information about shared vertices and curve edges
    curve_edges = np.zeros(len(pairwise), dtype=bool)
    
    for i, (e1, e2, _) in enumerate(pairwise):
        shared_vertex = frozenset(E[e1]) & frozenset(E[e2])
        if len(shared_vertex) == 1:
            vertex = next(iter(shared_vertex))
            if len(vertex_to_edges_map[vertex]) == 2:
                curve_edges[i] = True
    
    # Extract rotation data for the specific edge pairs in pairwise
    has_rotation = np.array([rotations_data['has_rotation'][e1, e2] for e1, e2 in zip(e1_indices, e2_indices)])
    rotation_matrices = np.array([rotations_data['matrices'][e1, e2] for e1, e2 in zip(e1_indices, e2_indices)])
    
    return {
        'E': jnp.array(E_array),
        'e1_indices': jnp.array(e1_indices),
        'e2_indices': jnp.array(e2_indices),
        'weights': jnp.array(weights),
        'curve_edges': jnp.array(curve_edges),
        'rotation_matrices': jnp.array(rotation_matrices),
        'has_rotation': jnp.array(has_rotation),
        'num_pairs': len(e1_indices)
    }

def optimize_with_jax_two_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map, one_normal=None, callback_fn = None):
    """
    JAX-based implementation of the two-normal per edge optimization
    
    Parameters:
    - thetas0: Initial theta values for each edge
    - Us, Vs: Frame vectors for each edge
    - edge_constraints: List of tuples (edge_index, desired_normal_vector)
    - pairwise: List of tuples (edge1, edge2, weight)
    - rotations_data: Dictionary with rotation data from create_edge_rotation_map_jax_improved
    - E: List of edge vertex indices
    - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices
    - one_normal: List of edge indices that should have only one normal
    
    Returns:
    - Optimization result
    """
    start_time = time.time()
    
    # Convert inputs to JAX arrays
    Us_jax = jnp.array(Us)
    Vs_jax = jnp.array(Vs)
    
    # Convert edge_constraints to JAX-friendly format
    constraint_indices = []
    constraint_normals = []
    for edge_idx, normal in edge_constraints:
        constraint_indices.append(edge_idx)
        constraint_normals.append(normal)
    
    constraints_jax = {
        'indices': jnp.array(constraint_indices),
        'normals': jnp.array(constraint_normals)
    }
    
    # Pre-process data for faster computation using improved rotations
    data = preprocess_data_for_jax_with_improved_rotations(E, pairwise, rotations_data, vertex_to_edges_map)
    
    # Create initial 2D array of thetas
    num_edges = len(thetas0)
    thetas_2d = np.column_stack((thetas0, thetas0))
    thetas_2d[:, 1] += 1e-3  # Small perturbation for second normal
    thetas0_flat = thetas_2d.flatten()
    
    if one_normal is None:
        one_normal = []
    one_normal_jax = jnp.array(one_normal)
    
    # Create jitted energy function
    energy_jax_jit = jit(energy_two_normal_jax)
    
    def energy_wrapper(thetas_flat):
        thetas_2d = jnp.array(thetas_flat).reshape(num_edges, 2)
        return energy_jax_jit(thetas_2d, Us_jax, Vs_jax, constraints_jax, data, one_normal_jax).item()
    
    # Create jitted gradient function
    grad_fn = grad(energy_two_normal_jax, argnums=0)
    grad_jit = jit(grad_fn)
    
    def grad_wrapper(thetas_flat):
        thetas_2d = jnp.array(thetas_flat).reshape(num_edges, 2)
        grad_val = grad_jit(thetas_2d, Us_jax, Vs_jax, constraints_jax, data, one_normal_jax)
        return np.array(grad_val.reshape(-1))
    
    # Run optimization using scipy (L-BFGS-B works well with JAX gradients)
    result = opt.minimize(
        energy_wrapper,
        thetas0_flat,
        jac=grad_wrapper,
        method='L-BFGS-B',
        tol=0.0000001,
        options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1e5}, 
        callback= callback_fn
    )
    
    end_time = time.time()
    print(f"JAX optimization took {end_time - start_time:.6f} seconds")
    
    return result

@jit
def energy_two_normal_jax(thetas, Us, Vs, constraints, data, one_normal):
    """
    JAX-optimized energy function for two normals per edge
    """
    # Pre-calculate all normals for both orientations
    normals0 = jnp.cos(thetas[:, 0, jnp.newaxis]) * Us + jnp.sin(thetas[:, 0, jnp.newaxis]) * Vs
    normals1 = jnp.cos(thetas[:, 1, jnp.newaxis]) * Us + jnp.sin(thetas[:, 1, jnp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = compute_constraint_energy_jax(normals0, normals1, constraints)
    
    # ===== One Normal Energy =====
    E_one_normal = compute_one_normal_energy_jax(thetas, one_normal)
    
    # ===== Pairwise Energy =====
    E_pairwise = compute_pairwise_energy_jax(
        normals0, 
        normals1, 
        data
    )
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal

@jit
def compute_constraint_energy_jax(normals0, normals1, constraints):
    """Compute constraint energy using JAX"""
    if len(constraints['indices']) == 0:
        return 0.0
    
    # Extract constraint data
    edge_indices = constraints['indices']
    desired_normals = constraints['normals']
    
    # Get constrained normals
    n0 = normals0[edge_indices]
    n1 = normals1[edge_indices]
    
    # Compute dot products (batch operation)
    dots0 = jnp.sum(n0 * desired_normals, axis=1)
    dots1 = jnp.sum(n1 * desired_normals, axis=1)
    
    # Calculate constraint energy
    energy = jnp.mean((1.0 - dots0)**2 + (1.0 - dots1)**2) / 2
    
    return energy

@jit
def compute_one_normal_energy_jax(thetas, one_normal):
    """Compute one-normal energy using JAX"""
    if one_normal.size == 0:
        return 0.0
    
    theta_diffs = thetas[one_normal, 0] - thetas[one_normal, 1]
    return jnp.mean(theta_diffs**2)

@jit
def compute_pairwise_energy_jax(normals0, normals1, data):
    """
    Compute pairwise energy using JAX in a way that's compatible with tracing
    """
    # Initialize accumulators
    energy_sum = 0.0
    weight_sum = 0.0
    
    # Get data for easier access
    e1_indices = data['e1_indices']
    e2_indices = data['e2_indices']
    weights = data['weights']
    curve_edges = data['curve_edges']
    rotation_matrices = data['rotation_matrices']
    has_rotation = data['has_rotation']
    
    # Process all pairs with a manual loop instead of scan
    # This avoids the need for arange with a traced length
    def body_fun(i, accum):
        energy_sum, weight_sum = accum
        
        # Get indices and data for this pair
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        is_curve = curve_edges[i]
        has_rot = has_rotation[i]
        rot_matrix = rotation_matrices[i]
        
        # Get normals for both edges
        n1_0 = normals0[e1]
        n1_1 = normals1[e1]
        n2_0 = normals0[e2]
        n2_1 = normals1[e2]
        
        # Apply rotation if needed (using JAX's where for conditional)
        n1_0_rot = jnp.where(has_rot, jnp.dot(rot_matrix, n1_0), n1_0)
        n1_1_rot = jnp.where(has_rot, jnp.dot(rot_matrix, n1_1), n1_1)
        
        # Compute costs for all combinations
        cost_00 = (1.0 - jnp.dot(n1_0_rot, n2_0))**2
        cost_01 = (1.0 - jnp.dot(n1_0_rot, n2_1))**2
        cost_10 = (1.0 - jnp.dot(n1_1_rot, n2_0))**2
        cost_11 = (1.0 - jnp.dot(n1_1_rot, n2_1))**2
        
        costs = jnp.array([[cost_00, cost_01], [cost_10, cost_11]])
        
        # Calculate pair energy based on edge type
        diagonal_sum = cost_00 + cost_11
        antidiagonal_sum = cost_01 + cost_10
        
        # Curve edges: use min of diagonal/antidiagonal sum
        # Non-curve edges: use global minimum
        pair_energy = jnp.where(
            is_curve,
            weight * jnp.minimum(diagonal_sum, antidiagonal_sum),
            weight * jnp.min(costs)
        )
        
        return energy_sum + pair_energy, weight_sum + weight
    
    # We need to use a fori_loop with static bounds
    # Get a static upper bound for the number of pairs
    # This will be the compile-time maximum number of pairs
    max_pairs = e1_indices.shape[0]
    
    # Use fori_loop for iteration
    energy_sum, weight_sum = jax.lax.fori_loop(
        0, max_pairs, 
        lambda i, accum: body_fun(i, accum), 
        (energy_sum, weight_sum)
    )
    
    # Normalize
    return jnp.where(weight_sum > 0, energy_sum / weight_sum, 0.0)

def recover_normals_from_jax_result(result, Us, Vs, E):
    """
    Recover normal vectors from JAX optimization result
    """
    num_edges = len(E)
    thetas_2d = result.x.reshape(num_edges, 2)
    
    # Calculate normals for both orientations
    normals = {}
    for edge_idx in range(num_edges):
        for which_edge in (0, 1):
            # Calculate the normal for this edge and orientation
            theta = thetas_2d[edge_idx, which_edge]
            normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
            normal = normal / np.linalg.norm(normal)  # Ensure normalized
            
            # Store in the normals dictionary with tuple key (edge_idx, which_edge)
            normals[(edge_idx, which_edge)] = normal
    
    return normals

def make_cylinder_example_jax():
    '''
    Example of using JAX-based optimization on a cylinder example
    '''
    V, E, P = load_sketch_polyline_data('made_examples/sketch/cylinder.obj')

    # Create constraints by hand 
    edge_constraints = [(0, np.array([0, 1, 0])),
                        (1, np.array([0, -1, 0]))]

    # Compute local coordinate frames for edges
    Us, Vs = create_frames_for_each_polyline(V, E, P)

    # Compute edge distances
    distances = edge_distance_matrix(V, E)
    pairwise = extract_pairwise_weight(E, distances)
    
    # Create JAX-friendly edge rotation map
    rotations_data = create_edge_rotation_map_jax_improved(V, E)
    
    # Create vertex to edges map
    vertex_to_edges_map = build_vertex_to_edges_map(E)

    # Propagate normals as initial guess
    estimate_normals = propagate_normals_from_constraints(V, E, vertex_to_edges_map, edge_constraints, edge_to_edge_normal_transport)
    
    # Get initial thetas
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)

    callback_fn = create_callback_two_normals(Us, Vs, E, P, V)
    
    # Specify which edges should have one normal
    one_normal = [23, 58]

    # Run JAX-based optimization
    result = optimize_with_jax_two_normal(
        thetas0,
        Us,
        Vs,
        edge_constraints,
        pairwise,
        rotations_data,
        E,
        vertex_to_edges_map,
        one_normal,
        callback_fn
    )
    
    # Recover normals from result
    normals = recover_normals_from_jax_result(result, Us, Vs, E)
    
    # Visualize results
    plot_edge_constraints_two_normals(V, E, P, normals, unconstrained_polylines_indices = None, 
                                scale=0.08, 
                                str="Two-Normal Optimization Result", 
                                block=True)
    return result, normals



def create_callback_two_normals(Us, Vs, E, P, V):
    """
    Create a faster callback function for visualization during optimization
    """
    iteration_counter = [0]
    # Set a visualization frequency (e.g., every 10 iterations)
    viz_frequency = 10
    
    def callback(thetas_flat_now):
        iteration_counter[0] += 1
        current_iter = iteration_counter[0]
        print(f"Iteration {current_iter}")
        
        # Only visualize every viz_frequency iterations
        # and for the first few iterations to see initial progress
        if current_iter % viz_frequency != 0 and current_iter > 5:
            return False
        
        # Reshape the flat thetas array to the 2D structure
        num_edges = len(E)
        thetas_2d_now = thetas_flat_now.reshape(num_edges, 2)
        
        # Use vectorized operations for normal calculation
        # Create mesh grid of edge indices and which_edge (0 or 1)
        edge_indices = np.arange(num_edges)
        which_edges = np.array([0, 1])
        
        # Use numpy's vectorized operations to compute all normals at once
        thetas = thetas_2d_now.reshape(-1)  # Flatten to 1D array of all thetas
        
        # Pre-allocate the normals dictionary
        normals_now = {}
        
        # Vectorized computation for cos(theta) and sin(theta)
        cos_theta = np.cos(thetas)
        sin_theta = np.sin(thetas)
        
        # Compute all normals in a single loop (still faster than nested loops)
        for i, edge_idx in enumerate(edge_indices):
            for j, which_edge in enumerate(which_edges):
                flat_idx = edge_idx * 2 + which_edge
                normal = cos_theta[flat_idx] * Us[edge_idx] + sin_theta[flat_idx] * Vs[edge_idx]
                # Quick normalization (avoid repeated calls to np.linalg.norm)
                norm = np.sqrt(np.sum(normal * normal))
                normal = normal / norm
                normals_now[(edge_idx, which_edge)] = normal
        
        # Update the plot
        plot_edge_constraints_two_normals(
            V, E, P, normals_now, 
            unconstrained_polylines_indices=None,
            scale=0.08,
            str=f"Two-Normal Optimization: Iteration {current_iter}",
            block=False
        )
        
        return False  # Continue optimization
    
    return callback


if __name__ == "__main__":
    result, normals = make_cylinder_example_jax()


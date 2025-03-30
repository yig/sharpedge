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

def prepare_constraints_jax(edge_constraints):
    """
    Convert edge constraints to JAX-friendly format
    
    Parameters:
    - edge_constraints: List of tuples (edge_index, desired_normal_vector)
    
    Returns:
    - Dictionary with constraint indices and normals as JAX arrays
    """
    constraint_indices = []
    constraint_normals = []
    
    for edge_idx, normal in edge_constraints:
        constraint_indices.append(edge_idx)
        constraint_normals.append(normal)
    
    return {
        'indices': jnp.array(constraint_indices),
        'normals': jnp.array(constraint_normals)
    }

# Common energy computation functions
@jit
def compute_constraint_energy_jax(normals0, normals1, constraints):
    """Compute constraint energy using JAX for two-normal case"""
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
def compute_one_normal_constraint_energy_jax(normals, constraints):
    """Compute constraint energy for one-normal case using JAX"""
    if len(constraints['indices']) == 0:
        return 0.0
    
    # Extract constraint data
    edge_indices = constraints['indices']
    desired_normals = constraints['normals']
    
    # Get constrained normals
    n = normals[edge_indices]
    
    # Compute dot products (batch operation)
    dots = jnp.sum(n * desired_normals, axis=1)
    
    # Calculate constraint energy
    energy = jnp.mean((1.0 - dots)**2)
    
    return energy

@jit
def compute_one_normal_energy_jax(thetas, one_normal):
    """Compute one-normal energy using JAX for two-normal optimization"""
    if one_normal.size == 0:
        return 0.0
    
    theta_diffs = thetas[one_normal, 0] - thetas[one_normal, 1]
    return jnp.mean(theta_diffs**2)

@jit
def compute_pairwise_energy_jax(normals0, normals1, data):
    """
    Compute pairwise energy using JAX for two-normal case
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
    
    # Get a static upper bound for the number of pairs
    max_pairs = e1_indices.shape[0]
    
    # Use fori_loop for iteration
    energy_sum, weight_sum = jax.lax.fori_loop(
        0, max_pairs, 
        lambda i, accum: body_fun(i, accum), 
        (energy_sum, weight_sum)
    )
    
    # Normalize
    return jnp.where(weight_sum > 0, energy_sum / weight_sum, 0.0)

@jit
def compute_one_normal_pairwise_energy_jax(normals, data):
    """
    Compute pairwise energy for one-normal case using JAX
    """
    # Initialize accumulators
    energy_sum = 0.0
    weight_sum = 0.0
    
    # Get data for easier access
    e1_indices = data['e1_indices']
    e2_indices = data['e2_indices']
    weights = data['weights']
    rotation_matrices = data['rotation_matrices']
    has_rotation = data['has_rotation']
    
    # Process all pairs with a manual loop instead of scan
    def body_fun(i, accum):
        energy_sum, weight_sum = accum
        
        # Get indices and data for this pair
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        has_rot = has_rotation[i]
        rot_matrix = rotation_matrices[i]
        
        # Get normals for both edges
        n1 = normals[e1]
        n2 = normals[e2]
        
        # Apply rotation if needed (using JAX's where for conditional)
        n1_rot = jnp.where(has_rot, jnp.dot(rot_matrix, n1), n1)
        
        # Compute cost
        cost = (1.0 - jnp.dot(n1_rot, n2))**2
        
        # Calculate pair energy
        pair_energy = weight * cost
        
        return energy_sum + pair_energy, weight_sum + weight
    
    # Get a static upper bound for the number of pairs
    max_pairs = e1_indices.shape[0]
    
    # Use fori_loop for iteration
    energy_sum, weight_sum = jax.lax.fori_loop(
        0, max_pairs, 
        lambda i, accum: body_fun(i, accum), 
        (energy_sum, weight_sum)
    )
    
    # Normalize
    return jnp.where(weight_sum > 0, energy_sum / weight_sum, 0.0)

# Energy functions for optimization
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
def energy_one_normal_jax(thetas, Us, Vs, constraints, data):
    """
    JAX-optimized energy function for one normal per edge
    """
    # Calculate all normals
    normals = jnp.cos(thetas[:, jnp.newaxis]) * Us + jnp.sin(thetas[:, jnp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = compute_one_normal_constraint_energy_jax(normals, constraints)
    
    # ===== Pairwise Energy =====
    E_pairwise = compute_one_normal_pairwise_energy_jax(normals, data)
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise

# Unified callback creation function with different behavior for one/two normals
def create_callback(Us, Vs, E, P, V, mode='two'):
    """
    Create a callback function for visualization during optimization
    
    Parameters:
    - Us, Vs: Frame vectors for each edge
    - E: List of edge vertex indices
    - P: Polyline data
    - V: Vertex coordinates
    - mode: 'one' for one-normal, 'two' for two-normal optimization
    
    Returns:
    - Callback function for visualization
    """
    iteration_counter = [0]
    viz_frequency = 10
    
    if mode == 'one':
        def callback(thetas_now):
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]
            print(f"Iteration {current_iter}")
            
            if current_iter % viz_frequency != 0 and current_iter > 5:
                return False
            
            # Calculate normals for one-normal case
            normals_now = {}
            for edge_idx, theta in enumerate(thetas_now):
                normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
                normal = normal / np.linalg.norm(normal)
                normals_now[edge_idx] = normal
            
            # Update the plot for one-normal case
            plot_edge_constraints(
                V, E, P, normals_now, 
                unconstrained_polylines_indices=None,
                scale=0.08,
                str=f"One-Normal Optimization: Iteration {current_iter}",
                block=False
            )
            
            return False
    else:  # two-normal case
        def callback(thetas_flat_now):
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]
            print(f"Iteration {current_iter}")
            
            if current_iter % viz_frequency != 0 and current_iter > 5:
                return False
            
            # Reshape and calculate normals for two-normal case
            num_edges = len(E)
            thetas_2d_now = thetas_flat_now.reshape(num_edges, 2)
            
            edge_indices = np.arange(num_edges)
            which_edges = np.array([0, 1])
            
            thetas = thetas_2d_now.reshape(-1)
            
            normals_now = {}
            cos_theta = np.cos(thetas)
            sin_theta = np.sin(thetas)
            
            for i, edge_idx in enumerate(edge_indices):
                for j, which_edge in enumerate(which_edges):
                    flat_idx = edge_idx * 2 + which_edge
                    normal = cos_theta[flat_idx] * Us[edge_idx] + sin_theta[flat_idx] * Vs[edge_idx]
                    norm = np.sqrt(np.sum(normal * normal))
                    normal = normal / norm
                    normals_now[(edge_idx, which_edge)] = normal
            
            # Update the plot for two-normal case
            plot_edge_constraints_two_normals(
                V, E, P, normals_now, 
                unconstrained_polylines_indices=None,
                scale=0.08,
                str=f"Two-Normal Optimization: Iteration {current_iter}",
                block=False
            )
            
            return False
    
    return callback

# Unified functions to recover normals from optimization results
def recover_normals(result, Us, Vs, E=None, mode='two'):
    """
    Unified function to recover normals from optimization results
    
    Parameters:
    - result: Optimization result
    - Us, Vs: Frame vectors for each edge
    - E: Edge data (required for two-normal mode)
    - mode: 'one' for one-normal, 'two' for two-normal optimization
    
    Returns:
    - Dictionary of normals
    """
    if mode == 'one':
        thetas = result.x
        normals = {}
        for edge_idx, theta in enumerate(thetas):
            normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
            normal = normal / np.linalg.norm(normal)
            normals[edge_idx] = normal
    else:  # two-normal mode
        assert E is not None, "Edge data (E) is required for two-normal mode"
        num_edges = len(E)
        thetas_2d = result.x.reshape(num_edges, 2)
        
        normals = {}
        for edge_idx in range(num_edges):
            for which_edge in (0, 1):
                theta = thetas_2d[edge_idx, which_edge]
                normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
                normal = normal / np.linalg.norm(normal)
                normals[(edge_idx, which_edge)] = normal
    
    return normals

# Unified optimization function with different behavior for one/two normals
def optimize_with_jax(thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map, 
                      mode='two', one_normal=None, callback_fn=None):
    """
    Unified JAX-based optimization function for both one-normal and two-normal cases
    
    Parameters:
    - thetas0: Initial theta values for each edge
    - Us, Vs: Frame vectors for each edge
    - edge_constraints: List of tuples (edge_index, desired_normal_vector)
    - pairwise: List of tuples (edge1, edge2, weight)
    - rotations_data: Dictionary with rotation data
    - E: List of edge vertex indices
    - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices
    - mode: 'one' for one-normal, 'two' for two-normal optimization
    - one_normal: List of edge indices that should have only one normal (for two-normal mode)
    - callback_fn: Optional callback function for visualization
    
    Returns:
    - Optimization result
    """
    start_time = time.time()
    
    # Convert inputs to JAX arrays
    Us_jax = jnp.array(Us)
    Vs_jax = jnp.array(Vs)
    
    # Convert edge_constraints to JAX-friendly format
    constraints_jax = prepare_constraints_jax(edge_constraints)
    
    # Pre-process data for faster computation using improved rotations
    data = preprocess_data_for_jax_with_improved_rotations(E, pairwise, rotations_data, vertex_to_edges_map)
    
    if mode == 'one':
        # One normal per edge
        energy_jax_jit = jit(energy_one_normal_jax)
        
        def energy_wrapper(thetas):
            return energy_jax_jit(thetas, Us_jax, Vs_jax, constraints_jax, data).item()
        
        grad_fn = grad(energy_one_normal_jax, argnums=0)
        grad_jit = jit(grad_fn)
        
        def grad_wrapper(thetas):
            return np.array(grad_jit(thetas, Us_jax, Vs_jax, constraints_jax, data))
        
        # Use thetas0 directly for one-normal case
        initial_thetas = thetas0
        
    else:  # Two normals per edge
        # Create initial 2D array of thetas
        num_edges = len(thetas0)
        thetas_2d = np.column_stack((thetas0, thetas0))
        thetas_2d[:, 1] += 1e-3  # Small perturbation for second normal
        initial_thetas = thetas_2d.flatten()
        
        if one_normal is None:
            one_normal = []
        one_normal_jax = jnp.array(one_normal)
        
        energy_jax_jit = jit(energy_two_normal_jax)
        
        def energy_wrapper(thetas_flat):
            thetas_2d = jnp.array(thetas_flat).reshape(num_edges, 2)
            return energy_jax_jit(thetas_2d, Us_jax, Vs_jax, constraints_jax, data, one_normal_jax).item()
        
        grad_fn = grad(energy_two_normal_jax, argnums=0)
        grad_jit = jit(grad_fn)
        
        def grad_wrapper(thetas_flat):
            thetas_2d = jnp.array(thetas_flat).reshape(num_edges, 2)
            grad_val = grad_jit(thetas_2d, Us_jax, Vs_jax, constraints_jax, data, one_normal_jax)
            return np.array(grad_val.reshape(-1))
    
    # Run optimization using scipy
    result = opt.minimize(
        energy_wrapper,
        initial_thetas,
        jac=grad_wrapper,
        method='L-BFGS-B',
        tol=0.0000001,
        options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1e5},
        callback=callback_fn
    )
    
    end_time = time.time()
    print(f"JAX {mode}-normal optimization took {end_time - start_time:.6f} seconds")
    
    return result

# Main function with option to choose optimization method
def make_cylinder_example_jax_with_options(normals_per_edge='two'):
    '''
    Example of using JAX-based optimization on a cylinder example with option to choose
    between one or two normals per edge
    
    Parameters:
    - normals_per_edge: 'one' or 'two' to specify the optimization method
    
    Returns:
    - Optimization result and computed normals
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
    
    # Create a callback function for the appropriate mode
    callback_fn = create_callback(Us, Vs, E, P, V, mode=normals_per_edge)
    
    # Set specific edges for one-normal constraint in two-normal mode
    one_normal = [23, 58] if normals_per_edge == 'two' else None
    
    # Run the unified optimization function with appropriate mode
    result = optimize_with_jax(
        thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map,
        mode=normals_per_edge, one_normal=one_normal, callback_fn=callback_fn
    )
    
    # Recover normals using the unified function
    normals = recover_normals(result, Us, Vs, E, mode=normals_per_edge)
    
    # Visualize results
    if normals_per_edge == 'one':
        plot_edge_constraints(
            V, E, P, normals, unconstrained_polylines_indices=None, 
            scale=0.08, str="One-Normal Optimization Result", block=True
        )
    else:
        plot_edge_constraints_two_normals(
            V, E, P, normals, unconstrained_polylines_indices=None, 
            scale=0.08, str="Two-Normal Optimization Result", block=True
        )
    
    return result, normals

if __name__ == "__main__":
    # Uncomment the mode you want to run
    
    # For one normal per edge:
    result, normals = make_cylinder_example_jax_with_options(normals_per_edge='one')
    
    # For two normals per edge:
    # result, normals = make_cylinder_example_jax_with_options(normals_per_edge='two')
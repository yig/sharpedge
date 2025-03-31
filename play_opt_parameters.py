import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import time
import numpy as np
import scipy.optimize as opt

from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_edge_info, plot_edge_constraints, plot_edge_constraints_two_normals, plot_edge_frames
from opt_edges import create_frames_for_each_polyline, edge_distance_matrix, extract_pairwise_weight, build_vertex_to_edges_map, estimate_initial_thetas
from collections import deque
from utility_rotate_vector import rotation_matrix_from

def edge_to_edge_normal_transport(V, e0, e1, n0, tol=1e-10):
    '''
    Parallel transport a normal vector n0 from edge e0 to edge e1.
    The edges must share a vertex.
    
    Parameters:
    - V: array of vertex coordinates, shape (num_vertices, 3)
    - e0: first edge as tuple/list of vertex indices (e0_v_idx0, e0_v_idx1)
    - e1: second edge as tuple/list of vertex indices (e1_v_idx0, e1_v_idx1)
    - n0: normal vector for edge e0, shape (3,)
    - tol: tolerance for floating point comparisons (default 1e-10)
    
    Returns:
    - n1: transported normal vector for edge e1, shape (3,)
    '''
    
    # Ensure the edges share a vertex
    shared_indices = set(e0) & set(e1)
    assert len(shared_indices) != 0, "Edges must share at least one vertex"
    
     ## Get the shared index
    shared_index = next(iter(shared_indices))
    ## Get the non-shared index from e0
    e0_other_index = next( iter( set(e0) - shared_indices ))
    ## Get the non-shared index from e1
    e1_other_index = next( iter( set(e1) - shared_indices ))

    t0 = V[shared_index] - V[e0_other_index]
    t1 = V[e1_other_index] - V[shared_index]
    
    # Normalize tangent vectors
    t0 = t0 / np.linalg.norm(t0)
    t1 = t1 / np.linalg.norm(t1)
    
    # Normalize input normal
    n0 = n0 / np.linalg.norm(n0)
    
    # Compute the rotation axis (perpendicular to the plane containing t0 and t1)
    B = np.cross(t0, t1)
    B_norm = np.linalg.norm(B)
    
    if B_norm < tol:
        # The tangent vectors are parallel (or anti-parallel)
        # In this case, parallel transport preserves the normal vector
        return n0
    else:
        # Normalize the rotation axis
        B_hat = B / B_norm
        
        # Compute the angle between the tangent vectors
        cos_theta = np.clip(np.dot(t0, t1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Rotate the normal vector
        # Implementation of Rodrigues' rotation formula
        n1 = n0 * np.cos(theta) + np.cross(B_hat, n0) * np.sin(theta) + B_hat * np.dot(B_hat, n0) * (1 - np.cos(theta))
        
        return n1
    
def propagate_normals_from_constraints(V, E, vertex_to_edges_map, edge_constraints, transport_normal_function):
    """
    Propagate normal vectors from constrained edges to as many other edges as possible.
    
    Parameters:
    - V: array of vertex coordinates, shape (num_vertices, 3)
    - E: array of edge vertex indices, shape (num_edges, 2)
    - vertex_to_edges_map: dictionary mapping vertex indices to lists of edge indices
    - edge_constraints: list of tuples (edge_index, normal_vector)
    - transport_normal_function: function to transport normal from one edge to another
    
    Returns:
    - list of tuples (edge_index, normal_vector) for all edges with computed normals
    """
    # Initialize edge normals dictionary with constraints
    edge_normals = {edge_idx: normal.copy() for edge_idx, normal in edge_constraints}
    
    # Keep track of edges we've already processed
    processed_edges = set(edge_normals.keys())
    
    # Queue for breadth-first traversal
    queue = deque(processed_edges)
    
    while queue:
        current_edge = queue.popleft()
        current_normal = edge_normals[current_edge]
        
        # Get the two vertices of the current edge
        v1, v2 = E[current_edge]
        
        # For each vertex of the current edge
        for vertex in [v1, v2]:
            # Get all edges connected to this vertex
            connected_edges = vertex_to_edges_map[vertex]
            
            # Process each connected edge that hasn't been processed yet
            for next_edge in connected_edges:
                if next_edge != current_edge and next_edge not in processed_edges:
                    # Transport the normal from current_edge to next_edge
                    transported_normal = transport_normal_function(
                        V, 
                        E[current_edge], 
                        E[next_edge], 
                        current_normal
                    )
                    
                    # Store the transported normal
                    edge_normals[next_edge] = transported_normal
                    
                    # plot_edge_constraints(V, E, None, edge_normals, unconstrained_polylines_indices=None, str = f'{current_edge} - {next_edge}', block=False)
                    # Mark as processed and add to queue for further propagation
                    processed_edges.add(next_edge)
                    queue.append(next_edge)
    
    # Convert the dictionary to a list of tuples in the requested format
    return [(edge_idx, normal) for edge_idx, normal in edge_normals.items()]

def precompute_edge_rotation_map(V, E):
    '''
    Create an optimized edge rotation map with precomputed rotation matrices.
    Designed for JAX-compatible operations and optimized for pairwise energy computation.

    Parameters:
    - V: (n, 3) array of vertex coordinates (JAX DeviceArray or NumPy array)
    - E: (m, 2) array of edge vertex index pairs (JAX DeviceArray or NumPy array)

    Returns:
    - Dictionary with JAX-compatible rotation data optimized for pairwise energy computation
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

def preprocess_edge_pair_data(E, pairwise, rotations_data, vertex_to_edges_map):
    """
    Preprocess geometric edge pair data into an optimized format for energy computation.
    Produces JAX-compatible arrays specifically designed for efficient computation.

    Parameters:
    - E: List of edge vertex indices
    - pairwise: List of tuples (edge1, edge2, weight)
    - rotations_data: Dictionary with rotation data between edge pairs
    - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices

    Returns:
    - Dictionary with JAX arrays containing pre-processed edge pair data ready for computation
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

def prepare_edge_constraints(edge_constraints):
    """
    Convert edge constraints to a computation-friendly format using JAX arrays.
    
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
def compute_two_normal_constraint_energy(normals0, normals1, constraints):
    """
    Compute constraint energy for edges with two normal vectors per edge.
    
    Measures the alignment between computed normal vectors and desired normal vectors
    by calculating the mean squared difference from perfect alignment.
    
    Parameters:
    - normals0: First set of normal vectors for all edges
    - normals1: Second set of normal vectors for all edges
    - constraints: Dictionary with 'indices' and 'normals' keys
    
    Returns:
    - Scalar energy value (lower means better constraint satisfaction)
    """
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
def compute_one_normal_constraint_energy(normals, constraints):
    """
    Compute constraint energy for edges with one normal vector per edge.
    
    Measures the alignment between computed normal vectors and desired normal vectors
    by calculating the mean squared difference from perfect alignment.
    
    Parameters:
    - normals: Normal vectors for all edges
    - constraints: Dictionary with 'indices' and 'normals' keys
    
    Returns:
    - Scalar energy value (lower means better constraint satisfaction)
    """
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
def compute_two_normals_coherence_energy(thetas, shared_normal_indices):
    """
    Calculates the coherence energy for edges where two normals should be identical.
    
    For edges that should have the same normal on both sides (e.g., at planar junctions),
    this function measures how well the two normal angles are coherent, penalizing
    differences between them.
    
    Parameters:
    -----------
    thetas : jnp.ndarray
        Array of shape (N, 2) containing angles for each of the two normals per edge.
    shared_normal_indices : jnp.ndarray
        Indices of edges where the two normals should be identical.
        
    Returns:
    --------
    float
        Mean squared difference between angles for the two normals on selected edges.
        A value of 0 indicates perfect coherence (identical normals).
    """
    if shared_normal_indices.size == 0:
        return 0.0
        
    theta_diffs = thetas[shared_normal_indices, 0] - thetas[shared_normal_indices, 1]
    return jnp.mean(theta_diffs**2)

@jit
def compute_two_normal_pairwise_energy(normals0, normals1, data):
    """
    Compute pairwise alignment energy between edges with dual normals.
    
    This function evaluates how well normal vectors align across edge pairs when
    each edge has two normal vectors. It handles special cases for curve edges
    and applies rotations where necessary.
    
    Parameters:
    -----------
    normals0 : jnp.ndarray
        First set of normal vectors for each edge.
    normals1 : jnp.ndarray
        Second set of normal vectors for each edge.
    data : dict
        Dictionary containing pair indices, weights, and geometric information.
        
    Returns:
    --------
    float
        Normalized pairwise alignment energy. Lower values indicate better alignment.
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
def compute_one_normal_pairwise_energy(normals, data):
    """
    Compute pairwise alignment energy between edges with single normals.
    
    This function evaluates how well normal vectors align across edge pairs when
    each edge has only one normal vector. It applies rotations where necessary.
    
    Parameters:
    -----------
    normals : jnp.ndarray
        Normal vectors for each edge.
    data : dict
        Dictionary containing pair indices, weights, and geometric information.
        
    Returns:
    --------
    float
        Normalized pairwise alignment energy. Lower values indicate better alignment.
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
def compute_two_normal_total_energy(thetas, Us, Vs, constraints, data, one_normal):
    """
    Calculate total energy for the two-normal-per-edge optimization.
    
    This function computes the weighted sum of constraint satisfaction energy,
    normal coherence energy, and pairwise alignment energy for a mesh with
    two normals per edge.
    
    Parameters:
    -----------
    thetas : jnp.ndarray
        Array of shape (N, 2) containing angle parameters for each edge.
    Us : jnp.ndarray
        First basis vectors for normal construction.
    Vs : jnp.ndarray
        Second basis vectors for normal construction.
    constraints : dict
        Constraint information for normal vectors.
    data : dict
        Pairwise relation data for evaluating alignment.
    one_normal : jnp.ndarray
        Indices of edges where both normals should be identical.
        
    Returns:
    --------
    float
        Total weighted energy combining all energy components.
    """
    # Pre-calculate all normals for both orientations
    normals0 = jnp.cos(thetas[:, 0, jnp.newaxis]) * Us + jnp.sin(thetas[:, 0, jnp.newaxis]) * Vs
    normals1 = jnp.cos(thetas[:, 1, jnp.newaxis]) * Us + jnp.sin(thetas[:, 1, jnp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = compute_two_normal_constraint_energy(normals0, normals1, constraints)
    
    # ===== One Normal Energy =====
    E_one_normal = compute_two_normals_coherence_energy(thetas, one_normal)
    
    # ===== Pairwise Energy =====
    E_pairwise = compute_two_normal_pairwise_energy(
        normals0, 
        normals1, 
        data
    )
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal

@jit
def compute_one_normal_total_energy(thetas, Us, Vs, constraints, data):
    """
    Calculate total energy for the one-normal-per-edge optimization.
    
    This function computes the weighted sum of constraint satisfaction energy
    and pairwise alignment energy for a mesh with one normal per edge.
    
    Parameters:
    -----------
    thetas : jnp.ndarray
        Angle parameters for each edge.
    Us : jnp.ndarray
        First basis vectors for normal construction.
    Vs : jnp.ndarray
        Second basis vectors for normal construction.
    constraints : dict
        Constraint information for normal vectors.
    data : dict
        Pairwise relation data for evaluating alignment.
        
    Returns:
    --------
    float
        Total weighted energy combining constraint and pairwise components.
    """
    # Calculate all normals
    normals = jnp.cos(thetas[:, jnp.newaxis]) * Us + jnp.sin(thetas[:, jnp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = compute_one_normal_constraint_energy(normals, constraints)
    
    # ===== Pairwise Energy =====
    E_pairwise = compute_one_normal_pairwise_energy(normals, data)
    
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
def recover_normals(result, Us, Vs, mode='two'):
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
    else:  
        num_edges = len(Us)
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
def optimize_normal_angles(thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map, 
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
    constraints_jax = prepare_edge_constraints(edge_constraints)
    
    # Pre-process data for faster computation using improved rotations
    data = preprocess_edge_pair_data(E, pairwise, rotations_data, vertex_to_edges_map)
    
    if mode == 'one':
        # One normal per edge
        energy_jax_jit = jit(compute_one_normal_total_energy)
        
        def energy_wrapper(thetas):
            return energy_jax_jit(thetas, Us_jax, Vs_jax, constraints_jax, data).item()
        
        grad_fn = grad(compute_one_normal_total_energy, argnums=0)
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
        
        energy_jax_jit = jit(compute_two_normal_total_energy)
        
        def energy_wrapper(thetas_flat):
            thetas_2d = jnp.array(thetas_flat).reshape(num_edges, 2)
            return energy_jax_jit(thetas_2d, Us_jax, Vs_jax, constraints_jax, data, one_normal_jax).item()
        
        grad_fn = grad(compute_two_normal_total_energy, argnums=0)
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
    rotations_data = precompute_edge_rotation_map(V, E)
    
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
    # one_normal = []

    # Run the unified optimization function with appropriate mode
    result = optimize_normal_angles(
        thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map,
        mode=normals_per_edge, one_normal=one_normal, callback_fn=callback_fn
    )
    
    # Recover normals using the unified function
    normals = recover_normals(result, Us, Vs, mode=normals_per_edge)
    
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


def make_cube_example_01(normals_per_edge = 'two'):
    '''
    '''
    V, E, P = load_sketch_polyline_data('made_examples/sketch/cube.obj')

    plot_edge_info(V, E)

    edge_constraints = [(6, np.array([0, 1, 0])),
                        (3, np.array([1, 0, 0]))]
    plot_edge_constraints(V, E, P, edge_constraints, scale=0.08, str = 'manually created edge constraints', filename= None, block= True)


    # computes local coordinate frames for edges by generating parallel transport 
    # frames along polylines and mapping them to the global edge indices
    Us, Vs = create_frames_for_each_polyline( V, E, P )


    # plot_edge_frames(V, E, P, Us, Vs, scale=0.08)



  # Compute edge distances
    distances = edge_distance_matrix(V, E)
    pairwise = extract_pairwise_weight(E, distances)
    
    # Create JAX-friendly edge rotation map
    rotations_data = precompute_edge_rotation_map(V, E)
    
    # Create vertex to edges map
    vertex_to_edges_map = build_vertex_to_edges_map(E)

    # Propagate normals as initial guess
    estimate_normals = propagate_normals_from_constraints(V, E, vertex_to_edges_map, edge_constraints, edge_to_edge_normal_transport)
    
    # Get initial thetas
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)
    
    # Create a callback function for the appropriate mode
    callback_fn = create_callback(Us, Vs, E, P, V, mode=normals_per_edge)

  # Create a callback function for the appropriate mode
    callback_fn = create_callback(Us, Vs, E, P, V, mode=normals_per_edge)
    
    # Set specific edges for one-normal constraint in two-normal mode
    one_normal = []
    
    # Run the unified optimization function with appropriate mode
    result = optimize_normal_angles(
        thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map,
        mode=normals_per_edge, one_normal=one_normal, callback_fn=callback_fn
    )
    
    # Recover normals using the unified function
    normals = recover_normals(result, Us, Vs, mode=normals_per_edge)
    
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
    
    # # For one normal per edge:
    result, normals = make_cylinder_example_jax_with_options(normals_per_edge='one')
    
    # # For two normals per edge:
    result, normals = make_cylinder_example_jax_with_options(normals_per_edge='two')

    result, normals = make_cube_example_01(normals_per_edge='one')
    result, normals = make_cube_example_01(normals_per_edge='two')
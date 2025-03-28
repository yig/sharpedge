from collections import  deque
from utility_io import load_sketch_polyline_data
from utility_plot_viewer import plot_edge_info, plot_edge_constraints
from opt_edges import create_frames_for_each_polyline, edge_distance_matrix, extract_pairwise_weight, create_edge_rotation_map, build_vertex_to_edges_map, estimate_initial_thetas




import autograd.numpy as anp
import autograd
import numpy as np
import scipy.optimize as opt
import time


def optimize_with_single_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, vertex_to_edges_map, callback_fn = None):
    """
    Optimized version of single-normal per edge optimization using autograd
    """
    start_time = time.time()
    
    # Convert inputs to autograd numpy arrays
    Us = anp.array(Us)
    Vs = anp.array(Vs)
    
    # Pre-process data for faster computation
    data = preprocess_data_for_single_normal(pairwise, rotations)
    
    # Define wrapper for optimization
    def E_total_wrapper(thetas, Us, Vs, constraints, pairwise_data):
        return E_total_single_normal_optimized(thetas, Us, Vs, constraints, pairwise_data)
    


    # Run optimization
    result = opt.minimize(
        E_total_wrapper,
        thetas0,
        jac=autograd.grad(E_total_wrapper),
        args=(Us, Vs, edge_constraints, data),
        method='L-BFGS-B',
        tol=0.0000001,
        options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1000},
        callback= callback_fn
    )
    
    end_time = time.time()
    print(f"Optimized single-normal optimization took {end_time - start_time:.6f} seconds")
    
    return result

def preprocess_data_for_single_normal(pairwise, rotations):
    """
    Pre-process data structures for faster single-normal optimization
    """
    # Extract pairwise data
    e1_indices = [p[0] for p in pairwise]
    e2_indices = [p[1] for p in pairwise]
    weights = [p[2] for p in pairwise]
    
    # Convert to arrays for faster indexing
    e1_indices = np.array(e1_indices)
    e2_indices = np.array(e2_indices)
    weights = np.array(weights)
    
    return {
        'e1_indices': e1_indices,
        'e2_indices': e2_indices,
        'weights': weights,
        'rotations': rotations
    }

def E_total_single_normal_optimized(thetas, Us, Vs, constraints, pairwise_data):
    """
    Optimized energy function for a single normal per edge
    
    Parameters:
    - thetas: An array of N real numbers, one per edge
    - Us: An N-by-3 array of vectors spanning the plane normal to each edge
    - Vs: An N-by-3 array of vectors spanning the plane normal to each edge
    - constraints: A sequence of pairs (edge_index, desired_normal_vector)
    - pairwise_data: Pre-processed pairwise data structure
    
    Returns:
    - The total energy
    """
    # Calculate all normals at once
    normals = anp.cos(thetas[:, anp.newaxis]) * Us + anp.sin(thetas[:, anp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = 0.0
    if constraints:
        constraint_sum = 0.0
        # This could be vectorized, but loop is often faster for smaller datasets with autograd
        for edge_index, desired_normal_vector in constraints:
            n = normals[edge_index]
            constraint_sum += (1.0 - anp.dot(n, desired_normal_vector))**2
        
        E_constraint = constraint_sum / len(constraints)
    
    # ===== Pairwise Energy =====
    E_pairwise = 0.0
    W_pairwise = 0.0
    
    # Extract pre-processed data
    e1_indices = pairwise_data['e1_indices']
    e2_indices = pairwise_data['e2_indices']
    weights = pairwise_data['weights']
    rotations = pairwise_data['rotations']
    
    # Process each pairwise constraint
    for i in range(len(e1_indices)):
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        
        # Get normals for the two edges
        n1 = normals[e1]
        n2 = normals[e2]
        
        # Apply rotation if needed
        # Autograd-friendly rotation application
        if (e1, e2) in rotations:
            n1 = anp.dot(rotations[(e1, e2)], n1)
        elif (e2, e1) in rotations:
            n1 = anp.dot(anp.transpose(rotations[(e2, e1)]), n1)
        
        # Calculate pairwise energy contribution
        dot_product = anp.dot(n1, n2)
        E_pairwise += weight * (1.0 - dot_product)**2
        W_pairwise += weight
    
    # Normalize pairwise energy
    if W_pairwise > 0:
        E_pairwise /= W_pairwise
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise

# Version with more vectorization - may be faster for large datasets
def E_total_single_normal_vectorized(thetas, Us, Vs, constraints, pairwise_data):
    """
    Vectorized energy function for a single normal per edge
    This version attempts to vectorize more operations for potentially better performance
    """
    # Calculate all normals at once
    normals = anp.cos(thetas[:, anp.newaxis]) * Us + anp.sin(thetas[:, anp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    if constraints:
        # Extract constraint indices and normals for vectorization
        constraint_indices = anp.array([c[0] for c in constraints])
        constraint_normals = anp.array([c[1] for c in constraints])
        
        # Calculate dot products vectorized
        constrained_normals = normals[constraint_indices]
        dots = anp.sum(constrained_normals * constraint_normals, axis=1)
        
        # Calculate constraint energy
        E_constraint = anp.mean((1.0 - dots)**2)
    else:
        E_constraint = 0.0
    
    # ===== Pairwise Energy =====
    # The pairwise energy is harder to vectorize due to the rotation lookups
    # We'll use the same approach as in the optimized function
    E_pairwise = 0.0
    W_pairwise = 0.0
    
    # Extract pre-processed data
    e1_indices = pairwise_data['e1_indices']
    e2_indices = pairwise_data['e2_indices']
    weights = pairwise_data['weights']
    rotations = pairwise_data['rotations']
    
    # Process each pairwise constraint
    for i in range(len(e1_indices)):
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        
        # Get normals for the two edges
        n1 = normals[e1]
        n2 = normals[e2]
        
        # Apply rotation if needed
        # Autograd-friendly rotation application
        if (e1, e2) in rotations:
            n1 = anp.dot(rotations[(e1, e2)], n1)
        elif (e2, e1) in rotations:
            n1 = anp.dot(anp.transpose(rotations[(e2, e1)]), n1)
        
        # Calculate pairwise energy contribution
        dot_product = anp.dot(n1, n2)
        E_pairwise += weight * (1.0 - dot_product)**2
        W_pairwise += weight
    
    # Normalize pairwise energy
    if W_pairwise > 0:
        E_pairwise /= W_pairwise
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise



def optimize_with_optimized_autograd(thetas0, Us, Vs, edge_constraints, pairwise, rotations, 
                                    E, vertex_to_edges_map, one_normal=None):
    """
    Optimized version of the two-normal per edge optimization using autograd
    """
    start_time = time.time()
    
    # Convert inputs to autograd numpy arrays
    Us = anp.array(Us)
    Vs = anp.array(Vs)
    
    # Pre-process data for faster computation
    data = preprocess_data_for_optimization(E, pairwise, rotations, vertex_to_edges_map)
    
    # Create initial 2D array of thetas
    num_edges = len(thetas0)
    thetas_2d = np.column_stack((thetas0, thetas0))
    thetas_2d[:, 1] += 1e-3
    thetas0_flat = thetas_2d.flatten()
    
    if one_normal is None:
        one_normal = []
    
    # Define wrapper that uses our optimized function
    def E_total_wrapper(thetas_flat, Us, Vs, constraints, pairwise_data, one_normal):
        num_edges = len(E)
        thetas_2d = thetas_flat.reshape(num_edges, 2)
        # return E_total_two_normal_optimized(thetas_2d, Us, Vs, constraints, pairwise_data, one_normal)
        return E_total_two_normal_vectorized(thetas_2d, Us, Vs, constraints, pairwise_data, one_normal)
    
    # Run optimization
    result = opt.minimize(
        E_total_wrapper,
        thetas0_flat,
        jac=autograd.grad(E_total_wrapper),
        args=(Us, Vs, edge_constraints, data, one_normal),
        method='L-BFGS-B',
        tol=0.0000001,
        options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 * 10}
    )
    
    end_time = time.time()
    print(f"Optimized autograd optimization took {end_time - start_time:.6f} seconds")
    
    return result

def preprocess_data_for_optimization(E, pairwise, rotations, vertex_to_edges_map):
    """
    Pre-process data structures for faster optimization
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
    
    return {
        'E': E_array,
        'e1_indices': e1_indices,
        'e2_indices': e2_indices,
        'weights': weights,
        'curve_edges': curve_edges,
        'rotations': rotations
    }

def E_total_two_normal_optimized(thetas, Us, Vs, constraints, pairwise_data, one_normal):
    """
    Optimized energy function for two normals per edge
    """
    # Pre-calculate all normals for both orientations
    normals0 = anp.cos(thetas[:, 0, anp.newaxis]) * Us + anp.sin(thetas[:, 0, anp.newaxis]) * Vs
    normals1 = anp.cos(thetas[:, 1, anp.newaxis]) * Us + anp.sin(thetas[:, 1, anp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    E_constraint = 0.0
    if constraints:
        constraint_sum = 0.0
        for edge_index, desired_normal_vector in constraints:
            n0 = normals0[edge_index]
            n1 = normals1[edge_index]
            constraint_sum += (1.0 - anp.dot(n0, desired_normal_vector))**2
            constraint_sum += (1.0 - anp.dot(n1, desired_normal_vector))**2
        E_constraint = constraint_sum / (2 * len(constraints))
    
    # ===== One Normal Energy =====
    E_one_normal = 0.0
    if one_normal and len(one_normal) > 0:
        one_normal_indices = anp.array(one_normal)
        theta_diffs = thetas[one_normal_indices, 0] - thetas[one_normal_indices, 1]
        E_one_normal = anp.mean(theta_diffs**2)
    
    # ===== Pairwise Energy =====
    E_pairwise = 0.0
    W_pairwise = 0.0
    
    # Get pairwise data
    e1_indices = pairwise_data['e1_indices']
    e2_indices = pairwise_data['e2_indices']
    weights = pairwise_data['weights']
    curve_edges = pairwise_data['curve_edges']
    rotations = pairwise_data['rotations']
    
    # For each pairwise constraint
    for i in range(len(e1_indices)):
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        is_curve = curve_edges[i]
        
        # Initialize cost matrix - use Python list for manual updates
        costs = [[0.0, 0.0], [0.0, 0.0]]
        
        # Calculate costs for all combinations
        for i_n1 in range(2):
            for i_n2 in range(2):
                n1 = normals0[e1] if i_n1 == 0 else normals1[e1]
                n2 = normals0[e2] if i_n2 == 0 else normals1[e2]
                
                # Apply rotation if needed
                if (e1, e2) in rotations:
                    n1 = anp.dot(rotations[(e1, e2)], n1)
                elif (e2, e1) in rotations:
                    n1 = anp.dot(anp.transpose(rotations[(e2, e1)]), n1)
                
                # Calculate dot product and cost, store in our cost matrix
                dot_val = anp.dot(n1, n2)
                costs[i_n1][i_n2] = (1.0 - dot_val)**2
        
        # Calculate energy based on edge type
        if is_curve:
            diagonal_sum = costs[0][0] + costs[1][1]
            antidiagonal_sum = costs[0][1] + costs[1][0]
            min_sum = anp.minimum(diagonal_sum, antidiagonal_sum)
            E_pairwise += weight * min_sum
        else:
            # Find minimum cost manually
            min_cost = anp.min(anp.array([[costs[0][0], costs[0][1]], [costs[1][0], costs[1][1]]]))
            E_pairwise += weight * min_cost
        
        W_pairwise += weight
    
    # Normalize pairwise energy
    if W_pairwise > 0:
        E_pairwise /= W_pairwise
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal

# Alternative implementation with more vectorization
def E_total_two_normal_vectorized(thetas, Us, Vs, constraints, pairwise_data, one_normal):
    """
    Alternative implementation with more vectorization
    Note: This might or might not be faster depending on your data size
    """
    # Pre-calculate all normals for both orientations
    normals0 = anp.cos(thetas[:, 0, anp.newaxis]) * Us + anp.sin(thetas[:, 0, anp.newaxis]) * Vs
    normals1 = anp.cos(thetas[:, 1, anp.newaxis]) * Us + anp.sin(thetas[:, 1, anp.newaxis]) * Vs
    
    # ===== Constraint Energy =====
    if constraints:
        # Extract constraint indices and normals
        constraint_indices = anp.array([c[0] for c in constraints])
        constraint_normals = anp.array([c[1] for c in constraints])
        
        # Compute dot products for both normals
        n0_dots = anp.sum(normals0[constraint_indices] * constraint_normals, axis=1)
        n1_dots = anp.sum(normals1[constraint_indices] * constraint_normals, axis=1)
        
        # Calculate constraint energy
        E_constraint = anp.mean((1.0 - n0_dots)**2 + (1.0 - n1_dots)**2) / 2
    else:
        E_constraint = 0.0
    
    # ===== One Normal Energy =====
    if one_normal and len(one_normal) > 0:
        one_normal_indices = anp.array(one_normal)
        theta_diffs = thetas[one_normal_indices, 0] - thetas[one_normal_indices, 1]
        E_one_normal = anp.mean(theta_diffs**2)
    else:
        E_one_normal = 0.0
    
    # ===== Pairwise Energy =====
    # For pairwise energy, we need to process each pair individually
    # This is harder to vectorize due to the rotations and curve edge logic
    E_pairwise = 0.0
    W_pairwise = 0.0
    
    # Get pairwise data
    e1_indices = pairwise_data['e1_indices']
    e2_indices = pairwise_data['e2_indices']
    weights = pairwise_data['weights']
    curve_edges = pairwise_data['curve_edges']
    rotations = pairwise_data['rotations']
    
    # For each pairwise constraint
    for i in range(len(e1_indices)):
        e1 = e1_indices[i]
        e2 = e2_indices[i]
        weight = weights[i]
        is_curve = curve_edges[i]
        
        # Initialize cost matrix using a Python list for better control
        costs = [[0.0, 0.0], [0.0, 0.0]]
        
        # Calculate costs for all combinations
        for i_n1 in range(2):
            for i_n2 in range(2):
                n1 = normals0[e1] if i_n1 == 0 else normals1[e1]
                n2 = normals0[e2] if i_n2 == 0 else normals1[e2]
                
                # Apply rotation if needed
                if (e1, e2) in rotations:
                    n1 = anp.dot(rotations[(e1, e2)], n1)
                elif (e2, e1) in rotations:
                    n1 = anp.dot(anp.transpose(rotations[(e2, e1)]), n1)
                
                # Calculate dot product and store in cost matrix
                dot_val = anp.dot(n1, n2)
                costs[i_n1][i_n2] = (1.0 - dot_val)**2
        
        # Calculate energy based on edge type
        if is_curve:
            diagonal_sum = costs[0][0] + costs[1][1]
            antidiagonal_sum = costs[0][1] + costs[1][0]
            min_sum = anp.minimum(diagonal_sum, antidiagonal_sum)
            E_pairwise += weight * min_sum
        else:
            # Find the minimum cost by converting to array
            costs_array = anp.array([[costs[0][0], costs[0][1]], [costs[1][0], costs[1][1]]])
            E_pairwise += weight * anp.min(costs_array)
        
        W_pairwise += weight
    
    # Normalize pairwise energy
    if W_pairwise > 0:
        E_pairwise /= W_pairwise
    
    # Return weighted sum of energies
    return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal

# Helper function to convert optimized results back to normals
def recover_normals_from_result(result, Us, Vs, E):
    """
    Recover normal vectors from optimization result
    """
    num_edges = len(E)
    thetas_2d = result.x.reshape(num_edges, 2)
    
    # Calculate normals for both edges
    normals = {}
    for edge_idx in range(num_edges):
        for which_edge in (0, 1):
            # Calculate the normal for this edge and orientation
            theta = thetas_2d[edge_idx, which_edge]
            normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
            
            # Store in the normals dictionary with tuple key (edge_idx, which_edge)
            normals[(edge_idx, which_edge)] = normal
    
    return normals

def create_callback(Us, Vs, E, P, V):
    """
    Create a callback function for visualization during optimization
    """
    iteration_counter = [0]
    
    def callback(thetas_now):
        iteration_counter[0] += 1
        print(f"Iteration {iteration_counter[0]}")
        
        # Calculate normals
        normals_now = {}
        for edge_idx in range(len(thetas_now)):
            theta = thetas_now[edge_idx]
            normal = np.cos(theta) * Us[edge_idx] + np.sin(theta) * Vs[edge_idx]
            normals_now[edge_idx] = normal
        
        # Update plot - this assumes the plot_edge_constraints function exists
        # Uncomment if you have this function
        # plot_edge_constraints(V, E, P, normals_now, 
        #                     unconstrained_polylines_indices=None, 
        #                     scale=0.08, 
        #                     str=f"Optimization: Iteration {iteration_counter[0]}", 
        #                     block=False)
        
        return False  # Continue optimization
    
    return callback

# Example usage:
# result = optimize_with_jax_jit(thetas0, Us, Vs, edge_constraints, pairwise, rotations)

# Usage example:
# result = optimize_with_jax_single_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map)
# Or for two normals:
# result = optimize_with_jax_two_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map, one_normal)


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


def propagate_normals_from_constraints_and_average(V, E, vertex_to_edges_map, edge_constraints, transport_normal_function):
    """
    Propagate normal vectors from constrained edges to as many other edges as possible.
    When an edge receives normals from multiple sources, they are averaged.
    
    Parameters:
    - V: array of vertex coordinates, shape (num_vertices, 3)
    - E: array of edge vertex indices, shape (num_edges, 2)
    - vertex_to_edges_map: dictionary mapping vertex indices to lists of edge indices
    - edge_constraints: list of tuples (edge_index, normal_vector)
    - transport_normal_function: function to transport normal from one edge to another
    
    Returns:
    - list of tuples (edge_index, normal_vector) for all edges with computed normals
    """
    # Initialize dictionaries to keep track of normal vectors and their counts
    edge_normal_sums = defaultdict(lambda: np.zeros(3))
    edge_normal_counts = defaultdict(int)
    
    # Initialize with constraints
    for edge_idx, normal in edge_constraints:
        normal_normalized = normal / np.linalg.norm(normal)
        edge_normal_sums[edge_idx] = normal_normalized
        edge_normal_counts[edge_idx] = 1
    
    # Keep track of edges to process
    # We'll use a set for edges that have been added to the queue at least once
    queued_edges = set(edge_idx for edge_idx, _ in edge_constraints)
    # And a deque for the actual processing queue
    queue = deque(queued_edges)
    
    # Keep track of which edges have contributed to which other edges
    # This avoids processing the same (source, target) edge pair multiple times
    processed_pairs = set()
    
    while queue:
        current_edge = queue.popleft()
        
        # Calculate the average normal for the current edge
        if edge_normal_counts[current_edge] > 0:
            current_normal = edge_normal_sums[current_edge] / edge_normal_counts[current_edge]
            current_normal = current_normal / np.linalg.norm(current_normal)
        else:
            continue  # Skip if no valid normal
        
        # Get the two vertices of the current edge
        v1, v2 = E[current_edge]
        
        # For each vertex of the current edge
        for vertex in [v1, v2]:
            # Get all edges connected to this vertex
            connected_edges = vertex_to_edges_map[vertex]
            
            # Process each connected edge
            for next_edge in connected_edges:
                if next_edge == current_edge:
                    continue
                
                # Check if this specific propagation path has been processed before
                pair_key = (current_edge, next_edge)
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Transport the normal from current_edge to next_edge
                transported_normal = transport_normal_function(
                    V, 
                    E[current_edge], 
                    E[next_edge], 
                    current_normal
                )
                
                # Normalize the transported normal
                transported_normal = transported_normal / np.linalg.norm(transported_normal)
                
                # Update the normal sum and count for the next edge
                edge_normal_sums[next_edge] += transported_normal
                edge_normal_counts[next_edge] += 1
                
                # Add to queue if not already queued
                if next_edge not in queued_edges:
                    queued_edges.add(next_edge)
                    queue.append(next_edge)
    
    # Calculate final averaged normals
    result = []
    for edge_idx, normal_sum in edge_normal_sums.items():
        if edge_normal_counts[edge_idx] > 0:
            avg_normal = normal_sum / edge_normal_counts[edge_idx]
            # Ensure the normal is normalized
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            result.append((edge_idx, avg_normal))
    
    return result
    


def make_cylinder_example_01():
    '''
    '''
    V, E, P = load_sketch_polyline_data('made_examples/sketch/cylinder.obj')

    plot_edge_info(V, E)


    # create constraints by hand 
    edge_constraints_dict = {0 :  np.array([0, 1, 0]),
                             1 :  np.array([0, -1, 0])}
    edge_constraints = [(0, np.array([0, 1, 0])),
                        (1, np.array([0, -1, 0]))]

    plot_edge_constraints(V, E, P, edge_constraints, scale=0.08, str = 'manually created edge constraints', filename= None, block= True)




    # computes local coordinate frames for edges by generating parallel transport 
    # frames along polylines and mapping them to the global edge indices
    Us, Vs = create_frames_for_each_polyline( V, E, P )


    # plot_edge_frames(V, E, P, Us, Vs, scale=0.08)





    distances = edge_distance_matrix(V, E)

    pairwise = extract_pairwise_weight(E, distances)

    rotations = create_edge_rotation_map(V, E)

    vertex_to_edges_map = build_vertex_to_edges_map( E )


    # how about just random the normal for the edges who don't have normals 
    # estimate_normals = generate_random_initial_guess(E, Us, Vs)
    # plot_edge_constraints(V, E, P, estimate_normals, scale=0.08, str = 'random normal', filename= None, block= True)

    # propagate normal as much as possible
    estimate_normals = propagate_normals_from_constraints(V, E, vertex_to_edges_map, edge_constraints, edge_to_edge_normal_transport)
    plot_edge_constraints(V, E, P, estimate_normals, scale=0.08, str = 'trying to propagate normal', filename= None, block= True)


    # turns out pure random cannot go correctly 
    # we must have 
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)
    # thetas0  = np.zeros(len(E))

    callback_fn = create_callback(Us, Vs, E, P, V)
    # result = optimize_with_jax_single_normal_simpler(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map)
    optimize_with_single_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, vertex_to_edges_map, callback_fn)

    one_normal = []
    one_normal = [23, 58]

    optimize_with_optimized_autograd(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map, one_normal)



    # optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='one')

    # optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='two', auto_grad = True)

    # optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='two', auto_grad = False)

# Usage example:
# result = optimize_with_jax_single_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map)
# Or for two normals:
# result = optimize_with_jax_two_normal(thetas0, Us, Vs, edge_constraints, pairwise, rotations, E, vertex_to_edges_map, one_normal)


if __name__ == "__main__":

    make_cylinder_example_01()
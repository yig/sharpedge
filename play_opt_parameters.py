from opt_edges import *
from utility_plot_viewer import plot_edge_info
from collections import  deque


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



    # optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='one')

    optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='two', auto_grad = True)

    # optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE='two', auto_grad = False)




def generate_random_initial_guess(E, Us, Vs):
    '''
    Generate random normal vectors for each edge
    '''
    estimate_normals = {}
    for index, edge in enumerate(E):
        estimate_normals[index] = random_normal_for_edge(Us[index], Vs[index])
    return estimate_normals

def optimize_noramals( V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations, vertex_to_edges_map, NORMALS_PER_EDGE = 'two', auto_grad = True):
    '''
    auto_grad: use jax to autograd or not
    '''

    if NORMALS_PER_EDGE == 'one':


        def E_total( thetas, Us, Vs, constraints, pairwise):
            '''
            Given a bag of edge data of the form:
                thetas: An array of N real numbers, one per edge
                Us: An N-by-3 array of vectors spanning the plane normal to each edge (along with Vs)
                Vs: An N-by-3 array of vectors spanning the plane normal to each edge (along with Us)
                constraints: A sequence of pairs ( edge index, desired normal vector ) such that edge "edge index" should have the normal "desired normal vector"
                pairwise: A sequence of triplets ( edge index 1, edge index 2, weight ) such that the difference in normals between "edge index 1" and "edge index 2" should be penalized with the given weight
            Returns:
                The total energy
            '''    

            normals = jnp.cos(thetas[:,np.newaxis]) * Us + jnp.sin(thetas[:,np.newaxis]) * Vs

            # Calculate the constraint energy
            E_constraint = 0.0
            for edge_index, desired_normal_vector in constraints:
                n = normals[edge_index]
                E_constraint += (1.0 - jnp.dot( n, desired_normal_vector ) )**2

            # normalize
            E_constraint /= len( constraints )
            
            E_pairwise = 0.0
            W_pairwise = 0.0
            for e1, e2, weight in pairwise:
                n1 = normals[e1]
                n2 = normals[e2]

                
                ## Get the rotation matrix if it exists
                if (e1,e2) in rotations: n1 = rotations[(e1,e2)] @ n1
                elif (e2,e1) in rotations: n1 = rotations[(e2,e1)].T @ n1

            
                E_pairwise += weight * (1.0 - jnp.dot( n1, n2 ) )**2   

                W_pairwise += weight
            # Normalize by the total weight
            E_pairwise /= W_pairwise
            
            return 1e-2 * E_constraint +  1e4 * E_pairwise
    
    


        # Showing the optimization process
        # Initialize iteration counter
        iteration_counter = [0]

        def callback(thetas_now):
            iteration_counter[0] += 1
            print(f"Iteration {iteration_counter[0]}")
            
            # Calculate normals
            normals_now = recover_normal_from_thetas(thetas_now, Us, Vs)
            
            # Update the plot - specify block=False for non-blocking
            plot_edge_constraints(V, E, P, normals_now, unconstrained_polylines_indices = None, 
                                scale=0.08, 
                                str=f"Optimization: Iteration {iteration_counter[0]}", 
                                block=False)




        import time
        start_time = time.time()
        jac = None
        if auto_grad:
            jac = jax.grad(E_total)
     
            
        
        
        result = opt.minimize( E_total,
                thetas0,  
                jac = jac,
                args=(Us, Vs, edge_constraints, pairwise),  # Pass additional arguments
                method = 'L-BFGS-B', 
                tol = 0.0000001, 
                options = { 'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 },
                # callback=callback
            )
        
        thetas = result.x

        
        end_time = time.time()
        # Calculate and print the execution time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

        opt_normals = recover_normal_from_thetas(thetas, Us, Vs)
        plot_edge_constraints(V, E, P,  opt_normals, unconstrained_polylines_indices = None, scale=0.08, str = "optimize result")

    elif NORMALS_PER_EDGE == 'two':

        # Assuming thetas0 is your 1D array with one value per edge
        num_edges = len(thetas0)


        # Create a 2D array where both columns are identical
        thetas_2d = np.column_stack((thetas0, thetas0))  # shape: (num_edges, 2)

        # Perturb the second normals by a small number
        thetas_2d[:,1] += 1e-3

        # Flatten this 2D array for the optimizer
        thetas0_flat = thetas_2d.flatten()  # shape: (2*num_edges,)

        # Constrain any normals we wish to have a single normal
        one_normal = []
        one_normal = [23, 58]

        # Add this wrapper function that reshapes the 1D array to 2D for your function
        def E_total_wrapper(thetas_flat, Us, Vs, constraints, pairwise, one_normal):
            # Reshape the 1D array to a 2D array
            num_edges = len(E)
            thetas_2d = thetas_flat.reshape(num_edges, 2)  
            return E_total_two_normal_per_edge(thetas_2d, Us, Vs, constraints, pairwise, one_normal)

        # Initialize iteration counter
        iteration_counter = [0]

        def callback(thetas_flat_now):
            iteration_counter[0] += 1
            print(f"Iteration {iteration_counter[0]}")
            
            # Reshape the flat thetas array to the 2D structure
            num_edges = len(E)
            thetas_2d_now = thetas_flat_now.reshape(num_edges, 2)
            
            # Calculate normals for both edges
            normals_now = {}
            # Calculate two normals per edge and format them for the plotting function
            for edge_idx in range(num_edges):
                for which_edge in (0, 1):
                    # Calculate the normal for this edge and orientation
                    normal = normal_for_edge(thetas_2d_now[edge_idx, which_edge], Us[edge_idx], Vs[edge_idx])
                
                    # Store in the normals dictionary with tuple key (edge_idx, which_edge)
                    normals_now[(edge_idx, which_edge)] = normal

                
         
            
            # Update the plot - specify block=False for non-blocking
            plot_edge_constraints_two_normals(V, E, P, normals_now, unconstrained_polylines_indices = None, 
                                scale=0.08, 
                                str=f"Two-Normal Optimization: Iteration {iteration_counter[0]}", 
                                block=False)

        
        def E_total_two_normal_per_edge(thetas, Us, Vs, constraints, pairwise, one_normal):
            '''
            '''

            # Pre-calculate all normals for both edges
            normals0 = jnp.cos(thetas[:,0,np.newaxis]) * Us + jnp.sin(thetas[:,0,np.newaxis]) * Vs
            normals1 = jnp.cos(thetas[:,1,np.newaxis]) * Us + jnp.sin(thetas[:,1,np.newaxis]) * Vs

            # Calculate the constraint energy
            E_constraint = 0.0
            for edge_index, desired_normal_vector in constraints:
                # Use pre-calculated normals
                n0 = normals0[edge_index]  # First normal
                n1 = normals1[edge_index]  # Second normal
                
                # Calculate constraint energy for both normals
                E_constraint += (1.0 - jnp.dot(n0, desired_normal_vector))**2
                E_constraint += (1.0 - jnp.dot(n1, desired_normal_vector))**2
            # normalize with a 2 because we have two normals
            E_constraint /= 2*len(constraints)

            # Calculate one_normal energy
            E_one_normal = 0.0
            if len(one_normal) != 0:
                # This part can be vectorized if one_normal is a list/array of indices
                one_normal_indices = jnp.array(one_normal)
                if len(one_normal_indices) > 0:
                    # Get the thetas for all one_normal edges
                    theta_diffs = thetas[one_normal_indices, 0] - thetas[one_normal_indices, 1]
                    E_one_normal = jnp.mean(theta_diffs**2)

            # Calculate pairwise energy
            E_pairwise = 0.0
            W_pairwise = 0.0
            for e1, e2, weight in pairwise:
                # Initialize cost matrix
                costs = [ [None, None], [None, None] ]
                
                # Fill cost matrix
                for i in range(2):
                    for j in range(2):
                        n1 = normals0[e1] if i == 0 else normals1[e1]
                        n2 = normals0[e2] if j == 0 else normals1[e2]
                        
                        # Apply rotation if needed
                        if (e1, e2) in rotations: 
                            n1 = rotations[(e1, e2)] @ n1
                        elif (e2, e1) in rotations: 
                            n1 = rotations[(e2, e1)].T @ n1
                            
                        costs[i][j] = (1.0 - jnp.dot( n1, n2 ) )**2
                
                costs = jnp.array( costs )

                shared_vertex = tuple(frozenset(E[e1]) & frozenset(E[e2]))
                
                # If this is a curve edge, penalize the best match
                if len(shared_vertex) == 1 and len(vertex_to_edges_map[shared_vertex[0]]) == 2:
                    diagonal_sum = costs[0, 0] + costs[1, 1]
                    antidiagonal_sum = costs[0, 1] + costs[1, 0]
                    
                    # Add the smaller sum to the energy
                    min_sum = jnp.minimum(diagonal_sum, antidiagonal_sum)
                    E_pairwise += weight * min_sum
                    W_pairwise += weight
                # Otherwise, edges are disconnected or higher-valence
                else:
                    E_pairwise += weight * jnp.min(costs)
                    W_pairwise += weight

            # Normalize pairwise energy by total weight
            if W_pairwise > 0:
                E_pairwise /= W_pairwise

            # Return the total energy with appropriate weights
            return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal


        # def E_total_two_normal_per_edge(thetas, Us, Vs, constraints, pairwise, one_normal):
        #     '''
        #     Optimized version calculating energy for two normals per edge using Autograd.
        #     '''
        #     E_constraint = 0.0
        #     for edge_index, desired_normal_vector in constraints:
        #         for which_edge in (0,1):
        #             n = normal_for_edge( thetas[ edge_index, which_edge ], Us[ edge_index ], Vs[ edge_index ] )
        #             E_constraint += (1.0 - jnp.dot( n, desired_normal_vector ) )**2
        #     # normalize with a 2 because we have two normals
        #     E_constraint /= 2*len( constraints )

        #     E_one_normal = 0.0
        #     if len(one_normal) > 0:
        #         for edge_index in one_normal:
        #             E_one_normal += ( thetas[ edge_index, 0 ] - thetas[ edge_index, 1 ] )**2
        #         # normalize without a 2 because this is operating on edges
        #         E_one_normal /= len( one_normal )
            
        #     E_pairwise = 0.0
        #     W_pairwise = 0.0
        #     for e1, e2, weight in pairwise:
        #         costs = [ [None, None], [None, None] ]
        #         for i in range(2):
        #             for j in range(2):
        #                 n1 = normal_for_edge( thetas[e1,i], Us[e1], Vs[e1] )
        #                 n2 = normal_for_edge( thetas[e2,j], Us[e2], Vs[e2] )

        #                 ## Get the rotation matrix if it exists
        #                 if (e1,e2) in rotations: n1 = rotations[(e1,e2)] @ n1
        #                 elif (e2,e1) in rotations: n1 = rotations[(e2,e1)].T @ n1

        #                 costs[i][j] = (1.0 - jnp.dot( n1, n2 ) )**2
                
        #         costs = jnp.array( costs )

        #         # E_pairwise += weight * costs[0,1]
        #         # E_pairwise += weight * costs.diagonal().sum() * .5
        #         # E_pairwise += weight * costs.sum() * .25
        #         # W_pairwise += weight

        #         shared_vertex = tuple(frozenset( E[e1] ) & frozenset( E[e2] ))
                
        #         ## If this is a curve edge, we want to penalize the best match
        #         if len( shared_vertex ) == 1 and len(vertex_to_edges_map[ shared_vertex[0] ]) == 2:
        #             if costs[0,0] + costs[1,1] < costs[0,1] + costs[1,0]:
        #                 E_pairwise += weight * (costs[0,0] + costs[1,1])
        #                 W_pairwise += weight
        #             else:
        #                 E_pairwise += weight * (costs[0,1] + costs[1,0])
        #                 W_pairwise += weight
        #         ## Otherwise, the edges are disconnected or higher-valence, in which case we just want the
        #         ## lowest cost.
        #         else:
        #             E_pairwise += weight * costs.min()
        #             W_pairwise += weight
                
        #     # Normalize by the total weight
        #     E_pairwise /= W_pairwise

        #     # Return the total energy
        #     return 1e-2 * E_constraint  +  1e4 * E_pairwise  +  1e6 * E_one_normal

            
            
            
        '''
        LITTLE_NEWTON = False
        STEP_SIZE = .00001
        thetas_i = thetas0_flat.copy()
        args=(Us, Vs, edge_constraints, pairwise, one_normal)
        jac = jax.grad( E_total_wrapper )
        if LITTLE_NEWTON:
            hess = jax.hessian( E_total_wrapper )
        w_regular = 1e2
        for iter in range( 100 ):
            f = E_total_wrapper( thetas_i, *args )
            g = jac( thetas_i, *args )
            if LITTLE_NEWTON:
                H = hess( thetas_i, *args )
                H += w_regular * np.eye(len(thetas_i))
            else:
                H = (1./STEP_SIZE) * np.eye(len(thetas_i))
            print( "Iteration:", iter )
            print( "f:", f )
            print( "|g|:", np.linalg.norm( g ) )
            thetas_i -= np.linalg.solve( H, g )
            callback( thetas_i )

        from collections import namedtuple
        result = namedtuple("OptimizationResult", ["x"])( thetas_i )
        '''

        import time
        start_time = time.time()

        jac = None
        if auto_grad:
            jac = jax.grad( E_total_wrapper )


        result = opt.minimize(E_total_wrapper,
                            thetas0_flat,
                            jac = jac,
                            # hess = jax.hessian( E_total_wrapper ),
                            args=(Us, Vs, edge_constraints, pairwise, one_normal),
                            method='L-BFGS-B',
                            # method= 'Newton-CG',
                            tol=0.0000001,
                            options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 * 10},
                            # callback=callback
                            )
        
        # Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time

        print(f"Optimization took {duration:.6f} seconds")


    
                

        thetas0_opt = result.x
        thetas_2d_now = thetas0_opt.reshape(num_edges, 2)
            
        # Calculate normals for both edges
        opt_normals = {}
        # Calculate two normals per edge and format them for the plotting function
        for edge_idx in range(num_edges):
            for which_edge in (0, 1):
                # Calculate the normal for this edge and orientation
                normal = normal_for_edge(thetas_2d_now[edge_idx, which_edge], Us[edge_idx], Vs[edge_idx])
            
                # Store in the normals dictionary with tuple key (edge_idx, which_edge)
                opt_normals[(edge_idx, which_edge)] = normal


        plot_edge_constraints_two_normals(V, E, P, opt_normals, unconstrained_polylines_indices = None, 
                                scale=0.08, 
                                str="Two-Normal Optimization", 
                                block=True)
        



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
    

def make_cube_example_01():
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


    plot_edge_frames(V, E, P, Us, Vs, scale=0.08)





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

    optimize_noramals(V, E, P, Us, Vs, edge_constraints, thetas0, pairwise, rotations)

if __name__ == "__main__":

    make_cylinder_example_01()
    # make_cube_example_01()

    
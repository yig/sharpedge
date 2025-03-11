import numpy as np

from utility_io import load_sketch_polyline_data, write_normal_data, write_string_to_file
from utility_plot_viewer import plot_sketch_data, plot_edge_constraints, plot_edge_frames, plot_polyline_best_constraints, plot_polyline_normals
from utility_segment_distance import segment_to_segment_distance

from utility_convex_hull import get_sketch_edge_constraints, export_sketch_normal_gltf, export_sketch_dict_normal_gltf
from utility_parallel_transport import compute_parallel_transport_frames
from utility_parallel_transport_bidirection import parallel_transport_bi_direction
from utility_rotate_vector import rotation_matrix_from

import scipy.optimize as opt

from pathlib import Path
from collections import defaultdict

import argparse

# import jax.numpy as jnp
# import jax




# this should be a matrix 
#      e0, e1, ... en-1
# e0   0
# e1       0
# ...         
# en-1              0

# 1. the diagnoal all 0, the weights of an edge to itself.
# 2. for all the edges, compute their segment-segment distance, divide 1 / (distance + epsilon) to avoid divide 0
# 3. loop through the polyline, find the neighboring edges and add them another extra high weights 
def create_edge_weight_matrix(V, E, P, neighbor_weight=1, epsilon=1e-2):
    """
    Create a matrix for edges with weights based on segment distances and neighbor relationships.
    The matrix is built in three steps:
    1. Initialize diagonal elements to 0 (edge to itself)
    2. Compute weights for all edge pairs based on inverse segment distance
    3. Add extra high weights for neighboring edges in the same polyline
    
    Args:
        V: (n,3) array of vertex coordinates
        E: (m,2) array of edge vertex index pairs
        P: list of arrays containing vertex indices for each polyline
        neighbor_weight: extra weight to add for neighboring edges in same polyline
        epsilon: small value to avoid division by zero
        
    Returns:
        numpy array: cost matrix where entry (i,j) is the weight between edges i and j
        rotations: a dictionary from a pair of edge indices to a 3x3 rotation matrix
    """
    num_edges = len(E)
    # Step 1: Initialize cost matrix with zeros (diagonal already zero)
    weight_matrix = np.zeros((num_edges, num_edges))

    # get polyline edge data 
    edge_to_polylines = {}
    
    for polyline_idx, polyline in enumerate(P):
        # Store edge data for reuse
        edge_indices, is_edge_reversed = find_edge_indices_from_polyline(polyline, E)
        for edge_index in edge_indices:
            edge_to_polylines[edge_index] = polyline_idx
    
    print('edge_to_polylines', edge_to_polylines)

    
    # Step 2: Compute weights based on segment distances
    for i in range(num_edges):
        for j in range(i+1, num_edges):  # Only compute upper triangle
            # Get vertex coordinates for both edges
            a1 = V[E[i,0]]
            a2 = V[E[i,1]]
            b1 = V[E[j,0]]
            b2 = V[E[j,1]]

            polyline_i = edge_to_polylines[i]
            polyline_j = edge_to_polylines[j]

            if polyline_i == polyline_j:
                continue 
            else:
                # Compute distance between segments
                distance, _, _ = segment_to_segment_distance(a1, a2, b1, b2)
                
                # Set weight as inverse of distance
                weight = 1.0 / (distance + epsilon)
                # maybe a sigmod function
                
                # Matrix is symmetric
                weight_matrix[i,j] = weight
                
                weight_matrix[j,i] = weight
    
    # Step 3: Add extra weights for neighboring edges in polylines
    rotations = {}
    for polyline in P:
        # For each edge pair in the polyline
        for i in range(len(polyline)-2):
            v1, v2, v3 = polyline[i], polyline[i+1], polyline[i+2]
            
            # Find indices in E for edge1 (v1-v2) and edge2 (v2-v3)
            # Check both directions for each edge
            edge1_mask = ((E[:,0] == v1) & (E[:,1] == v2)) | ((E[:,0] == v2) & (E[:,1] == v1))
            edge2_mask = ((E[:,0] == v2) & (E[:,1] == v3)) | ((E[:,0] == v3) & (E[:,1] == v2))
            
            edge1_idx = np.where(edge1_mask)[0][0]
            edge2_idx = np.where(edge2_mask)[0][0]
            
            # # Add neighbor weight to both positions
            # weight_matrix[edge1_idx, edge2_idx] += neighbor_weight
            # weight_matrix[edge2_idx, edge1_idx] += neighbor_weight
            
            # Create a parallel transport matrix from edge1 to edge2
            rotations[(edge1_idx, edge2_idx)] = rotation_matrix_from( V[v2] - V[v1], V[v3] - V[v2] )
    
    return weight_matrix, rotations

# pairwise: A sequence of triplets ( edge index 1, edge index 2, weight ) 
# such that the difference in normals between "edge index 1" and "edge index 2" should be penalized with the given weight
# len(pairwise) <= 3 * len(edges)
# each edge will get at most 3 pair
# some may duplicate. so do not use duplicates
def extract_pairwise_weight(weight_matrix, P, unconstrained_polylines, edge_constraints):
    """
    Extract the n highest pairwise edge weights for each edge from the weight matrix,
    do not chose the edge from the same polyline.
    avoiding duplicates and ensuring each edge pair appears only once.
    
    Args:
        weight_matrix (np.ndarray): NxN array where entry (i,j) is the weight between edges i and j
        P : array-like, shape (K,)
            Polyline definitions, where each element is a list/array of vertex indices
            forming a polyline. K is number of polylines.
        unconstrained_polylines: polyline who doesn't have any edge normals set 
        edge_constraints : the edge normal constraints get from convex hull, a list [(index, normal)]
    
    Returns:
        list of tuples: [(edge_idx1, edge_idx2, weight), ...] sorted by weight in descending order,
        representing edges that should have similar normals. Each pair appears only once with
        edge_idx1 < edge_idx2 to avoid duplicates.
    """
    # Create set for unique pairwise penalties
    pairwise = []

    # get polyline edge data 
    # also for the same polyline, for the neigboring edges, just give them 1
    edge_to_polylines = {}
    
    for polyline_idx, polyline in enumerate(P):
        # Store edge data for reuse
        edge_indices, is_edge_reversed = find_edge_indices_from_polyline(polyline, E)
        for edge_index in edge_indices:
            edge_to_polylines[edge_index] = polyline_idx
        
        for i in range( len(edge_indices)-2):
            e0 = edge_indices[i]
            e1 = edge_indices[i+1]
            e2 = edge_indices[i+2]
            pairwise.append((e0, e1, 10))
            pairwise.append((e1, e2, 10))
        if len(edge_indices) == 2:
            pairwise.append((edge_indices[0], edge_indices[1], 10))
    
    edge_constraints_dict = {}

    for edge_index, edge_normal in edge_constraints:
        edge_constraints_dict[edge_index] = edge_normal

    # Get matrix size
    matrix_size = len(weight_matrix)
    polyline_weight_pair_cnt = 0

    # same polyline, must be zero in the weights matrix

    print('focus on i',i )
    # Process each edge
    for i in range(matrix_size):
        if i not in edge_constraints_dict and edge_to_polylines[i] in unconstrained_polylines:

            # Get weights for current edge (excluding self-connection)
            weights = weight_matrix[i].copy()
            weights[i] = 0  # Zero out self-connection
            
            # Get indices of all non-zero weights
            non_zero_indices = np.where(weights > 0)[0]
            # Sort these indices by their weights in descending order
            sorted_indices = non_zero_indices[np.argsort(-weights[non_zero_indices])]
            sorted_weights = weights[sorted_indices]

            print('i, sorted_indices, sorted_weights', i, sorted_indices, sorted_weights)


            # Add the pairs to result, ensuring i < j to avoid duplicates
            for j in sorted_indices[:3]:
                # edge_pair = tuple(sorted([i, j]))  # Sort indices to ensure consistent ordering
                pairwise.append((i, j, weights[j]))
             
    # then for the same polyline, add 

    # Convert set to sorted list
    pairwise_list = sorted(list(pairwise), key=lambda x: x[2], reverse=True)
    
    print('pairwise_list', pairwise_list)
    return pairwise_list



def find_edge_indices_from_polyline(polyline, E):
    '''
    Find edge indices and their orientations from a polyline in edge list E.
    
    Given:
        polyline: list of vertex indices [v0, v1, v2, ...]
        E: (n,2) array of edge vertex pairs [(i0, i1), ...]
    
    Return:
        edge_indices: list of indices in E that correspond to polyline edges
        edge_reversed: list of booleans, True if edge is stored in reverse in E
    '''
    edge_indices = []
    edge_reversed = []
    
    # For each consecutive pair in polyline
    for i in range(len(polyline) - 1):
        v1, v2 = polyline[i], polyline[i + 1]
        
        # Look for edge (v1,v2) or (v2,v1) in E
        forward_mask = (E[:, 0] == v1) & (E[:, 1] == v2)
        backward_mask = (E[:, 0] == v2) & (E[:, 1] == v1)
        
        # Find the edge index
        if np.any(forward_mask):
            # Edge found in forward orientation
            edge_idx = np.where(forward_mask)[0][0]
            edge_indices.append(edge_idx)
            edge_reversed.append(False)
        elif np.any(backward_mask):
            # Edge found in reverse orientation
            edge_idx = np.where(backward_mask)[0][0]
            edge_indices.append(edge_idx)
            edge_reversed.append(True)
        else:
            raise ValueError(f"Edge ({v1},{v2}) not found in edge list")
    
    return edge_indices, edge_reversed


def create_frames_for_each_polyline(V, E, P):
    '''
    Creates local coordinate frames (U, V) for each edge in a network of polylines.
    
    Args:
        V (list/array): Vertex coordinates, where V[i] gives the 3D position of vertex i
        E (list of tuple): Edge definitions, where each edge is (i0, i1) vertex indices
        P (list of list): Polyline definitions, where each polyline is a list of vertex indices
                         that form a continuous curve
    
    Returns:
        Us (list): List of U vectors for each edge, where Us[i] corresponds to E[i]
                  U vectors represent the primary direction of the local frame
        Vs (list): List of V vectors for each edge, where Vs[i] corresponds to E[i]
                  V vectors represent the secondary direction of the local frame
    '''
    Us = [None] * len(E)
    Vs = [None] * len(E)

    for polyline in P:
        points = [V[point_index] for point_index in polyline]
        polyline_u, polyline_v = compute_parallel_transport_frames( points )
        edge_indices, edge_reversed = find_edge_indices_from_polyline(polyline, E)
   
        # Negate vectors where edge_reversed is True using list comprehension
        Us_poly = [-u if rev else u for u, rev in zip(polyline_u, edge_reversed)]
        Vs_poly = [-v if rev else v for v, rev in zip(polyline_v, edge_reversed)]
        
        # Assign to corresponding indices
        for idx, (u, v) in enumerate(zip(Us_poly, Vs_poly)):
            Us[edge_indices[idx]] = u
            Vs[edge_indices[idx]] = v
    

    return Us, Vs 

def compute_edge_tangent(V, edge):
    # Compute normalized tangent vector for an edge
    e0, e1 = edge 
    tangent = V[e1] - V[e0]
    assert np.linalg.norm(tangent) != 0
    return tangent / np.linalg.norm(tangent)
    
def edge_distance_matrix(V, E):
    '''
    Compute a matrix of minimum distances between all pairs of edges in a mesh using 
    segment-to-segment distance calculations.
    '''
    n_edges = len(E)
    distances = np.zeros((n_edges, n_edges))
    
    for i, edge1 in enumerate(E):
        # Get vertices of first edge
        e1_v0, e1_v1 = V[edge1[0]], V[edge1[1]]
        
        for j in range(i + 1):  # Only compute lower triangle due to symmetry
            edge2 = E[j]
            e2_v0, e2_v1 = V[edge2[0]], V[edge2[1]]
            
            # Compute distance between segments, discarding closest points
            dist, _, _ = segment_to_segment_distance(e1_v0, e1_v1, e2_v0, e2_v1)
            
            # Store distance in matrix (symmetric)
            distances[i, j] = dist
            if i != j:
                distances[j, i] = dist
    
    return distances

def find_most_perpendicular_edge_normal_on_polyline(V, E, polyline_edge_data, edge_to_normal_map):
    '''
    From the edge_to_normal_map, find the most perpendicular edge normal on the polyline.
    Notice depend on the edge_to_normal_map, this may only find a subset of the polylines
    that have most perpendicular edge normal.


    Parameters:
    -----------
    V : array-like, shape (N, 3)
        Vertex coordinates in 3D space, where N is number of vertices.

    E : array-like, shape (M, 2)
        Edge connectivity, where each row contains indices (i,j) representing 
        an edge between vertices V[i] and V[j]. M is number of edges.

    polyline_edge_data : dictionary, {polyline_idx: (edge_indices, is_edge_reversed)}
                         the polyline edge_indices and whether the edge is reversed

    edge_to_normal_map : dictionary , {edge_idx, normal}

    Return:
        polyline_to_best_normal: 
            Store the most perpendicular normal for polyline
            Format: {polyline_idx: (position_in_polyline, normal_vector)}
    '''

    # Store the most perpendicular normal for each polyline
    # Format: {polyline_idx: (position_in_polyline, normal_vector)}
    polyline_to_best_normal = {}

    for polyline_idx, (edge_indices, is_edge_reversed) in polyline_edge_data.items():
        
        # Track the best normal found for this polyline
        best_normal_vector = None
        smallest_dot_product = 1  # cos(0°) = 1, largest possible dot product
        best_edge_normal_pair = None
        
        # print(f"Polyline {polyline_idx} edges:", edge_indices, is_edge_reversed)
        
        # Check each edge in polyline for normal constraints
        for pos_in_polyline, edge_idx in enumerate(edge_indices):
            if edge_idx in edge_to_normal_map:
                current_normal = edge_to_normal_map[edge_idx]
                edge_direction = compute_edge_tangent(V, E[edge_idx])
                
                # Smaller dot product means more perpendicular vectors
                perpendicularity = np.abs(np.dot(edge_direction, current_normal))
                
                # Update if this is the most perpendicular normal so far
                if best_normal_vector is None or perpendicularity < smallest_dot_product:
                    best_normal_vector = current_normal
                    smallest_dot_product = perpendicularity
                    
                    # Handle edge orientation for the normal vector
                    # the orientation should not matter here ?
                    # if is_edge_reversed[pos_in_polyline]:
                    #     best_edge_normal_pair = (pos_in_polyline, -best_normal_vector)
                    # else:
                    #     best_edge_normal_pair = (pos_in_polyline, best_normal_vector)
                    
                    best_edge_normal_pair = (pos_in_polyline, best_normal_vector)
        
        # Store the best normal if one was found
        if best_normal_vector is not None:
            polyline_to_best_normal[polyline_idx] = best_edge_normal_pair
    

    return polyline_to_best_normal


def parallel_transport_with_best_normal_on_polyline(V, P, polyline_edge_data, polyline_to_best_normal):
    '''
    From the best normal of the polyline, parallel transport it on both direction of the polyline.
    Notice depend on the polyline_to_best_normal, this will transport on a subset of polyline.
    Means not all polylines will get normals after this.

    
    Parameters:
    -----------
    V : array-like, shape (N, 3)
        Vertex coordinates in 3D space, where N is number of vertices.
    
    P : array-like, shape (K,)
        Polyline definitions, where each element is a list/array of vertex indices
        forming a polyline. K is number of polylines.
    
    polyline_edge_data : dictionary, {polyline_idx: (edge_indices, is_edge_reversed)}
                       : the polyline edge_indices and whether the edge is reversed

    polyline_to_best_normal: dictionary,{polyline_idx: (position_in_polyline, normal_vector)}
                             Store the most perpendicular normal for polyline

    Return :
        polyline_normals: dictionary, {polyline_index: [normals]} 
                          polyline_index as key and a list of normals corresponding to each edge in the polyline

    '''
    polyline_normals = {}
    for polyline_idx, polyline in enumerate(P):
        if polyline_idx in polyline_to_best_normal:
            # Reuse stored edge data instead of recomputing
            edge_indices, _ = polyline_edge_data[polyline_idx]
            
            # Get polyline points for parallel transport
            polyline_points = [V[index] for index in polyline]
            polyline_best_normal_constraint = polyline_to_best_normal[polyline_idx]
            
            # Compute parallel transport
            normal_vectors = parallel_transport_bi_direction(polyline_points, polyline_best_normal_constraint)
            polyline_normals[polyline_idx] = normal_vectors
    
    return polyline_normals


def unconstrained_polyline_borrow_nearby_edges_normals(V, E, polyline_edge_data, unconstrained_polylines, edge_to_normal_map, distances):
    '''
    For the unconstrained polyline, which they don't get any normal constraint from convex hull.
    I want to borrow normals from the nearby edges. Which is nearby edge normal from convex hull.

    If there's no normal found, use random value ? 


    Parameters:
    -----------
    V : array-like, shape (N, 3)
        Vertex coordinates in 3D space, where N is number of vertices.

    E : array-like, shape (M, 2)
        Edge connectivity, where each row contains indices (i,j) representing 
        an edge between vertices V[i] and V[j]. M is number of edges.

    polyline_edge_data : dictionary, {polyline_idx: (edge_indices, is_edge_reversed)}
                       : the polyline edge_indices and whether the edge is reversed

    unconstrained_polylines : set, contain the index of polyline which doesnot have any normal constraints

    edge_to_normal_map : edge_idx, normal

    distances : a matrix of minimum distances between all pairs of edges

    Return:
        edge_to_normal_map : 
    '''

    ## 
    unconstrained_edge_to_normal_map = {}
    unconstrained_polylines_candidate_normals = defaultdict(list)

    for polyline_idx in unconstrained_polylines:
        edge_indices, is_edge_reversed = polyline_edge_data[polyline_idx]


        for e0 in edge_indices:
            # most nearby edge to e0
            sorted_nearby_edges = sorted([(i, distances[e0, i]) for i in range(len(distances))], key=lambda x: x[1])

            for e1, e0_e1_distance in sorted_nearby_edges:
                # print(e1)
                # print(e0, e1)
                if e1 in edge_to_normal_map:
                    # get the n1 from the most nearby 
                    n1 = edge_to_normal_map[e1]
                    # compute e0 tangent
                    e0_tangent = compute_edge_tangent(V, E[e0])
                    
                    perpendicularity = np.clip(1.0 - np.abs(np.dot(e0_tangent, n1)), 0, 1)
                    if perpendicularity > 1e-12:
                        # a tuple of distance from e0 - e1, perpendicularity, e0, n1
                        unconstrained_polylines_candidate_normals[polyline_idx].append((e0_e1_distance, perpendicularity, e0, n1))
                        break
    
    # print('unconstrained_polylines_candidate_normals', unconstrained_polylines_candidate_normals)
    # sort by distance, then perpendicularity

    for polyline_idx in unconstrained_polylines_candidate_normals:
        distance_normal_list = unconstrained_polylines_candidate_normals[polyline_idx]
        # sort and then only get the top 3 that most near and perpendicular 
        # sort by distance, then perpendicular 
        # or sort by perpendicular and then distance 
        # sorted_distance_normal_list = sorted(distance_normal_list, key=lambda x: (x[0], x[1]))[:3]
        sorted_distance_normal_list = sorted(distance_normal_list, key=lambda x: (x[1], x[0]))[:3]
        for distance_normal_item in sorted_distance_normal_list:
            _, _, e0, n1 = distance_normal_item
            unconstrained_edge_to_normal_map[e0] = n1


    
    unconstrained_polylines_best_normal = find_most_perpendicular_edge_normal_on_polyline(V, E, polyline_edge_data, unconstrained_edge_to_normal_map)



    return unconstrained_polylines_best_normal

def polyline_normal_to_edge_normal(polyline_normals, polyline_edge_data):
    '''
    Convert polyline normals to edge normals using polyline edge data.
    
    Parameters:
    -----------
    polyline_normals : dict
        Dictionary mapping polyline indices to normal vectors {edge_idx: normal_vector}
    
    polyline_edge_data : dict
        Dictionary mapping polyline indices to tuples of (edge_indices, is_edge_reversed)
        where edge_indices are the edges comprising the polyline and
        is_edge_reversed indicates if each edge's orientation is reversed
    
    Returns:
    --------
    edge_normals : dict
        Dictionary mapping edge indices to normal vectors {edge_idx: normal_vector}
    '''
    
    edge_normals = {}
    
    for polyline_idx, normal_vectors in polyline_normals.items():
        # Get the edge indices and orientation flags for this polyline
        edge_indices, is_edge_reversed = polyline_edge_data[polyline_idx]
        
        # Map each normal vector to its corresponding edge
        for i, edge_idx in enumerate(edge_indices):
            # Note: Normal orientation handling is currently disabled
            # The commented line below would handle edge orientation reversal
            # normal = normal_vectors[i] if not is_edge_reversed[i] else -normal_vectors[i]
            
            edge_normals[edge_idx] = normal_vectors[i]
            
    return edge_normals


def estimate_initial_normals(V, E, P, edge_normal_constraints):
    '''
    Estimates initial normal angles (thetas) for polylines using parallel transport.
    
    Parameters:
    -----------
    V : array-like, shape (N, 3)
        Vertex coordinates in 3D space, where N is number of vertices.
    
    E : array-like, shape (M, 2)
        Edge connectivity, where each row contains indices (i,j) representing 
        an edge between vertices V[i] and V[j]. M is number of edges.
    
    P : array-like, shape (K,)
        Polyline definitions, where each element is a list/array of vertex indices
        forming a polyline. K is number of polylines.
    
    edge_normal_constraints :  list of tuple
        Each tuple contains (edge_idx, normal) where:
        - edge_idx (int): Index of the edge in E
        - normal (np.ndarray): (3,) normalized direction vector
        
    Returns:
    --------
    thetas : array-like, shape (M,)
        Estimated normal angles (in radians) for each edge in E.
        The angles define the normal vector orientation for each edge.
    
    Algorithm:
    ----------
    1. For polylines with existing normal constraints:
       - Find the normal vector most perpendicular to the polyline
       - Parallel transport this normal along the polyline to get initial angles
       
    2. For polylines without normal constraints:
       - Locate the nearest polyline that has normal constraints
       - As long as the normal constraint are not parallel to the polyline
       - Try to locate a good one and parallel transport those normals to initialize angles
       
    3. Convert all parallel transported normals to angles (thetas)
    
    Notes:
    ------
    - Angles are returned in radians
    - Uses parallel transport to maintain smooth normal vector field
    - For unconstrained polylines, proximity is determined by shortest distance
      between polyline midpoints
    '''
    # Convert constraints list to dictionary for O(1) lookup time
    edge_to_normal_map = {edge_idx: normal for edge_idx, normal in edge_normal_constraints}
    
    # Store polyline edge data to avoid recomputing
    # Format: {polyline_idx: (edge_indices, is_edge_reversed)}
    polyline_edge_data = {}
    
    for polyline_idx, polyline in enumerate(P):
        # Store edge data for reuse
        edge_indices, is_edge_reversed = find_edge_indices_from_polyline(polyline, E)
        polyline_edge_data[polyline_idx] = (edge_indices, is_edge_reversed)
    
    # First pass: Find most perpendicular normal for each polyline
    # Store the most perpendicular normal for each polyline
    # Format: {polyline_idx: (position_in_polyline, normal_vector)}
    constrained_polyline_to_best_normal =  find_most_perpendicular_edge_normal_on_polyline(V, E, polyline_edge_data, edge_to_normal_map)

    if save_debug_gltf:
        export_sketch_dict_normal_gltf(V, E, P, constrained_polyline_to_best_normal, 'debug_normals_gltf/initial_most_perpendicular/' + curve_name + '.gltf')
    
    # print("Polylines with their most perpendicular normals:", polyline_to_best_normal)
    print("Polylines with their most perpendicular normals:", len(constrained_polyline_to_best_normal))

    if show_plot is True:
        plot_polyline_best_constraints(V, E, P, constrained_polyline_to_best_normal, str='most perpendicular on polyline')

    # Second pass: Compute parallel transport for each polyline with a normal
    polyline_normals = parallel_transport_with_best_normal_on_polyline(V, P , polyline_edge_data, constrained_polyline_to_best_normal)

    # print('constrained_polyline_normals', constrained_polyline_normals)
    print('len(polyline_normals)', len(polyline_normals))

    if show_plot is True:
        plot_polyline_normals(V, E, P, polyline_normals, str = 'parallel transport most perpendicular normal')
        # trusted_edge_normals = polyline_normal_to_edge_normal(polyline_normals, polyline_edge_data)
        # # print('unconstrained_edge_normals', unconstrained_edge_normals)
        # trusted_edge_normals_list = edge_normal_dict_to_ndarray(trusted_edge_normals, len(E))
        # trusted_normalized = np.asarray(trusted_edge_normals_list)
        # # trusted_normalized = trusted_normalized / np.linalg.norm(trusted_normalized, axis=1)[:, np.newaxis]
        # trusted_normalized = np.divide(trusted_normalized, np.linalg.norm(trusted_normalized, axis=1)[:, np.newaxis], out=np.zeros_like(trusted_normalized), where=np.linalg.norm(trusted_normalized, axis=1)[:, np.newaxis] > 0)
        # write_normal_data(V, E, trusted_normalized, 'debug_normals/' + curve_name + '.normal')
    
    
    # Convert polyline normals to edge normals
    # Format: {edge_idx: normal_vector}
    edge_normals = polyline_normal_to_edge_normal(polyline_normals, polyline_edge_data)

    print('len(constrained_edge_normals)', len(edge_normals))

    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, polylines, edge_normal_dict_to_ndarray(edge_normals, len(E)), 'debug_normals_gltf/initial_parallel_transport/' + curve_name + '.gltf' )
    
  
    ## Third pass:
    # now let me do this, for the polyline which does not have normals
    # use the edge from the polyline, then use the edge_distance_matrix to find from the 
    # edge < epsilon edge_normals as constraints to propogate on those who does not have normal
    # polylines

    # Identify polylines without normals
    edge_distances = edge_distance_matrix(V, E)

    # print('polyline_normals.keys()', constrained_polyline_normals.keys())


    unconstrained_polylines = set(range(len(P))) - set(polyline_normals.keys())

    print("Polylines without normals:", unconstrained_polylines)

    unconstrained_polylines_best_normal = unconstrained_polyline_borrow_nearby_edges_normals(V, E, polyline_edge_data, unconstrained_polylines, edge_normals, edge_distances)
    # print('unconstrained_polylines_best_normal', unconstrained_polylines_best_normal)

    if show_plot is True:
        plot_polyline_best_constraints(V, E, P, unconstrained_polylines_best_normal, scale=0.09, str = 'borrow nearby edge normal')
    if save_debug_gltf:
        export_sketch_dict_normal_gltf(V, E, P, unconstrained_polylines_best_normal, 'debug_normals_gltf/borrowed_normal/' + curve_name + '.gltf')
    
    unconstrained_polyline_normals = parallel_transport_with_best_normal_on_polyline(V, P, polyline_edge_data, unconstrained_polylines_best_normal)
    

    if show_plot is True:
        plot_polyline_normals(V, E, P, unconstrained_polyline_normals,scale=0.09,  str = 'parallel transport of the polyline with borrowed normal')
    if save_debug_gltf:
        unconstrained_edge_normals = polyline_normal_to_edge_normal(unconstrained_polyline_normals, polyline_edge_data)
        # print('unconstrained_edge_normals', unconstrained_edge_normals)
        unconstrained_edge_normals_list = edge_normal_dict_to_ndarray(unconstrained_edge_normals, len(E))
        # print('unconstrained_edge_normals_list', unconstrained_edge_normals_list)
        export_sketch_normal_gltf(V, E, polylines, unconstrained_edge_normals_list, 'debug_normals_gltf/borrowed_parallel_transport/' + curve_name + '.gltf')


    polyline_normals = {**polyline_normals, **unconstrained_polyline_normals}

    edge_normals = polyline_normal_to_edge_normal(polyline_normals, polyline_edge_data)

    if save_debug_gltf:
        info = (
            f' total_edges: {len(E)}\n'
            f' has_normal_edges: {len(edge_normal_constraints)}\n'
            f' total_polylines: {len(P)}\n'
            f' has_normal_polylines: {len(constrained_polyline_to_best_normal)}\n'
            f' borrowed_normal_polylines: {len(unconstrained_polyline_normals)}'
        )
        
        write_string_to_file(info, 'debug_normals_gltf/normal_info/' + curve_name + '.txt')


    # print("unconstrained_polyline_normals.keys()", unconstrained_polyline_normals.keys())
    # print("unconstrained_polylines" , set(unconstrained_polylines) - set(unconstrained_polyline_normals))
    # print("len(polyline_normals.keys())", len(polyline_normals.keys()))
    # print("len(P)", len(P))
    assert len( polyline_normals.keys() ) == len(P), "Some polylines do not have normals"

    edge_normal_list = [(i,edge_normals[i]) for i in range(len(E)) if i in edge_normals]    

    if show_plot is True:
        plot_edge_constraints(V, E, P, edge_normal_list, scale= 0.03, str= "initial estimate")

    return edge_normal_list, unconstrained_polylines


def estimate_initial_thetas(Us, Vs, estimated_normals):
    '''
    Convert normal vectors to theta values using frame vectors as basis.
    
    Parameters:
    -----------
    Us : array-like
        First frame vector for each edge
    Vs : array-like
        Second frame vector for each edge
    estimated_normals : list of tuple
        Each tuple contains (edge_idx, normal_vector)
        
    Returns:
    --------
    thetas : array-like
        Angles (in radians) for each edge in the frame coordinate system
    '''
    n_edges = len(Us)
    thetas = np.zeros(n_edges)
    
    # Convert normal vector list to dictionary for easy lookup
    normal_dict = dict(estimated_normals)
    
    # Compute theta for each edge that has a normal
    for edge_idx, normal in normal_dict.items():
        # Project normal onto frame vectors
        cos_theta = np.dot(normal, Us[edge_idx])
        sin_theta = np.dot(normal, Vs[edge_idx])
        
        # Compute angle using atan2 for correct quadrant
        thetas[edge_idx] = np.arctan2(sin_theta, cos_theta)
    
    return thetas

def propagate_edge_normals_along_polylines(V, E, P, edge_normal_constraints):
    """
    Propagates normal vectors along polylines using parallel transport,
    starting from the most perpendicular existing normal constraint for each polyline.
    It will only propagate edge which has 

    Parameters:
    -----------
    V : array-like
        Vertex positions in 3D space
    E : array-like
        Edge definitions as pairs of vertex indices
    P : list of lists
        Polylines defined as sequences of vertex indices
    edge_normal_constraints : list of tuples
        List of (edge_index, normal_vector) pairs specifying known normal constraints

    Returns:
    --------
    list of tuples
        List of (edge_index, normal_vector) pairs for all edges that received propagated normals

    Algorithm:
    ----------
    1. For each polyline:
        - Identifies the most perpendicular normal vector from existing constraints
        - Uses this as a starting point for parallel transport
    2. Performs bidirectional parallel transport along each polyline
    3. Maps the resulting normals back to individual edges

    Notes:
    ------
    - Edge orientation is preserved during normal propagation
    - Polylines without any normal constraints are skipped
    - Normal vectors are assumed to be unit vectors
    """    
     # Convert constraints list to dictionary for O(1) lookup time
    edge_to_normal_map = {edge_idx: normal for edge_idx, normal in edge_normal_constraints}
    
    # Store the most perpendicular normal for each polyline
    # Format: {polyline_idx: (position_in_polyline, normal_vector)}
    polyline_to_best_normal = {}
    
    # Store polyline edge data to avoid recomputing
    # Format: {polyline_idx: (edge_indices, is_edge_reversed)}
    polyline_edge_data = {}
    
  # First pass: Find most perpendicular normal for each polyline
    for polyline_idx, polyline in enumerate(P):
        # Store edge data for reuse
        edge_indices, is_edge_reversed = find_edge_indices_from_polyline(polyline, E)
        polyline_edge_data[polyline_idx] = (edge_indices, is_edge_reversed)
        
        # Track the best normal found for this polyline
        best_normal_vector = None
        smallest_dot_product = 1  # cos(0°) = 1, largest possible dot product
        best_edge_normal_pair = None
        
        print(f"Polyline {polyline_idx} edges:", edge_indices, is_edge_reversed)
        
        # Check each edge in polyline for normal constraints
        for pos_in_polyline, edge_idx in enumerate(edge_indices):
            if edge_idx in edge_to_normal_map:
                current_normal = edge_to_normal_map[edge_idx]
                edge_direction = compute_edge_tangent(V, E[edge_idx])
                
                # Smaller dot product means more perpendicular vectors
                perpendicularity = np.abs(np.dot(edge_direction, current_normal))
                
                # Update if this is the most perpendicular normal so far
                if best_normal_vector is None or perpendicularity < smallest_dot_product:
                    best_normal_vector = current_normal
                    smallest_dot_product = perpendicularity
                    
                    # Handle edge orientation for the normal vector
                    # the orientation should not matter here ?
                    # if is_edge_reversed[pos_in_polyline]:
                    #     best_edge_normal_pair = (pos_in_polyline, -best_normal_vector)
                    # else:
                    #     best_edge_normal_pair = (pos_in_polyline, best_normal_vector)
                    
                    best_edge_normal_pair = (pos_in_polyline, best_normal_vector)
        
        # Store the best normal if one was found
        if best_normal_vector is not None:
            polyline_to_best_normal[polyline_idx] = best_edge_normal_pair
    
    print("Polylines with their most perpendicular normals:", polyline_to_best_normal)
    

    # Second pass: Compute parallel transport for each polyline with a normal
    polyline_normals = {}
    for polyline_idx, polyline in enumerate(P):
        if polyline_idx in polyline_to_best_normal:
            # Reuse stored edge data instead of recomputing
            edge_indices, _ = polyline_edge_data[polyline_idx]
            
            # Get polyline points for parallel transport
            polyline_points = [V[index] for index in polyline]
            polyline_best_normal_constraint = polyline_to_best_normal[polyline_idx]
            
            # Compute parallel transport
            normal_vectors = parallel_transport_bi_direction(polyline_points, polyline_best_normal_constraint)
            polyline_normals[polyline_idx] = normal_vectors
            
    print('polyline_normals', polyline_normals)
    
    
    # Convert polyline normals to edge normals
    # Format: {edge_idx: normal_vector}
    edge_normals = {}
    for polyline_idx, normal_vectors in polyline_normals.items():
        # Get the edge indices for this polyline
        edge_indices, is_edge_reversed = polyline_edge_data[polyline_idx]
        
        # Map each normal vector to its corresponding edge
        for i, (edge_idx, is_reversed) in enumerate(zip(edge_indices, is_edge_reversed)):
            # Reverse normal if edge orientation is reversed
            # the orientation should also not matter here ?
            # normal = normal_vectors[i] if not is_reversed else normal_vectors[i]
            edge_normals[edge_idx] = normal_vectors[i]

    print('edge_normals', edge_normals)
    
    
    edge_normal_list = [(i,edge_normals[i]) for i in range(len(E)) if i in edge_normals]
    print('edge_normal_list', edge_normal_list)

    plot_edge_constraints(V, E, P, edge_normal_list)
    return edge_normal_list

def normal_for_edge( theta, U, V ): return np.cos( theta ) * U + np.sin( theta ) * V

def recover_normal_from_thetas(thetas, Us, Vs):
    '''
    Compute normal vectors for all edges given their angles and frame vectors.
    '''
    return [
        (i, normal_for_edge(theta, Us[i], Vs[i]))
        for i, theta in enumerate(thetas)
    ]

def edge_normal_dict_to_ndarray(edge_normals, num_edges):
    """
    Convert edge normals from dictionary format to numpy array.
    
    Args:
        edge_normals (dict): Dictionary with edge indices as keys and normal vectors as values
        num_edges (int): Total number of edges
        
    Returns:
        numpy.ndarray: Array of shape (num_edges, 3) containing edge normal vectors
    """
    if not isinstance(edge_normals, dict):
        raise TypeError("edge_normals must be a dictionary")
        
    normals = np.zeros((num_edges, 3))
    for edge_index, normal in edge_normals.items():
        normals[edge_index] = normal
        
    return normals

def edge_normal_tuple_to_ndarray(edge_normals, num_edges):
    '''
    Convert edge normal tuples format to numpy array
    '''
    normals = np.zeros((num_edges, 3))
    for edge_index, normal in edge_normals:
        normals[edge_index] = normal
        
    return normals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('normal_file', nargs='?', help='The curve sketch with optimized normal information.')
    parser.add_argument('gltf_file', nargs='?', help='The normal gltf file to save.')
    parser.add_argument('--show_plot', type=str, choices=['true', 'false'], default='true',
                   help='Whether to show the visualization plot (default: true)')    
    parser.add_argument('--save_debug_gltf', type=str, choices=['true', 'false'], default='true',
                   help='Save the gltf files for debug (default: true)')    
    
    args = parser.parse_args()

    curve_file = args.curve_file
    normal_file = args.normal_file
    gltf_file = args.gltf_file
    show_plot = args.show_plot.lower() == 'true'
    save_debug_gltf = args.save_debug_gltf.lower() == 'true'

    print('show_plot', show_plot)
    print('save_debug_gltf',save_debug_gltf)
    
    if curve_file is None:
        curve_file = 'files_starter/simple_objs/onshape_simple_mouse.obj'


    curve_name = Path(curve_file).stem

    '''
    - V: nx3 array of vertex coordinates
    - E: mx2 array of edge vertex index pairs with no duplicates
    - P: list of arrays containing vertex indices for each polyline
    '''
    V, E, P = load_sketch_polyline_data(curve_file)
    polylines = [[V[i] for i in line] for line in P]

    print('len(V)', len(V))
    print('len(E)', len(E))
    print('len(P)', len(P))


    # only need points and polyline indices to draw
    # same polyline, same color
    if show_plot is True:
        plot_sketch_data(V, P)

    '''
    normals: convex hull point normals
    edge_normals: assign and filter the point normals on edges
    plot edge normals
    '''
    edge_constraints = get_sketch_edge_constraints(V, E)
    # print('edge_constraints', edge_constraints)
    print('len(edge_constraints)', len(edge_constraints))
    if show_plot is True:
        plot_edge_constraints(V, E, P, edge_constraints, scale=0.03, str = 'edge constraints from convex hull')
        write_normal_data(V, E, edge_normal_tuple_to_ndarray(edge_constraints, len(E)) , 'debug_normals_gltf/edge_normals/' + curve_name + '.normal')

    # edge_filtered_normals = np.zeros((len(E), 3))
    # for edge_index, normal in edge_constraints:
    #     edge_filtered_normals[edge_index] = normal



    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, polylines, edge_normal_tuple_to_ndarray(edge_constraints, len(E)),'debug_normals_gltf/edge_normals/' + curve_name + '.gltf' )
        

    '''
    estimate initial constraints
    '''

    estimate_normals, unconstrained_polylines = estimate_initial_normals(V, E, P, edge_constraints)

    estimate_normals_list = [normal for _,normal in estimate_normals]
    estimate_normals_list = estimate_normals_list / np.linalg.norm(estimate_normals_list, axis=1)[:, np.newaxis]
    # plot_edge_constraints(V, E, P, estimate_normals)
    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, polylines, edge_normal_tuple_to_ndarray(estimate_normals, len(E)), 'debug_normals_gltf/initial_estimate/' + curve_name + '.gltf')
        write_normal_data(V, E, estimate_normals_list , 'debug_normals_gltf/initial_estimate/' + curve_name + '.normal')




    weight_matrix, rotations = create_edge_weight_matrix(V, E, P)
    pairwise = extract_pairwise_weight(weight_matrix, P, unconstrained_polylines, edge_constraints)
    # print('len(pairwise)', len(pairwise))



    # '''
    # optimization:

    # I think initial guesses are important. To have better initial guess
    
    # 1. all the polylines have 
    # '''

    Us, Vs = create_frames_for_each_polyline(V, E, P )
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)





    def E_total( thetas, Us, Vs, constraints, pairwise):
        '''
        Given a bag of edge data of the form:
            thetas: An array of N real numbers, one per edge
            tangents: An N-by-3 array of tangent vectors
            Us: An N-by-3 array of vectors spanning the plane normal to each edge (along with Vs)
            Vs: An N-by-3 array of vectors spanning the plane normal to each edge (along with Us)
            constraints: A sequence of pairs ( edge index, desired normal vector ) such that edge "edge index" should have the normal "desired normal vector"
            pairwise: A sequence of triplets ( edge index 1, edge index 2, weight ) such that the difference in normals between "edge index 1" and "edge index 2" should be penalized with the given weight
        Returns:
            The total energy
        '''    
        # Calculate the constraint energy
        E_constraint = 0.0
        for edge_index, desired_normal_vector in constraints:
            n = normal_for_edge( thetas[ edge_index ], Us[ edge_index ], Vs[ edge_index ] )
            E_constraint += (1.0 - np.dot( n, desired_normal_vector ) )**2
        # normalize
        E_constraint /= len( constraints )
        
        E_pairwise = 0.0
        W_pairwise = 0.0
        for e1, e2, weight in pairwise:
            n1 = normal_for_edge( thetas[e1], Us[e1], Vs[e1] )
            n2 = normal_for_edge( thetas[e2], Us[e2], Vs[e2] )
            
            ## Get the rotation matrix if it exists
            if (e1,e2) in rotations: n1 = rotations[(e1,e2)] @ n1
            elif (e2,e1) in rotations: n1 = rotations[(e2,e1)].T @ n1

            E_pairwise += weight * (1.0 - np.dot( n1, n2 ) )**2     
            W_pairwise += weight
        # Normalize by the total weight
        E_pairwise /= W_pairwise
        
        # return 1 * E_constraint +  1e4 * E_pairwise
        return 1e2 * E_constraint +  1 * E_pairwise
        # return E_constraint


    result = opt.minimize( E_total,
        thetas0,  
        args=(Us, Vs, edge_constraints, pairwise),  # Pass additional arguments
        method = 'L-BFGS-B', 
        tol = 0.0000001, 
        options = { 'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 } 
    )

    thetas = result.x

    # print(thetas)

    opt_normals = recover_normal_from_thetas(thetas, Us, Vs)

    if show_plot is True:
        plot_edge_constraints(V, E, P,  opt_normals, scale=0.08, str = "optimize result")

    # # print(opt_normals)

    N = [normal for _,normal in opt_normals]
    N_normalized = N / np.linalg.norm(N, axis=1)[:, np.newaxis]


    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, polylines, N_normalized, 'debug_normals_gltf/final_optimize/' + curve_name + '.gltf')
        write_normal_data(V, E, N_normalized , 'debug_normals_gltf/final_optimize/' + curve_name + '.normal')

    
    if gltf_file:
        export_sketch_normal_gltf(V, E, polylines, N_normalized, gltf_file)




    write_normal_data(V, E, N_normalized , normal_file)








  
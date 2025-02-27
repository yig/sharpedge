import numpy as np

from utility_io import load_sketch_polyline_data, write_normal_data
from utility_plot_viewer import plot_sketch_data, plot_edge_constraints, plot_edge_frames, plot_polyline_best_constraints, plot_polyline_normals
from utility_segment_distance import segment_to_segment_distance

from utility_convex_hull import get_sketch_edge_constraints, export_sketch_normal_gltf
from utility_parallel_transport import compute_parallel_transport_frames
from utility_parallel_transport_bidirection import parallel_transport_bi_direction

import scipy.optimize as opt

from pathlib import Path
from collections import defaultdict

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

def create_edge_weight_matrix(V, E, P, neighbor_weight=10, epsilon=1e5):
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
    """
    num_edges = len(E)
    # Step 1: Initialize cost matrix with zeros (diagonal already zero)
    weight_matrix = np.zeros((num_edges, num_edges))
    
    # Step 2: Compute weights based on segment distances
    for i in range(num_edges):
        for j in range(i+1, num_edges):  # Only compute upper triangle
            # Get vertex coordinates for both edges
            a1 = V[E[i,0]]
            a2 = V[E[i,1]]
            b1 = V[E[j,0]]
            b2 = V[E[j,1]]
            
            # Compute distance between segments
            distance, _, _ = segment_to_segment_distance(a1, a2, b1, b2)
            
            # Set weight as inverse of distance
            weight = 1.0 / (distance + epsilon)
            
            # Matrix is symmetric
            weight_matrix[i,j] = weight
            
            weight_matrix[j,i] = weight
    
    # Step 3: Add extra weights for neighboring edges in polylines
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
            
            # Add neighbor weight to both positions
            weight_matrix[edge1_idx, edge2_idx] += neighbor_weight
            weight_matrix[edge2_idx, edge1_idx] += neighbor_weight
    
    return weight_matrix

# pairwise: A sequence of triplets ( edge index 1, edge index 2, weight ) 
# such that the difference in normals between "edge index 1" and "edge index 2" should be penalized with the given weight
# len(pairwise) <= 3 * len(edges)
# each edge will get at most 3 pair
# some may duplicate. so do not use duplicates
def extract_pairwise_weight(weight_matrix, n=3):
    """
    Extract the n highest pairwise edge weights for each edge from the weight matrix,
    avoiding duplicates and ensuring each edge pair appears only once.
    
    Args:
        weight_matrix (np.ndarray): NxN array where entry (i,j) is the weight between edges i and j
        n (int): number of highest weight pairs to keep for each edge (default: 3)
    
    Returns:
        list of tuples: [(edge_idx1, edge_idx2, weight), ...] sorted by weight in descending order,
        representing edges that should have similar normals. Each pair appears only once with
        edge_idx1 < edge_idx2 to avoid duplicates.
    """
    # Get matrix size
    matrix_size = len(weight_matrix)
    
    # Create set for unique pairwise penalties
    pairwise = set()
    
    # Process each edge
    for i in range(matrix_size):
        # Get weights for current edge (excluding self-connection)
        weights = weight_matrix[i].copy()
        weights[i] = 0  # Zero out self-connection
        
        # Get indices of n highest weights
        top_indices = np.argpartition(weights, -n)[-n:]
        top_indices = top_indices[weights[top_indices] > 0]  # Filter out zero weights
        
        # Sort by weight in descending order
        top_indices = top_indices[np.argsort(weights[top_indices])[::-1]]
        
        # Add the pairs to result, ensuring i < j to avoid duplicates
        for j in top_indices:
            edge_pair = tuple(sorted([i, j]))  # Sort indices to ensure consistent ordering
            if edge_pair[0] < edge_pair[1]:  # Only add if first index is smaller
                pairwise.add((edge_pair[0], edge_pair[1], weights[j]))
    
    # Convert set to sorted list
    pairwise_list = sorted(list(pairwise), key=lambda x: x[2], reverse=True)
    
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
    
    # print("Polylines with their most perpendicular normals:", polyline_to_best_normal)
    print("Polylines with their most perpendicular normals:", len(polyline_to_best_normal))
    plot_polyline_best_constraints(V, E, P, polyline_to_best_normal, str='most perpendicular on polyline')

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
    plot_polyline_normals(V, E, P, polyline_normals, str = 'parallel transport most perpendicular normal')
    
    
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


    export_sketch_normal_gltf(V, E, polylines, edge_normal_dict_to_ndarray(edge_normals, len(E)), 'debug_normals/initial_parallel_transport/' + curve_name + '.gltf' )
    
  
    ## Third pass:
    # now let me do this, for the polyline which does not have normals
    # use the edge from the polyline, then use the edge_distance_matrix to find from the 
    # edge < epsilon edge_normals as constraints to propogate on those who does not have normal
    # polylines

    # Identify polylines without normals
    edge_distances = edge_distance_matrix(V, E)

    print('polyline_normals.keys()', polyline_normals.keys())
    unconstrained_polylines = set(range(len(P))) - set(polyline_normals.keys())
    print("Polylines without normals:", unconstrained_polylines)
    unconstrained_polylines_best_normal = {}

    # Propagate normals to unconstrained polylines
    for polyline_idx in unconstrained_polylines:
        edge_indices, is_edge_reversed = polyline_edge_data[polyline_idx]
        
        # Get first and last edges of the polyline
        start_edge_idx = edge_indices[0]
        end_edge_idx = edge_indices[-1]
        
        # Find closest edges with existing normals for start and end edges
        start_constraints = []  # (edge_idx, normal, distance, perpendicularity)
        end_constraints = []
        
        # Process start edge
        start_distances = edge_distances[start_edge_idx]
        start_edge_direction = compute_edge_tangent(V, E[start_edge_idx])
        
        # Find the closest edge with normal for start edge
        for other_edge_idx, dist in enumerate(start_distances):
            if other_edge_idx in edge_normals:
                normal = edge_normals[other_edge_idx]
                perpendicularity = np.clip(1.0 - np.abs(np.dot(start_edge_direction, normal)), 0, 1)
                # print('perpendicularity', perpendicularity)
                if perpendicularity > 1e-12:
                    start_constraints.append(
                        (other_edge_idx, normal, dist, perpendicularity)
                    )
        
        # Process end edge
        end_distances = edge_distances[end_edge_idx]
        end_edge_direction = compute_edge_tangent(V, E[end_edge_idx])
        
        # Find the closest edge with normal for end edge
        for other_edge_idx, dist in enumerate(end_distances):
            if other_edge_idx in edge_normals:
                normal = edge_normals[other_edge_idx]
                perpendicularity = np.clip(1.0 - np.abs(np.dot(start_edge_direction, normal)), 0, 1)
                if perpendicularity > 1e-12:
                    end_constraints.append(
                        (other_edge_idx, normal, dist, perpendicularity)
                    )
        
        # Sort constraints by distance and perpendicularity
        start_constraints.sort(key=lambda x: (x[2], x[3]))  # Sort by distance, then perpendicularity
        end_constraints.sort(key=lambda x: (x[2], x[3]))

        print('start_constraints', start_constraints)
        print('end_constraints', end_constraints)
        
        if start_constraints or end_constraints:
            # Choose the better constraint between start and end
            best_normal = None
            pos_in_polyline = None
            
            if start_constraints and end_constraints:
                # Compare best start and end constraints
                best_start = start_constraints[0]
                best_end = end_constraints[0]
                
                # Choose the one that's more perpendicular
                if best_start[3] < best_end[3]:  # start is more perpendicular
                    best_normal = best_start[1]
                    pos_in_polyline = 0
                else:
                    best_normal = best_end[1]
                    pos_in_polyline = len(edge_indices) - 1
            elif start_constraints:
                best_normal = start_constraints[0][1]
                pos_in_polyline = 0
            else:  # end_constraints must exist
                best_normal = end_constraints[0][1]
                pos_in_polyline = len(edge_indices) - 1
            
            # Get polyline points for parallel transport
            polyline_best_normal_constraint = (pos_in_polyline, best_normal)
            unconstrained_polylines_best_normal[polyline_idx] = (pos_in_polyline, best_normal)
    
    print('unconstrained_polylines_best_normal', unconstrained_polylines_best_normal)
    plot_polyline_best_constraints(V, E, P, unconstrained_polylines_best_normal, scale=0.08, str = 'borrow nearby edge normal')

    for polyline_idx, polyline_best_normal_constraint in unconstrained_polylines_best_normal.items():
        edge_indices, is_edge_reversed = polyline_edge_data[polyline_idx]
        
        # Get first and last edges of the polyline
        start_edge_idx = edge_indices[0]
        end_edge_idx = edge_indices[-1]

        pos_in_polyline, best_normal = polyline_best_normal_constraint
        # Get polyline points for parallel transport
        polyline_points = [V[index] for index in P[polyline_idx]]


        # Compute parallel transport
        normal_vectors = parallel_transport_bi_direction(
            polyline_points, 
            polyline_best_normal_constraint
        )
        polyline_normals[polyline_idx] = normal_vectors
        
        # Update edge_normals with new normals
        for i, (edge_idx, is_reversed) in enumerate(zip(edge_indices, is_edge_reversed)):
            edge_normals[edge_idx] = normal_vectors[i]
                
    edge_normal_list = [(i,edge_normals[i]) for i in range(len(E)) if i in edge_normals]
    #print('edge_normal_list', edge_normal_list)

    plot_edge_constraints(V, E, P, edge_normal_list)


    return edge_normal_list


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
    import argparse
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('normal_file', nargs='?', help='The curve sketch with optimized normal information.')
    parser.add_argument('gltf_file', nargs='?', help='The gltf normal file to save. If not provided, no gltf will be generated.')
    # parser.add_argument('--plot', type = str, default= 'true', help='Plot the figure or not.')

    args = parser.parse_args()

    curve_file = args.curve_file
    normal_file = args.normal_file
    gltf_file = args.gltf_file
    # plot = args.plot
    
    
    if curve_file is None:
    #     from PyQt6.QtWidgets import QApplication, QFileDialog
    #     app = QApplication([])
    #     curve_file, _ = QFileDialog.getOpenFileName(None, "Open File", 'files_resampled_3d_objs', "obj (*.obj)")
    #     # curve_file = 'files_resampled_3d_objs/author2_sofa.obj'

    # curve_file = 'files_starter/simple_objs/author2_sofa.obj'
    # curve_file = 'pre_processing_uniform/made_objs/single_tetrahedron.obj'

    # curve_file = 'files_starter/simple_objs/flowrep_spherecylinder.obj'
    # curve_file = 'files_starter/simple_objs/flowrep_trebol.obj'
    # curve_file = 'files_starter/simple_objs/onshape_bishop.obj'
        curve_file = 'files_starter/simple_objs/onshape_simple_mouse.obj'
    # curve_file = 'files_starter/simple_objs/onshape_simple_shape.obj'


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
    plot_sketch_data(V, P)

    '''
    normals: convex hull point normals
    edge_normals: assign and filter the point normals on edges
    plot edge normals
    '''
    edge_constraints = get_sketch_edge_constraints(V, E)
    # print('edge_constraints', edge_constraints)
    print('len(edge_constraints)', len(edge_constraints))
    plot_edge_constraints(V, E, P, edge_constraints)

    # edge_filtered_normals = np.zeros((len(E), 3))
    # for edge_index, normal in edge_constraints:
    #     edge_filtered_normals[edge_index] = normal





    export_sketch_normal_gltf(V, E, polylines, edge_normal_tuple_to_ndarray(edge_constraints, len(E)),'debug_normals/edge_normals/' + curve_name + '.gltf' )

    '''
    estimate initial constraints
    '''

    estimate_normals = estimate_initial_normals(V, E, P, edge_constraints)
    # plot_edge_constraints(V, E, P, estimate_normals)
    export_sketch_normal_gltf(V, E, polylines, edge_normal_tuple_to_ndarray(estimate_normals, len(E)), 'debug_normals/initial_estimate/' + curve_name + '.gltf')
    




    weight_matrix = create_edge_weight_matrix(V, E, P)
    pairwise = extract_pairwise_weight(weight_matrix, n = 3)
    print('len(pairwise)', len(pairwise))



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
            # if the edges are adjacent, correct for parallel transport
            # if adjacent_and_ordered( e1, e2 ): n1 = rotation( from = e1, to = e2 ) * n1
            # elif adjacent_and_ordered( e2, e1 ): n2 = rotation( from = e2, to = e1 ) * n2
            E_pairwise += weight * (1.0 - np.dot( n1, n2 ) )**2
            W_pairwise += weight
        # Normalize by the total weight
        E_pairwise /= W_pairwise
        
        return 1e4 * E_constraint + 1 * E_pairwise


    result = opt.minimize( E_total,
        thetas0,  
        args=(Us, Vs, edge_constraints, pairwise),  # Pass additional arguments
        method = 'L-BFGS-B', 
        tol = 0.0000001, 
        options = { 'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 } 
    )

    thetas = result.x

    print(thetas)

    opt_normals = recover_normal_from_thetas(thetas, Us, Vs)


    # # def solve_E_total( points, thetas0 ):
    # #     assert len(thetas0) > 1
    
    # #     # Make an autodiff vector copy of `thetas0`.
    # #     ## How do ensure this is autodiff?
    # #     thetas = jax.numpy.array( thetas0 )
    
    # #     dE_total = jax.grad( E_total, argnums = 1 )
    
    # #     # result = scipy.optimize.minimize( lambda x: E_angle( points, x ), thetas0, jac = lambda x: dE_angle( points, jax.numpy.array( x ) ), options = { 'disp': True } )
    # #     # return result.x
        
    # #     iter: int = 0
    # #     MAX_ITER: int = 100
    # #     STEP_SIZE: float = 0.5
    # #     GRAD_TOL: float = 1e-5
    # #     for iter in range(MAX_ITER):
    # #         e = E_total( points, thetas )
    # #         grad = dE_total( points, thetas )
    # #         grad_norm = np.linalg.norm( grad )
    # #         print( f"Iteration {iter} gradient norm: {grad_norm}" )
    # #         if grad_norm < GRAD_TOL: break
    # #         thetas -= STEP_SIZE * grad
        
    # #     print( "Optimization", ("converged" if iter < MAX_ITER else "diverged"), "after", iter, "iterations." )
        
    # #     # print( thetas - result.x )
        
    # #     return thetas
    
    # # thetas = solve_E_total(V, thetas0)
    # # opt_normals = recover_normal_from_thetas(thetas, Us, Vs)

    plot_edge_constraints(V, E, P,  opt_normals, scale=0.03)

    # # print(opt_normals)

    N = [normal for _,normal in opt_normals]
    N_normalized = N / np.linalg.norm(N, axis=1)[:, np.newaxis]
    # write_normal_data(V, E, N, 'sketch_normal/' + curve_name + '.obj')
    write_normal_data(V, E, N_normalized , normal_file)
    
    export_sketch_normal_gltf(V, E, polylines, N_normalized, 'debug_normals/final_optimize/' + curve_name + '.gltf')

    if gltf_file:
      export_sketch_normal_gltf(V, E, polylines, N_normalized, gltf_file)












  
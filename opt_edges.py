import numpy as np

from utility_io import load_sketch_polyline_data, write_normal_data, write_string_to_file
from utility_plot_viewer import plot_sketch_data, plot_edge_constraints, plot_edge_frames, plot_polyline_best_constraints, plot_polyline_normals
from utility_segment_distance import segment_to_segment_distance

from utility_convex_hull import get_sketch_edge_constraints, export_sketch_normal_gltf
from utility_parallel_transport import compute_parallel_transport_frames
from utility_parallel_transport_bidirection import parallel_transport_bi_direction
from utility_rotate_vector import rotation_matrix_from

import scipy.optimize as opt

from pathlib import Path
from collections import defaultdict

import argparse


# import jax.numpy as jnp
# import jax

def edge_distance_matrix(V, E):
    '''
    Compute a matrix of minimum distances between all pairs of edges in a mesh using 
    segment-to-segment distance calculations.
    '''
    n_edges = len(E)
    distances = np.zeros((n_edges, n_edges))
    
    for i, ei in enumerate(E):
        # Get vertices of first edge
        ei_v0, ei_v1 = V[ei[0]], V[ei[1]]
        
        for j in range(i + 1, len(E)):  # Only compute lower triangle due to symmetry
            ej = E[j]
            ej_v0, ej_v1 = V[ej[0]], V[ej[1]]

            # help me to save some time
            # if ei and ej already shared a vertex
            # their distance must be 0
            # Check if the set is empty
            if not set(ei) & set(ej):

                # Compute distance between segments, discarding closest points
                dist, _, _ = segment_to_segment_distance(ei_v0, ei_v1, ej_v0, ej_v1)
            
                distances[i, j] = dist
                distances[j, i] = dist
    
    return distances


# Only extract the distance 0 edges. No thresholds.
def extract_pairwise_weight(V, E, edge_to_polyline_map, unconstrained_polylines, edge_constraints_map, distances, n = 3):
    """
    Extract the n highest pairwise edge weights for each edge from the weight matrix,
    do not chose the edge from the same polyline.
    avoiding duplicates and ensuring each edge pair appears only once.
    
    Args:
        V: Vertex coordinates, shape (num_vertices, dimension)
        E: Edges as vertex index pairs, shape (num_edges, 2)
        edge_to_polyline_map: Mapping from edge index to its polyline index
        unconstrained_polylines: Set of polyline indices that don't have edge normals set
        edge_constraints_map: Edge normal constraints from convex hull, {edge_index: normal_vector}
        distances: NxN array where entry (i,j) is the weight between edges i and j
        n: Number of highest weights to extract for each edge (default: 3)

    Returns:
        list of tuples: [(edge_idx1, edge_idx2, weight)] 
    """
    pairwise = set()
    
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if distances[i, j] == 0:
                pairwise.add((i, j, 1))

    return pairwise


def build_vertex_to_edges_map(edges):
    '''
    Create a mapping from each vertex to all edges that contain it.
    
    Parameters:
    vertices: (n,3) array of vertex coordinates
    edges: (m,2) array of edge vertex index pairs
    
    Returns:
    dict: Mapping from vertex index to list of edge indices
    '''
    vertex_to_edges = defaultdict(list)
    
    for edge_idx, edge in enumerate(edges):
        # Add this edge to both of its vertices' lists
        vertex_to_edges[edge[0]].append(edge_idx)
        vertex_to_edges[edge[1]].append(edge_idx)
    
    for vertex_idx in vertex_to_edges:
        assert len(vertex_to_edges[vertex_idx]) == len(set(vertex_to_edges[vertex_idx])), \
            f"Vertex {vertex_idx} has duplicate edge entries"
    
    return vertex_to_edges

def create_edge_rotation_map( V, E ):
    '''
    V: (n,3) array of vertex coordinates
    E: (m,2) array of edge vertex index pairs
    '''
    rotations = {}

    for i in range(len(E)):
        for j in range(len(E)):

            if i == j:
                continue
            
            ei = E[i]
            ej = E[j]

            # if they share an endpoint
            shared_indices = set(ei) & set(ej)
            if len(shared_indices) != 0:
                assert len(shared_indices) == 1
                ## Get the shared index
                shared_index = next(iter(shared_indices))
                ## Get the non-shared index from ei
                ei_other_index = next( iter( set(ei) - shared_indices ))
                ## Get the non-shared index from ej
                ej_other_index = next( iter( set(ej) - shared_indices ))

                # Let's always go ei to ej
                vector_ei = V[shared_index] - V[ei_other_index]
                vector_ej = V[ej_other_index] - V[shared_index]

                rotations[(i, j)] = rotation_matrix_from( vector_ei, vector_ej )

    return rotations


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

def random_normal_on_polylines(Us, Vs, polyline_edge_data, polyline_indices):
    '''
    From the best normal of the polyline, parallel transport it on both direction of the polyline.
    Notice depend on the polyline_to_best_normal, this will transport on a subset of polyline.
    Means not all polylines will get normals after this.

    
    Parameters:
    -----------
    Us : frame
    Vs : frame
    polyline_edge_data : dictionary, {polyline_idx: (edge_indices, is_edge_reversed)}
                       : the polyline edge_indices and whether the edge is reversed

    polyline_indices: set or list containing the polyline which we want random normal on them

    Return :
        polyline_normals: dictionary, {polyline_index: [normals]} 
                          polyline_index as key and a list of normals corresponding to each edge in the polyline

    '''
    polyline_normals = {}

    for polyline_idx in polyline_indices:
        edge_indices, _ = polyline_edge_data[polyline_idx]
            

        normal_vectors = []
        # Compute parallel transport
        for edge_index in edge_indices:
            n = normal_for_edge( np.random.uniform(0, 2 * np.pi)  , Us[ edge_index ], Vs[ edge_index ] )
            normal_vectors.append(n)

        polyline_normals[polyline_idx] = normal_vectors
    
    return polyline_normals

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
    
def assign_normals_to_unconstrained_polylines(V, E, polyline_edge_data, edge_normal_map, distances):
    '''
    Assigns normals to unconstrained polylines by borrowing normals from nearby constrained edges.
    
    For polylines that don't have normal constraints from the convex hull, this function
    finds nearby edges that do have assigned normals and borrows them based on geometric
    compatibility. The best normal is selected based on perpendicularity to the edge tangent
    and proximity.
    
    Parameters:
    -----------
    vertices : array-like, shape (N, 3)
        Vertex coordinates in 3D space, where N is the number of vertices.
    
    edges : array-like, shape (M, 2)
        Edge connectivity, where each row contains indices (i,j) representing 
        an edge between vertices V[i] and V[j]. M is the number of edges.
    
    polyline_edge_data : tuple(list, list)
        A tuple containing (edge_indices, is_edge_reversed) where:
        - edge_indices: list of edge indices that form the polyline
        - is_edge_reversed: boolean list indicating whether each edge direction is reversed
    
    edge_normal_map : dict
        Dictionary mapping edge indices to their assigned normal vectors {edge_idx: normal_vector}
    
    distances : array-like, shape (M, M)
        Matrix of minimum distances between all pairs of edges
    
    Returns:
    --------
    tuple(int, ndarray)
        A tuple containing:
        - best_position: The position in the polyline where the best normal was found
        - best_normal: The normal vector assigned to the polyline
    '''
    # Collect potential normal candidates for each edge in the polyline
    normal_candidates = []
    edge_indices, is_edges_reversed = polyline_edge_data
    
    # For each edge in the polyline, find the closest edge with a known normal
    for target_edge_idx in edge_indices:
        # Sort nearby edges by distance to the target edge


        sorted_nearby_edges = sorted(
            [(nearby_edge_idx, distances[target_edge_idx, nearby_edge_idx]) 
             for nearby_edge_idx in range(len(distances))], 
            key=lambda x: x[1]
        )
        
        # Find the closest edge with a normal constraint
        for nearby_edge_idx, distance in sorted_nearby_edges:
            if nearby_edge_idx in edge_normal_map:
                # Get the normal from the nearby edge
                candidate_normal = edge_normal_map[nearby_edge_idx]
                
                # Compute the tangent of the target edge
                target_edge_tangent = compute_edge_tangent(V, E[target_edge_idx])
                
                # Calculate how perpendicular the normal is to the edge
                # A value of 1.0 means perfectly perpendicular, 0.0 means parallel
                perpendicularity = np.clip(1.0 - np.abs(np.dot(target_edge_tangent, candidate_normal)), 0, 1)
                
                # Only consider normals that are sufficiently perpendicular
                if perpendicularity > 1e-12:
                    # Store: (distance, perpendicularity, target edge, candidate normal)
                    normal_candidates.append((distance, perpendicularity, target_edge_idx, candidate_normal))
                    break  # Found a valid candidate, move to next edge
    
    # Create a map of candidate normals for edges in the polyline
    edge_to_candidate_normal_map = {}
    
    # Select the best candidates based on perpendicularity first, then distance
    # Limit to top 3 candidates for stability
    best_candidates = sorted(normal_candidates, key=lambda x: (x[1], x[0]))[:3]
    
    for _, _, target_edge_idx, candidate_normal in best_candidates:
        edge_to_candidate_normal_map[target_edge_idx] = candidate_normal
    
    # Find the best position and normal on the polyline using the candidate normals
    best_position, best_normal = find_best_perpendicular_normal_on_polyline(
        V, E, polyline_edge_data, edge_to_candidate_normal_map
    )
    
    return (best_position, best_normal)

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

def find_best_perpendicular_normal_on_polyline(vertices, edges, polyline_edge_data, edge_constraints_map):
    '''
    Identifies the edge normal from constraint map that is most perpendicular to its edge direction.
    
    For edges with normal constraints, calculates which normal is most perpendicular 
    to its corresponding edge tangent direction (smallest absolute dot product).
    
    Parameters:
    -----------
    vertices : ndarray, shape (n_vertices, 3)
        3D coordinates of vertices.
        
    edges : ndarray, shape (n_edges, 2)
        Edge connectivity, where edges[i] = [v1, v2] connects vertices[v1] to vertices[v2].
        
    polyline_edge_data : tuple (edge_indices, is_edge_reversed)
        Contains the indices of edges in this polyline and boolean flags indicating 
        whether each edge direction is reversed in the polyline.
        
    edge_constraints_map : dict {edge_idx: normal_vector}
        Maps edge indices to their constrained normal vectors (unit length).
        
    Returns:
    --------
    tuple (position_in_polyline, normal_vector) or None
        The position of the edge in the polyline and its normal vector that is
        most perpendicular to the edge direction. Returns None if no constrained edges found.
    '''
    edge_indices, _ = polyline_edge_data  # Unpack the tuple
    
    best_normal = None 
    min_dot_product = 1.0  # Dot product of perpendicular vectors is 0, so smaller is better
    best_position = None
    
    # Check each edge in the polyline
    for position, edge_idx in enumerate(edge_indices):
        # Skip edges without normal constraints
        if edge_idx not in edge_constraints_map:
            continue
            
        # Get the constrained normal and calculate the edge direction
        normal = edge_constraints_map[edge_idx]
        edge_tangent = compute_edge_tangent(vertices, edges[edge_idx])
        
        # Calculate perpendicularity (smaller absolute dot product = more perpendicular)
        perpendicularity = abs(np.dot(edge_tangent, normal))
        
        # Update if this is more perpendicular than previous best
        if best_normal is None or perpendicularity < min_dot_product:
            best_normal = normal
            min_dot_product = perpendicularity
            best_position = position
    
    assert best_position != None , "This should not happen"
    return (best_position, best_normal)

def estimate_initial_normals(V, E, P, polyline_to_edge_map, edge_to_polyline_map, edge_constraints_map, distances):
    '''
    Estimates initial normals for polylines using parallel transport.
    
    Parameters:
    -----------
    vertices : ndarray, shape (n_vertices, 3)
        Vertex coordinates in 3D space.
    
    edges : ndarray, shape (n_edges, 2)
        Edge connectivity, where each row contains indices (i,j) representing
        an edge between vertices[i] and vertices[j].
    
    polylines : list of arrays
        List where each element is an array of vertex indices forming a polyline.
    
    polyline_to_edge_map : dict {polyline_idx: (edge_indices, is_edge_reversed)}
        Mapping from polyline indices to the edges that compose them and whether
        each edge direction is reversed in the polyline.
    
    edge_to_polyline_map : dict {edge_idx: polyline_idx}
        Mapping from edge indices to the polyline they belong to.
    
    normal_constraints_map : {edge_idx, normal} 
        - edge_idx (int): Index of the edge in the edges array
        - normal (ndarray): (3,) normalized direction vector constraining the normal
    
    distances : array-like, shape (M, M)
        Matrix of minimum distances between all pairs of edges
        
    Returns:
    --------
    normals : ndarray, shape (n_edges, 3)
        The estimated normal vector for each edge.
    
    
    Algorithm:
    ----------
    1. For polylines with existing normal constraints:
       - Find the normal vector most perpendicular to the polyline
       - Parallel transport this normal along the polyline to get initial angles
       
    2. For polylines without normal constraints:
       - Locate the nearest polyline that has normal constraints
       - As long as the normal constraint are not parallel to the polyline
       - Try to locate a good one and parallel transport those normals to initialize angles
           
    Notes:
    ------
    - Uses parallel transport to maintain smooth normal vector field
    - For unconstrained polylines, proximity is determined by shortest distance
      between polyline midpoints
    '''

    # constrained_polyline_indices : which polyline_index have normal
    constrained_polyline_indices = set()
    for edge_idx, normal in edge_constraints_map.items():
        constrained_polyline_indices.add( edge_to_polyline_map[edge_idx] )

    print('constrained_polyline_indices',constrained_polyline_indices)


    ### First : Find most perpendicular normal for each polyline
    polyline_to_best_normal_map = {} 
    for polyline_idx in constrained_polyline_indices:
        polyline_edge_data = polyline_to_edge_map[polyline_idx]
        (pos_in_polyline, normal) = find_best_perpendicular_normal_on_polyline(V, E, polyline_edge_data, edge_constraints_map)
        polyline_to_best_normal_map[polyline_idx] = (pos_in_polyline, normal)

    # print('polyline_to_best_normal_map', polyline_to_best_normal_map)
    # print("len(polyline_to_best_normal_map)", len(polyline_to_best_normal_map))


    if show_plot:
        plot_polyline_best_constraints(V, E, P, polyline_to_best_normal_map, str='most perpendicular on polyline')

    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, convert_edge_dict_to_array(polyline_to_best_normal_map, len(E), polyline_to_edge_map), unconstrained_polylines_indices = None, filename = 'debug_normals_gltf/initial_most_perpendicular/' + curve_name + '.gltf')
    
    ### Second : Compute parallel transport for each polyline with a normal   
    polyline_normals = {}
    for polyline_idx, (pos_in_polyline, normal) in polyline_to_best_normal_map.items():
        polyline_points = [V[index] for index in P[polyline_idx]]            
        normal_vectors = parallel_transport_bi_direction(polyline_points, (pos_in_polyline, normal))
        polyline_normals[polyline_idx] = normal_vectors

    # print('len(polyline_normals)', len(polyline_normals))
    # print('polyline_normals', polyline_normals)

    if show_plot:
        plot_polyline_normals(V, E, P, polyline_normals, str = 'parallel transport most perpendicular normal')

    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, convert_edge_dict_to_array( polyline_normals, len(E), polyline_to_edge_map), unconstrained_polylines_indices = None, filename = 'debug_normals_gltf/initial_parallel_transport/' + curve_name + '.gltf' )
    
  
    ## Third pass:
    # now let me do this, for the polyline which does not have normals
    # edge < epsilon edge_normals as constraints to propogate on those who does not have normal
    # polylines

    unconstrained_polylines = set(range(len(P))) - set(polyline_normals.keys())
    # print("Polylines without normals:", unconstrained_polylines)

    # Update the edge-to-normal mapping 
    # with propagated normals from each polyline
    edge_constraints_map_updated = {}
    for polyline_idx, propagated_normals in polyline_normals.items():
        polyline_edges, _ = polyline_to_edge_map[polyline_idx]
    
        # Associate each edge with its corresponding normal vector from the parallel transport
        for position, edge_id in enumerate(polyline_edges):
            edge_constraints_map_updated[edge_id] = propagated_normals[position]

    

    # Identify polylines without normals
    unconstrained_polyline_to_best_normal_map = {}
    for polyline_idx in unconstrained_polylines:
        unconstrained_polyline_to_best_normal_map[polyline_idx] = assign_normals_to_unconstrained_polylines(V, E, polyline_to_edge_map[polyline_idx], edge_constraints_map_updated, distances)

    # print('len(unconstrained_polyline_to_best_normal_map)', len(unconstrained_polyline_to_best_normal_map))
    # print('unconstrained_polyline_to_best_normal_map', unconstrained_polyline_to_best_normal_map)

    if show_plot:
        plot_polyline_best_constraints(V, E, P, unconstrained_polyline_to_best_normal_map, scale=0.08, str = 'borrow nearby edge normal')
    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, convert_edge_dict_to_array(unconstrained_polyline_to_best_normal_map, len(E), polyline_to_edge_map), unconstrained_polylines, 'debug_normals_gltf/borrowed_normal/' + curve_name + '.gltf')
    
    unconstrained_polyline_normals = {}
    for polyline_idx, (pos_in_polyline, normal) in unconstrained_polyline_to_best_normal_map.items():
        polyline_points = [V[index] for index in P[polyline_idx]]          
        normal_vectors = parallel_transport_bi_direction(polyline_points, (pos_in_polyline, normal))
        unconstrained_polyline_normals[polyline_idx] = normal_vectors
    
    ## how about just use random normals for those who are unconstrainted? no, not good!

    if show_plot:
        plot_polyline_normals(V, E, P, unconstrained_polyline_normals, scale=0.08,  str = 'parallel transport of the polyline with borrowed normal')
    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, convert_edge_dict_to_array( unconstrained_polyline_normals, len(E), polyline_to_edge_map), unconstrained_polylines,  'debug_normals_gltf/borrowed_parallel_transport/' + curve_name + '.gltf')

    polyline_normals = {**polyline_normals, **unconstrained_polyline_normals}

    edge_constraints_map_estimated = {}
    for polyline_idx, propagated_normals in polyline_normals.items():
        polyline_edges, _ = polyline_to_edge_map[polyline_idx]
    
        # Associate each edge with its corresponding normal vector from the parallel transport
        for position, edge_id in enumerate(polyline_edges):
            edge_constraints_map_estimated[edge_id] = propagated_normals[position]

    if save_debug_gltf:
        info = (
            f' total_edges: {len(E)}\n'
            f' has_normal_edges: {len(edge_constraints_map)}\n'
            f' total_polylines: {len(P)}\n'
            f' has_normal_polylines: {len(polyline_to_best_normal_map)}\n'
            f' borrowed_normal_polylines: {len(unconstrained_polyline_normals)}'
        )
        
        write_string_to_file(info, 'debug_normals_gltf/normal_info/' + curve_name + '.txt')


    assert len( polyline_normals.keys() ) == len(P), "Some polylines do not have normals"
    
    if show_plot:
        plot_edge_constraints(V, E, P, edge_constraints_map_estimated, scale= 0.08, str= "initial estimate")

    return edge_constraints_map_estimated

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

def convert_edge_normals_to_array(edge_normals, num_edges):
    """
    Convert edge normal data to a numpy array, supporting both list and dictionary formats.
    This function is used so that we can save the edge constraints in a standardized format.
    
    Parameters:
    -----------
    edge_normals : list of tuples or dictionary
        Either:
        - List of (edge_index, normal_vector) pairs where normal_vector is a 3D vector.
        - Dictionary mapping edge indices to normal vectors {edge_index: normal_vector}
    
    num_edges : int
        Total number of edges in the mesh.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (num_edges, 3) containing normal vectors for each edge.
        Positions without assigned normals will be zero vectors.
    """
    # Initialize array with zeros
    normals = np.zeros((num_edges, 3))
    
    # Handle dictionary input format
    if isinstance(edge_normals, dict):
        for edge_index, normal in edge_normals.items():
            normals[edge_index] = normal
    # Handle list of tuples input format
    else:
        for edge_index, normal in edge_normals:
            normals[edge_index] = normal
            
    return normals

def convert_edge_dict_to_array(poly_line_edge_normal, num_edges, polyline_to_edge_map):
    '''
    Given:
        poly_line_edge_normal

        Dictionary in one of two formats
        1. {polyline_idx: (position_in_polyline, normal_vector)}
           - polyline_idx: index of the polyline
           - position_in_polyline: position of the edge in the polyline
           - normal_vector: reference normal vector
        2. {polyline_idx: [normal_vectors]}
           - polyline_idx: index of the polyline
           - normal_vectors: list of normal vectors corresponding to each edge in the polyline
    Return :
        numpy.ndarray
        Array of shape (num_edges, 3) containing normal vectors for each edge.
        Positions without assigned normals will be zero vectors.
    '''
    normals = np.zeros((num_edges, 3))

    for polyline_idx, normal_data in poly_line_edge_normal.items():
        
        edge_indices, _ = polyline_to_edge_map[polyline_idx]

        if isinstance(normal_data, tuple):
            # Format 1: (position_in_polyline, normal_vector)
            edge_pos, normal = normal_data
            edge_index = edge_indices[edge_pos]

            normals[edge_index] = normal 

        elif isinstance(normal_data, list):
            # Format 2: [normal_vectors] - list of normals for each edge
            assert len(edge_indices) == len(normal_data)
            for i in range(len(edge_indices)):
                edge_index = edge_indices[i]
                normal = normal_data[i]
                normals[edge_index] = normal
        

    return normals
            
                


## helper - never used
def create_edge_weight_matrix(E, distances, epsilon=1):
    """
    Create a matrix for edges with weights based on segment distances.
    The matrix is built in three steps:
    1. Initialize diagonal elements to 0 (edge to itself)
    2. When 2 edges are not in the same polyline,
       compute weights for all edge pairs based on inverse segment distance. 
    
    Args:
        V: (n,3) array of vertex coordinates
        distances: array-like, shape (M, M)
                   Matrix of minimum distances between all pairs of edges
        epsilon: small value to avoid division by zero
        
    Returns:
        numpy array: cost matrix where entry (i,j) is the weight between edges i and j
    """
    num_edges = len(E)
    weight_matrix = np.zeros((num_edges, num_edges))

    
    for i in range(num_edges):
        for j in range(i+1, num_edges):  
            
            # maybe a sigmod function here
            weight = 1.0 / (distances[i, j] * 10 + epsilon)

            weight_matrix[i,j] = weight     
            weight_matrix[j,i] = weight 
    
    return weight_matrix



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('normal_file', nargs='?', help='The curve sketch with optimized normal information.')
    parser.add_argument('gltf_file', nargs='?', help='The normal gltf file to save.')
    parser.add_argument('--normal_per_edge', type= str, choices=['one', 'two'], default='one',
                        help='One normal or 2 normal per edge')
    parser.add_argument('--show_plot', type=str, choices=['true', 'false'], default='true',
                   help='Whether to show the visualization plot (default: true)')    
    parser.add_argument('--save_debug_gltf', type=str, choices=['true', 'false'], default='true',
                   help='Save the gltf files for debug (default: true)')    
    
    args = parser.parse_args()

    curve_file = args.curve_file
    normal_file = args.normal_file
    NORMALS_PER_EDGE = args.normal_per_edge
    gltf_file = args.gltf_file
    show_plot = args.show_plot.lower() == 'true'
    save_debug_gltf = args.save_debug_gltf.lower() == 'true'


    if curve_file is None:
        curve_file = 'sketches/onshape/onshape_simple_mouse.obj'


    curve_name = Path(curve_file).stem

    #region Load Sketch and Plot Sketch
    ####################################

    # V: nx3 array of vertex coordinates
    # E: mx2 array of edge vertex index pairs with no duplicates
    # P: list of arrays containing vertex indices for each polyline
    V, E, P = load_sketch_polyline_data(curve_file)


    # This is not needed, but I want to check there's no duplicates
    from utility_dup_checker import check_duplicates
    dup_checker = check_duplicates(V, E, P)

    assert dup_checker['duplicate_vertices'] == False
    assert dup_checker['duplicate_edges'] == False  
    assert dup_checker['duplicate_polylines'] == False 



    print('len(V)', len(V))
    print('len(E)', len(E))
    print('len(P)', len(P))
    print()

    print('V.shape', V.shape)
    print('E.shape', E.shape)



    # only need points and polyline indices to draw
    # same polyline, same color
    if show_plot:
        plot_sketch_data(V, P)

    #####################################
    #endregion


    #region Initial normal estimate
    #####################################

    # edge normal from convex hull
    # edge_constraints : [ (edge_idx, normal) ]
    # only the ones with normal will be in the list 
    edge_constraints = get_sketch_edge_constraints(V, E)
    # print('edge_constraints', edge_constraints)
    print('len(edge_constraints)', len(edge_constraints))
    
    edge_constraints_map = {edge_idx: normal for edge_idx, normal in edge_constraints}


 
    # Build polyline to edge and edge to polyline mappings
    polyline_to_edge_map = {}
    edge_to_polyline_map = {}
    for polyline_idx, polyline in enumerate(P):
        # Map polyline to its edges and edge orientations
        edge_indices, is_edge_reversed = find_edge_indices_from_polyline(polyline, E)
        polyline_to_edge_map[polyline_idx] = (edge_indices, is_edge_reversed)
        
        # Create reverse mapping from each edge to its parent polyline
        for edge_idx in edge_indices:
            edge_to_polyline_map[edge_idx] = polyline_idx


    constrained_polyline_indices = set()
    for edge_idx, normal in edge_constraints:
        constrained_polyline_indices.add( edge_to_polyline_map[edge_idx] )

    unconstrained_polylines_indices = set(range(len(P))) - constrained_polyline_indices

    # plot and save debug gltf
    if show_plot:
        plot_edge_constraints(V, E, P, edge_constraints, scale=0.08, str = 'edge constraints from convex hull', filename= None, block= True)
        write_normal_data(V, E, convert_edge_normals_to_array(edge_constraints, len(E)) , 'debug_normals_gltf/edge_normals/' + curve_name + '.normal')
    
    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, edge_constraints, unconstrained_polylines_indices= unconstrained_polylines_indices, filename ='debug_normals_gltf/edge_normals/' + curve_name + '.gltf' )



    print('constrained_polyline_indices', constrained_polyline_indices)
    print('unconstrained_polylines_indices', unconstrained_polylines_indices)

    vertex_to_edges_map = build_vertex_to_edges_map( E )

    # print('vertex_to_edges', vertex_to_edges)
    # compute distance between edges 
    
    distances = edge_distance_matrix(V, E)

    estimate_normals = estimate_initial_normals(V, E, P, polyline_to_edge_map, edge_to_polyline_map, edge_constraints_map, distances)

    if save_debug_gltf:
        export_sketch_normal_gltf(V, E, P, convert_edge_normals_to_array(estimate_normals, len(E)), unconstrained_polylines_indices , filename = 'debug_normals_gltf/initial_estimate/' + curve_name + '.gltf')
        write_normal_data(V, E, convert_edge_normals_to_array(estimate_normals, len(E)) , 'debug_normals_gltf/initial_estimate/' + curve_name + '.normal')



    # computes local coordinate frames for edges by generating parallel transport 
    # frames along polylines and mapping them to the global edge indices
    Us, Vs = create_frames_for_each_polyline( V, E, P )
    
    # show the frame on each edge
    # if show_plot:
    #     plot_edge_frames(V, E, P, Us, Vs, scale=0.08)
    
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)
    # print('thetas0', thetas0)

    #####################################
    #endregion



    # weight_matrix = create_edge_weight_matrix(E, distances)
    rotations = create_edge_rotation_map(V, E)

    # print('weight_matrix', weight_matrix)
    pairwise = extract_pairwise_weight(V, E, edge_to_polyline_map, unconstrained_polylines_indices, edge_constraints_map, distances)


    #region Optimization one normal
    #####################################


    
    # print('len(pairwise)', len(pairwise))


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
            plot_edge_constraints(V, E, P, normals_now, unconstrained_polylines_indices, 
                                scale=0.08, 
                                str=f"Optimization: Iteration {iteration_counter[0]}", 
                                block=False)





            
        result = opt.minimize( E_total,
            thetas0,  
            args=(Us, Vs, edge_constraints, pairwise),  # Pass additional arguments
            method = 'L-BFGS-B', 
            tol = 0.0000001, 
            options = { 'disp': True, 'gtol': 0.0000001, 'maxiter': 1000 },
            callback=callback
        )

        thetas = result.x

        # print(thetas)
        
        ##################################
        #endregion

        opt_normals = recover_normal_from_thetas(thetas, Us, Vs)


        N = [normal for _,normal in opt_normals]
        N_normalized = N / np.linalg.norm(N, axis=1)[:, np.newaxis]


        
        if save_debug_gltf:
            export_sketch_normal_gltf(V, E, P, N_normalized, unconstrained_polylines_indices, filename ='debug_normals_gltf/final_optimize/' + curve_name + '.gltf')
            write_normal_data(V, E, N_normalized , 'debug_normals_gltf/final_optimize/' + curve_name + '.normal')

        if show_plot:
            plot_edge_constraints(V, E, P,  opt_normals, unconstrained_polylines_indices, scale=0.08, str = "optimize result")

        if gltf_file:
            export_sketch_normal_gltf(V, E, P, N_normalized,  unconstrained_polylines_indices, filename = gltf_file)

        if normal_file:
            write_normal_data(V, E, N_normalized , normal_file)

    elif NORMALS_PER_EDGE == 'two':

        def E_total_two_edges_per_normal( thetas, Us, Vs, constraints, pairwise ):
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
            # Calculate the constraint energy
            # Constrain both edges
            E_constraint = 0.0
            for edge_index, desired_normal_vector in constraints:
                for which_edge in (0,1):
                    n = normal_for_edge( thetas[ edge_index, which_edge ], Us[ edge_index ], Vs[ edge_index ] )
                    E_constraint += (1.0 - np.dot( n, desired_normal_vector ) )**2
            # normalize
            E_constraint /= 2*len( constraints )
            
            E_pairwise = 0.0
            W_pairwise = 0.0
            for e1, e2, weight in pairwise:
                costs = np.zeros( (2,2) )
                for i in range(2):
                    for j in range(2):
                        n1 = normal_for_edge( thetas[e1,i], Us[e1], Vs[e1] )
                        n2 = normal_for_edge( thetas[e2,j], Us[e2], Vs[e2] )

                        ## Get the rotation matrix if it exists
                        if (e1,e2) in rotations: n1 = rotations[(e1,e2)] @ n1
                        elif (e2,e1) in rotations: n1 = rotations[(e2,e1)].T @ n1

                        costs[i,j] = (1.0 - np.dot( n1, n2 ) )**2
                
                shared_vertex = tuple(frozenset( E[e1] ) & frozenset( E[e2] ))
                
                ## If this is a curve edge, we want to penalize the best match
                if len( shared_vertex ) == 1 and len(vertex_to_edges_map[ shared_vertex[0] ]) == 2:
                    if costs[0,0] + costs[1,1] < costs[0,1] + costs[1,0]:
                        E_pairwise += weight * (costs[0,0] + costs[1,1])
                        W_pairwise += weight
                    else:
                        E_pairwise += weight * (costs[0,1] + costs[1,0])
                        W_pairwise += weight
                ## Otherwise, the edges are disconnected or higher-valence, in which case we just want the
                ## lowest cost.
                else:
                    E_pairwise += weight * costs.min()
                    W_pairwise += weight
            
            # Normalize by the total weight
            E_pairwise /= W_pairwise
            return 1e-2 * E_constraint +  1e4 * E_pairwise
        
        # Assuming thetas0 is already a 1D array with length equal to len(E)
        # Create a flattened version with two copies for the optimizer
        thetas0_flat = np.concatenate((thetas0, thetas0))

        # Add this wrapper function that reshapes the 1D array to 2D for your function
        def E_total_wrapper(thetas_flat, Us, Vs, constraints, pairwise):
            # Reshape the 1D array to a 2D array
            num_edges = len(Us)
            thetas_2d = thetas_flat.reshape(num_edges, 2)  
            return E_total_two_edges_per_normal(thetas_2d, Us, Vs, constraints, pairwise)

        # Use the wrapper in your optimization
        result = opt.minimize(E_total_wrapper,
                            thetas0_flat,
                            args=(Us, Vs, edge_constraints, pairwise),
                            method='L-BFGS-B',
                            tol=0.0000001,
                            options={'disp': True, 'gtol': 0.0000001, 'maxiter': 1000}
                            )

        # Reshape the result back to 2D
        thetas = result.x.reshape(len(E), 2)

        thetas = result.x
        assert len( thetas ) == 2*len(E)

        print(thetas)

        opt_normals1 = recover_normal_from_thetas(thetas[:len(E)], Us, Vs)
        opt_normals2 = recover_normal_from_thetas(thetas[len(E):], Us, Vs)

        plot_edge_constraints(V, E, P,  opt_normals1, scale=0.08, str= 'normal_1')
        plot_edge_constraints(V, E, P,  opt_normals2, scale=0.08, str='normal_2')





  
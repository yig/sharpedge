import numpy as np

from utility_io import load_sketch_polyline_data, write_normal_data, write_string_to_file
from utility_plot_viewer import plot_sketch_data, plot_edge_constraints, plot_edge_frames, plot_polyline_best_constraints, plot_polyline_normals, plot_edge_constraints_two_normals, plot_edge_info
from utility_segment_distance import segment_to_segment_distance

from utility_convex_hull import get_sketch_edge_constraints, export_sketch_normal_gltf, export_sketch_two_normal_gltf
from utility_parallel_transport import compute_parallel_transport_frames
from utility_parallel_transport_bidirection import parallel_transport_bi_direction
from utility_rotate_vector import rotation_matrix_from

import scipy.optimize as opt

from pathlib import Path
from collections import defaultdict

import argparse

import jax
import jax.numpy as jnp
from jax import grad, jit
import time 

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
def extract_pairwise_weight(E, distances):
    """
    Extract the n highest pairwise edge weights for each edge from the weight matrix,
    do not chose the edge from the same polyline.
    avoiding duplicates and ensuring each edge pair appears only once.
    
    Args:
        E: Edges as vertex index pairs, shape (num_edges, 2)
        distances: NxN array where entry (i,j) is the weight between edges i and j

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

    
    # TODO:
    # This should not happen, but incase, I can just assign a random normal 




    
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
    estimated_normals : dict
        {edge_idx, normal_vector}
        
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

def normal_for_edge( theta, U, V):  return jnp.cos( theta ) * U + jnp.sin( theta ) * V

##JAX-compatible functions

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
        # thetas_2d[:, 0] -= 1e-3  # Small perturbation for first normal
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
            

def random_normal_for_edge( U, V) :
    '''
    create a random normal for edge, given the frame U and V.
    This normal will always be perpendicular to the edge
    '''
    return normal_for_edge( np.random.uniform(0, 2 * np.pi), U, V )                


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


def collect_edge_indices_from_strokes(stroke_indices, polylines_edge_data):
    """
    Collects all edge indices from specified strokes/polylines.
    
    This function gathers all edge indices that make up the specified strokes.
    
    Parameters
    ----------
    stroke_indices : list of int
        Indices of strokes/polylines to extract edge information from.
    
    polyline_to_edge_map : dict {polyline_idx: (edge_indices, is_edge_reversed)}
        Mapping from polyline indices to the edges that compose them and whether
        each edge direction is reversed in the polyline.
    
    Returns
    -------
    list of int
        A flattened list of all edge indices that make up the specified strokes.
        May contain duplicates if edges are shared between strokes.

    """
    all_edge_indices = []
    
    for stroke_index in stroke_indices:
        if stroke_index in polyline_to_edge_map:
            edge_indices, _ = polylines_edge_data[stroke_index]
            all_edge_indices.extend(edge_indices)
    
    return all_edge_indices
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Optimize edges to get normals')
    parser.add_argument('curve_file', nargs='?', help='The curve sketch to load.')
    parser.add_argument('normal_file', nargs='?', help='The curve sketch with optimized normal information.')
    parser.add_argument('gltf_file', nargs='?', help='The normal gltf file to save.')
    parser.add_argument('-p', '--normal_per_edge', type=int, choices=[1, 2], default=1,
                    help='Number of normals per edge (1 or 2)')
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
    mode_map = {1: 'one', 2: 'two'}
    normals_per_edge = mode_map[args.normal_per_edge]

    print('normals_per_edge', normals_per_edge)


    if curve_file is None:
        curve_file = 'sketches/onshape/onshape_simple_mouse.obj'
        curve_file = 'made_examples/sketch/cylinder.obj'



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
        # plot_edge_info(V, E)

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



  

    # print('weight_matrix', weight_matrix)
    pairwise = extract_pairwise_weight(E, distances)
    
    # Create JAX-friendly edge rotation map
    rotations_data = precompute_edge_rotation_map(V, E)

    #region Optimization 
    #####################################


    callback_fn = create_callback(Us, Vs, E, P, V, mode=normals_per_edge)

    # one_normal = [23, 58]
    
    # for sketches/onshape/onshape_simple_mouse.obj
    one_normal = collect_edge_indices_from_strokes([0, 2, 4], polyline_to_edge_map)
    
    # for sketches/onshape/onshape_simple_shape.obj
    one_normal = collect_edge_indices_from_strokes([0], polyline_to_edge_map)

    print('one_normal', one_normal)

    result = optimize_normal_angles(
        thetas0, Us, Vs, edge_constraints, pairwise, rotations_data, E, vertex_to_edges_map,
        mode=normals_per_edge, one_normal=one_normal, callback_fn=callback_fn
    )

    normals = recover_normals(result, Us, Vs, mode=normals_per_edge)

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

        export_sketch_two_normal_gltf(V, E, P, normals, unconstrained_polylines_indices, filename='debug_normals_gltf/final_optimize_two_normals/' + curve_name + '.gltf')


        # if save_debug_gltf:
        #     export_sketch_normal_gltf(V, E, P, N_normalized, unconstrained_polylines_indices, filename ='debug_normals_gltf/final_optimize/' + curve_name + '.gltf')
        #     write_normal_data(V, E, N_normalized , 'debug_normals_gltf/final_optimize/' + curve_name + '.normal')

        # if show_plot:
        #     plot_edge_constraints(V, E, P,  opt_normals, unconstrained_polylines_indices, scale=0.08, str = "optimize result")

        # if gltf_file:
        #     export_sketch_normal_gltf(V, E, P, N_normalized,  unconstrained_polylines_indices, filename = gltf_file)

        # if normal_file:
        #     write_normal_data(V, E, N_normalized , normal_file)

    #####################################
    #endregion
  
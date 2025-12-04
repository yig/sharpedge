import numpy as np

from utility_io import load_sketch_polyline_data, write_normal_data, write_string_to_file, write_two_normal
from utility_plot_viewer import plot_sketch_data, plot_edge_constraints, plot_edge_frames, plot_polyline_best_constraints, plot_polyline_normals, plot_edge_constraints_two_normals, plot_edge_info
from utility_segment_distance import segment_to_segment_distance

from utility_convex_hull import get_sketch_edge_constraints, export_sketch_normal_gltf, export_sketch_two_normal_gltf
from utility_parallel_transport import compute_parallel_transport_frames
from utility_parallel_transport_bidirection import parallel_transport_bi_direction
from utility_rotate_vector import rotation_matrix_from
from utility_geometry_tools import compute_edge_tangent, are_parallel_cos
from utility_debug_options import DebugOptions

from utility_high_valence_sort_edges import compute_edge_circulation_graph_laplacian

import scipy.optimize as opt

from pathlib import Path
from collections import defaultdict

import argparse

import jax
import jax.numpy as jnp
from jax import grad, jit
import time 

def edge_distance_matrix(V, E):
    """
    Compute a symmetric matrix of minimum distances between all edges.

    V : (n, 3) float array of vertex positions
    E : (m, 2) int array of edges (vertex indices)
    """
    n_edges = len(E)
    distances = np.zeros((n_edges, n_edges))
    
    for i, ei in enumerate(E):
        ei_v0, ei_v1 = V[ei[0]], V[ei[1]]
        
        # Only compute for j > i (matrix is symmetric)
        for j in range(i + 1, len(E)):  
            ej = E[j]
            
            # Shared vertex → distance = 0 (skip computation)
            if set(ei) & set(ej):
                continue

            ej_v0, ej_v1 = V[ej[0]], V[ej[1]]
            dist, _, _ = segment_to_segment_distance(ei_v0, ei_v1, ej_v0, ej_v1)
        
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances

def extract_pairwise_weight(V, E, distances):
    """
    Extract all edge pairs whose distance == 0.
    
   
    V : (n, 3) float array of vertex positions 
    E : (m, 2) int array of edges (vertex indices)
    distances : (m, m) float array of edge-to-edge distances


    Returns:
        list of (edge_i, edge_j, weight)
        weight is fixed to 1. 
        (Tried cosine-based weights |e1_vec·e2_vec|/||e1_vec||·||e2_vec|| * 0.5 + 0.5; 
        made no practical difference.)
    """
    pairwise = set()
    
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            # Only include edges whose segment distance is exactly zero
            if distances[i, j] == 0:
                pairwise.add((i, j, 1))
    return pairwise
   
def build_vertex_to_edges_map(edges):
    """
    Map each vertex index to the list of edges that use it.

    edges : (m, 2) int array of vertex index pairs

    Returns
    -------
    dict
        vertex_index → list of edge indices
    """
    vertex_to_edges = defaultdict(list)
    
    for edge_idx, (v0, v1) in enumerate(edges):
        vertex_to_edges[v0].append(edge_idx)
        vertex_to_edges[v1].append(edge_idx)
    
    # Safety check: ensure no duplicate edge entries
    for v, lst in vertex_to_edges.items():
        assert len(lst) == len(set(lst)), f"Vertex {v} has duplicate edge entries."

    return vertex_to_edges

def find_edge_indices_from_polyline(polyline, E):
    '''
    Convert a polyline (vertex sequence) into edge indices in E.

    polyline : list of vertex indices [v0, v1, v2, ...]
    E        : (m, 2) int array of edges

    Returns
    -------
    edge_indices : list of int
        Indices of edges corresponding to each segment of the polyline.
    edge_reversed : list of bool
        False if edge direction in E matches the polyline direction True otherwise.
    '''
    edge_indices = []
    edge_reversed = []
    
    for i in range(len(polyline) - 1):
        v1, v2 = polyline[i], polyline[i + 1]
        
        forward = (E[:, 0] == v1) & (E[:, 1] == v2)
        backward = (E[:, 0] == v2) & (E[:, 1] == v1)
        
        if np.any(forward):
            edge_idx = np.where(forward)[0][0]
            edge_indices.append(edge_idx)
            edge_reversed.append(False)
        elif np.any(backward):
            edge_idx = np.where(backward)[0][0]
            edge_indices.append(edge_idx)
            edge_reversed.append(True)
        else:
            raise ValueError(f"Edge ({v1},{v2}) not found in E.")
    
    return edge_indices, edge_reversed

def create_frames_for_each_polyline(V, E, P):
    '''
    Construct local (U, V) frames for each edge in the polyline network.
    
    Both U and V lie in the normal plane of the edge, i.e., both are
    perpendicular to the edge tangent. 

    V : (n, 3) array of vertex positions
    E : (m, 2) array of edges (vertex index pairs)
    P : list of polylines, each a list of vertex indices
    
    Returns
    -------
    Us : list of ndarray
        First basis vector in the normal plane
    Vs : list of ndarray
        Second basis vector in the normal plane
    '''
    Us = [None] * len(E)
    Vs = [None] * len(E)

    for polyline in P:
        points = [V[idx] for idx in polyline]
        polyline_u, polyline_v = compute_parallel_transport_frames( points )

        edge_indices, edge_reversed = find_edge_indices_from_polyline(polyline, E)
   
        # Flip if edge orientation is reversed
        Us_poly = [-u if rev else u for u, rev in zip(polyline_u, edge_reversed)]
        Vs_poly = [-v if rev else v for v, rev in zip(polyline_v, edge_reversed)]
        
        for k, (u, v) in enumerate(zip(Us_poly, Vs_poly)):
            Us[edge_indices[k]] = u
            Vs[edge_indices[k]] = v
    
    return Us, Vs 
 
def assign_normals_to_unconstrained_polylines(V, E, polyline_edge_data, edge_normal_map, distances):
    '''
    Assigns normals to unconstrained polylines by borrowing normals from nearby constrained edges.
    
    For polylines that don't have normal constraints from the convex hull, this function
    finds nearby edges that do have assigned normals and borrows them based on geometric
    compatibility. The best normal is selected based on perpendicularity to the edge tangent
    and proximity.
    

    V : (n, 3) array of vertex positions
    E : (m, 2) array of edges
    polyline_edge_data : (edge_indices, edge_reversed)
    edge_normal_map : dict {edge_idx → normal vector}
    distances : (m, m) array of pairwise edge distances
    
    Returns
    -------
    best_position : int
        Index along the polyline where the normal is assigned.
    best_normal : ndarray
        Selected normal vector.
    '''
    # Gather candidate normals for edges in the polyline
    normal_candidates = []
    edge_indices, is_edges_reversed = polyline_edge_data
    
    # For each polyline edge, find the nearest constrained edge
    for target_edge_idx in edge_indices:

        # Edges sorted by distance to the target edge
        sorted_nearby_edges = sorted(
            [(j, distances[target_edge_idx, j]) for j in range(len(distances))],
            key=lambda x: x[1]
        )

        # Pick the first nearby edge that has an assigned normal
        for nearby_edge_idx, distance in sorted_nearby_edges:
            if nearby_edge_idx in edge_normal_map:

                candidate_normal = edge_normal_map[nearby_edge_idx]
                tangent = compute_edge_tangent(V, E[target_edge_idx])

                # Perpendicularity score (1 = perfectly perpendicular)
                perpendicularity = 1.0 - abs(np.dot(tangent, candidate_normal))
                perpendicularity = np.clip(perpendicularity, 0, 1)

                # Keep only sufficiently perpendicular normals
                if perpendicularity > 1e-12:
                    # Store: (distance, perpendicularity, target_edge_idx, normal)
                    normal_candidates.append(
                        (distance, perpendicularity, target_edge_idx, candidate_normal)
                    )
                    break  # Move to next target edge
    

    # Pick top candidates: prioritize perpendicularity, then distance
    best_candidates = sorted(normal_candidates, key=lambda x: (x[1], x[0]))[:3]

    # Sanity check: this should never happen
    if not best_candidates:
        raise RuntimeError("Unexpected: no valid normal candidates for this polyline.")


    # Map polyline edges → candidate normals
    edge_to_candidate_normal_map = {
        t_idx: normal for _, _, t_idx, normal in best_candidates
    }

    # Determine the best position and normal along the polyline
    best_position, best_normal = find_best_perpendicular_normal_on_polyline(
        V, E, polyline_edge_data, edge_to_candidate_normal_map
    )

    return (best_position, best_normal)

def find_best_perpendicular_normal_on_polyline(vertices, edges, polyline_edge_data, edge_constraints_map):
    """
    Select the constrained normal (from edge_constraints_map) that is most
    perpendicular to its edge tangent within a polyline.

    Parameters
    ----------
    vertices : (N, 3) float array
    edges : (M, 2) int array
    polyline_edge_data : tuple (edge_indices, is_edge_reversed)
    edge_constraints_map : dict {edge_idx: normal_vector}

    Returns
    -------
    (position_in_polyline, normal_vector)
        The edge-position and its normal with minimal |dot(tangent, normal)|.
        Raises if the polyline contains no constrained edges.
    """
    edge_indices, _ = polyline_edge_data  # Unpack the tuple
    
    best_pos = None
    best_normal = None
    best_dot = 1.0   # dot = 0 means perfect perpendicular
    
    for pos, edge_idx in enumerate(edge_indices):
        if edge_idx not in edge_constraints_map:
            continue

        normal = edge_constraints_map[edge_idx]
        tangent = compute_edge_tangent(vertices, edges[edge_idx])
        dot_val = abs(np.dot(tangent, normal))

        if best_normal is None or dot_val < best_dot:
            best_dot = dot_val
            best_normal = normal
            best_pos = pos

    # Sanity check: the caller guarantees at least one constrained edge.
    if best_pos is None:
        raise RuntimeError("No constrained edges found in this polyline (unexpected).")

    return best_pos, best_normal

def estimate_initial_normals(V, E, P, polyline_to_edge_map, edge_to_polyline_map, edge_constraints_map, distances, vertex_to_edges_map):
    '''

    Estimate an initial normal field for all polylines using constrained normals,
    local propagation, and borrowing from nearby polylines.
        
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
    
    vertex_to_edges_map: dict
        Mapping from vertex index to list of edge indices

    Returns:
    --------
    normals : ndarray, shape (n_edges, 3)
        The estimated normal vector for each edge.
    
    
    Algorithm:
    ----------
    1. For polylines with existing normal constraints:
       - Find the normal vector most perpendicular to the polyline
       - Parallel transport this normal along the polyline to get initial angles
       
    2. For polylines without normals:
       - If any edge(polyline) connect to it has normal, also propogate them on the neighbor edges
       - It could be both start and end edge of the polyline have neighboring edges, use one, also use the average normal.
       - Parallel transport this normal along the polyline to get initial angles
   
       
    3. For polylines still without normal constraints:
       - Locate the nearest polyline that has normal constraints
       - As long as the normal constraint are not parallel to the polyline
       - Try to locate a good one and parallel transport those normals to initialize angles

    All polylines must end with an assigned normal.
    '''

    # Stage 1: Identify polylines with explicit normal constraints
    constrained_polyline_indices = {
        edge_to_polyline_map[edge_idx]
        for edge_idx in edge_constraints_map
    }

    debug.log("constrained_polylines:", constrained_polyline_indices)

    # Stage 1a: Find most perpendicular normals
    polyline_to_best_normal_map = {}

    for polyline_idx in constrained_polyline_indices:
        poly_edge_data = polyline_to_edge_map[polyline_idx]
        pos, normal = find_best_perpendicular_normal_on_polyline(
            V, E, poly_edge_data, edge_constraints_map
        )
        polyline_to_best_normal_map[polyline_idx] = (pos, normal)

    debug.plot(
        plot_polyline_best_constraints,
        V, E, P, polyline_to_best_normal_map,
        str='most perpendicular on polyline'
    )

    debug.save(
        export_sketch_normal_gltf,
        V, E, P,
        convert_edge_dict_to_array(polyline_to_best_normal_map, len(E), polyline_to_edge_map),
        unconstrained_polylines_indices=None,
        filename=f'debug_normals_gltf/initial_most_perpendicular/{curve_name}.gltf'
    )
    

    # Stage 1b: Parallel transport these normals along each polyline
    polyline_normals = {}

    for polyline_idx, (pos, normal) in polyline_to_best_normal_map.items():
        polyline_points = [V[i] for i in P[polyline_idx]]
        normals = parallel_transport_bi_direction(polyline_points, (pos, normal))
        polyline_normals[polyline_idx] = normals

    debug.plot(
        plot_polyline_normals,
        V, E, P, polyline_normals,
        scale=0.05,
        str='parallel transport most perpendicular normal'
    )

    debug.save(
        export_sketch_normal_gltf,
        V, E, P,
        convert_edge_dict_to_array(polyline_normals, len(E), polyline_to_edge_map),
        unconstrained_polylines_indices=None,
        filename=f'debug_normals_gltf/initial_parallel_transport/{curve_name}.gltf'
    )


    # Build updated edge→normal map
    edge_constraints_map_updated = {}
    for polyline_idx, normals in polyline_normals.items():
        polyline_edges, _ = polyline_to_edge_map[polyline_idx]
        assign_normals_to_edges(edge_constraints_map_updated, polyline_edges, normals)


    # Propagate normals from constrained polylines to adjacent ones:
    #
    # 1. Initialize frontier_edges using the start/end edges of polylines that already have normals.
    # 2. From each frontier edge, collect neighboring edges without normals → to_expand_edges.
    # 3. For each edge in to_expand_edges, choose the vertex to expand from (one connected to a known-normal edge).
    # 4. Compute a new normal for each expandable edge by rotating nearby known normals.
    # 5. Determine which polylines these edges belong to and parallel-transport the normal along them.
    # 6. These newly-normalized polylines contribute new frontier edges.
    #
    # Repeat until no new edges can be expanded; remaining polylines will borrow normals.


    # --- Stage 2: Graph expansion to neighboring polylines ---

    unconstrained_polylines = set(range(len(P))) - set(polyline_normals.keys())
    debug.log("unconstrained polylines (initial):", unconstrained_polylines)

    # Frontier edges = start and end edges of all polylines that already have normals
    frontier_edges = set()

    for poly_idx in polyline_normals:
        edges_in_polyline, _ = polyline_to_edge_map[poly_idx]
        frontier_edges.add(edges_in_polyline[0])      # start edge
        frontier_edges.add(edges_in_polyline[-1])     # end edge
        

    while frontier_edges:
        # debug.log("frontier_edges:", frontier_edges)

        # ---- Step 1: find expandable edges ----
        to_expand_edges = set()      
        to_expand_vertex = {}    

        for f_edge in frontier_edges:
            v0, v1 = E[f_edge]

            # All neighbor edges that share v0 or v1
            v0_neighbors = vertex_to_edges_map[v0]
            v1_neighbors = vertex_to_edges_map[v1]

            # From v0 side: if neighbor edge has NO normal, it should be expanded
            for nbr_edge in v0_neighbors:
                if nbr_edge not in edge_constraints_map_updated:
                    to_expand_edges.add(nbr_edge)

            # From v1 side: same logic
            for nbr_edge in v1_neighbors:
                if nbr_edge not in edge_constraints_map_updated:
                    to_expand_edges.add(nbr_edge)
        
        # debug.log('to_expand_edges', to_expand_edges)
        
        # ---- Step 2: determine expansion direction for each edge ----
        for e_edge in to_expand_edges:
            v0, v1 = E[e_edge] 
            
            v0_neighbors = vertex_to_edges_map[v0]
            v1_neighbors = vertex_to_edges_map[v1]

            for nbr_edge in v0_neighbors:
                if nbr_edge in edge_constraints_map_updated:
                    to_expand_vertex[e_edge] = v0
            for nbr_edge in v1_neighbors:
                if nbr_edge in edge_constraints_map_updated:
                    to_expand_vertex[e_edge] = v1
        
        # debug.log('to_expand_vertex', to_expand_vertex)

        # ---- Step 3: propagate normals to expandable edges ----
        to_expand_normals = {}

        for edge_id, shared_vertex in to_expand_vertex.items():

            accumulated = []
            incident_edges = vertex_to_edges_map[shared_vertex]

            for nbr_edge in incident_edges:

                # skip unconstrained neighbors
                if nbr_edge not in edge_constraints_map_updated:
                    continue

                nbr_normal = edge_constraints_map_updated[nbr_edge]

                # parallel transport from nbr_edge → edge_id
                transported = transport_normal_between_edges(
                    V, E, nbr_edge, edge_id, shared_vertex, nbr_normal
                )

                # filter out rotated normals parallel to tangent
                tangent = compute_edge_tangent(V, E[edge_id])
                if are_parallel_cos(tangent, transported):
                    continue

                accumulated.append(transported)

            if accumulated:
                avg = np.sum(accumulated, axis=0)
                to_expand_normals[edge_id] = avg / np.linalg.norm(avg)

        debug.log('to_expand_normals', to_expand_normals)

        debug.plot(
            plot_edge_constraints,
            V, E, P, 
            to_expand_normals, 
            scale=0.08, 
            str = 'propagate from nearby normals', 
            block = True)

        # ---- Step 4: propagate normals from expandable edges to their polylines ----

        expand_polylines = set()          # which polylines will receive new normals
        expand_polyline_normals = {}      # edge_id → propagated normal

        for edge_id, base_normal in to_expand_normals.items():

            # Which polyline does this edge belong to?
            poly_idx = edge_to_polyline_map[edge_id]
            edge_indices, _ = polyline_to_edge_map[poly_idx]

            expand_polylines.add(poly_idx)

            # Collect polyline geometry
            poly_points = [V[v_idx] for v_idx in P[poly_idx]]

            # Decide where the known normal attaches
            if edge_id == edge_indices[0]:
                start_pos = 0
            elif edge_id == edge_indices[-1]:
                start_pos = len(edge_indices) - 1
            else:
                start_pos = edge_indices.index(edge_id)
    
            # Parallel transport to get full normal field on this polyline
            transported_normals = parallel_transport_bi_direction(
                poly_points,
                (start_pos, base_normal)
            )

            # Store results
            polyline_normals[poly_idx] = transported_normals

            # Assign propagated normals to each edge of the polyline
            for pos, eid in enumerate(edge_indices):
                edge_constraints_map_updated[eid] = transported_normals[pos]
                expand_polyline_normals[eid] = transported_normals[pos]
                
        # debug.plot( 
        #     plot_edge_constraints, 
        #     V, E, P, 
        #     expand_polyline_normals, 
        #     scale= 0.05, 
        #     str= "expand polyline normals",
        #     block=True)

        # ---- Step 5: update frontier edges for the next BFS-like iteration ----
        frontier_edges = set()

        for poly_idx in expand_polylines:
            edge_indices, _ = polyline_to_edge_map[poly_idx]

            # Boundary edges of this polyline will act as new frontier
            start_edge = edge_indices[0]
            end_edge   = edge_indices[-1]

            frontier_edges.add(start_edge)
            frontier_edges.add(end_edge)

        # Optionally visualize the updated state
        debug.plot(
            plot_edge_constraints,
            V, E, P,
            edge_constraints_map_updated,
            scale=0.05,
            str="after propagate to nearby"
        )


    # ---- Final stage: handle polylines that still have no normals ----
    # These polylines must "borrow" a normal from nearby constrained edges.
    # Polylines that were not handled in the propagation stage

    # This will not happen if everything connects
    unconstrained_polylines = set(range(len(P))) - set(polyline_normals.keys())

    # For each unconstrained polyline, pick the best nearby edge normal
    borrowed_best_normals = {}
    for poly_idx in unconstrained_polylines:
        poly_edge_data = polyline_to_edge_map[poly_idx]
        borrowed_best_normals[poly_idx] = assign_normals_to_unconstrained_polylines(
            V, E, poly_edge_data, edge_constraints_map_updated, distances
        )

    # Visualization: the borrowed "anchor" normals
    debug.plot(
        plot_polyline_best_constraints,
        V, E, P,
        borrowed_best_normals,
        scale=0.05,
        str="borrow nearby edge normal"
    )

    debug.save(
        export_sketch_normal_gltf,
        V, E, P,
        convert_edge_dict_to_array(borrowed_best_normals, len(E), polyline_to_edge_map),
        unconstrained_polylines,
        f"debug_normals_gltf/borrowed_normal/{curve_name}.gltf"
    )

    # ---- Parallel-transport these borrowed normals along each unconstrained polyline ----
    unconstrained_polyline_normals = {}
    for poly_idx, (pos_in_polyline, normal) in borrowed_best_normals.items():
        poly_points = [V[v] for v in P[poly_idx]]
        normal_vectors = parallel_transport_bi_direction(poly_points, (pos_in_polyline, normal))
        unconstrained_polyline_normals[poly_idx] = normal_vectors

    # Debug: propagated borrowed normals
    debug.plot(
        plot_polyline_normals, V, E, P,
        unconstrained_polyline_normals,
        scale=0.05,
        str="parallel transport of borrowed normal"
    )

    debug.save(
        export_sketch_normal_gltf,
        V, E, P,
        convert_edge_dict_to_array(unconstrained_polyline_normals, len(E), polyline_to_edge_map),
        unconstrained_polylines,
        f"debug_normals_gltf/borrowed_parallel_transport/{curve_name}.gltf"
    )

    # Merge borrowed-normals with previously computed ones
    polyline_normals = { **polyline_normals, **unconstrained_polyline_normals }

    # ---- Convert per-polyline normals → per-edge normal map ----
    edge_constraints_map_estimated = {}
    for poly_idx, normals in polyline_normals.items():
        edge_indices, _ = polyline_to_edge_map[poly_idx]
        for k, edge_id in enumerate(edge_indices):
            edge_constraints_map_estimated[edge_id] = normals[k]

    # Final sanity check
    assert len(polyline_normals) == len(P), "Some polylines still do not have normals"

    # Final debug plot
    debug.plot(
        plot_edge_constraints,
        V, E, P,
        edge_constraints_map_estimated,
        scale=0.05,
        str="initial estimate"
    )

    return edge_constraints_map_estimated

def assign_normals_to_edges(edge_norm_map, polyline_edges, normals):
    # Insert normals for all edges of a polyline
    for i, edge_id in enumerate(polyline_edges):
        edge_norm_map[edge_id] = normals[i]

def transport_normal_between_edges(V, E, src_edge, dst_edge, shared_vertex, normal):
    """
    Parallel transport a normal from src_edge → dst_edge across the shared vertex.
    """

    s0, s1 = E[src_edge]
    d0, d1 = E[dst_edge]

    # find the opposite vertices
    src_other = s0 if s0 != shared_vertex else s1
    dst_other = d0 if d0 != shared_vertex else d1

    # local edge directions
    v_src = V[src_other] - V[shared_vertex]
    v_dst = V[shared_vertex] - V[dst_other]

    R = rotation_matrix_from(v_src, v_dst)
    return R @ normal

def estimate_initial_thetas(Us, Vs, normals):
    '''
    Convert normal vectors to theta values using frame vectors as basis.
    
    Parameters:
    -----------
    Us : array-like
        First frame vector for each edge
    Vs : array-like
        Second frame vector for each edge
    normals : (n_edges, 3)
        Estimated per-edge normal. Unassigned edges should be zero.

        
    Returns:
    --------
    thetas : array-like
        Angles (in radians) for each edge in the frame coordinate system
    '''
    n_edges = len(Us)
    thetas = np.zeros(n_edges)

    for i in range(n_edges):
        n = normals[i]

        # Project normal n = cosθ·U + sinθ·V
        cosθ = np.dot(n, Us[i])
        sinθ = np.dot(n, Vs[i])

        thetas[i] = np.arctan2(sinθ, cosθ)

    return thetas

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

    rotation_mats = np.zeros((n_edges, n_edges, 3, 3))
    has_rot = np.zeros((n_edges, n_edges), dtype=bool)

    for i in range(n_edges):
        ei = E[i]
        set_ei = set(ei)

        for j in range(n_edges):
            if i == j:
                continue

            ej = E[j]
            shared = set_ei & set(ej)

            if len(shared) != 1:
                continue  # no shared vertex → no rotation

            shared_v = next(iter(shared))

            # Identify the "other" vertices
            ei_other = ei[0] if ei[1] == shared_v else ei[1]
            ej_other = ej[0] if ej[1] == shared_v else ej[1]

            # Construct vectors with consistent orientation
            vi = V[shared_v] - V[ei_other]
            vj = V[ej_other] - V[shared_v]

            # Compute rotation matrix ei → ej
            rotation_mats[i, j] = rotation_matrix_from(vi, vj)
            has_rot[i, j] = True

    return {
        "matrices": rotation_mats,
        "has_rotation": has_rot,
    }

def preprocess_edge_pair_data(distances, rotations_data, vertex_to_edges_map, mode = 'one'):
    """
    Preprocess geometric edge pair data into an optimized format for energy computation.
    Produces JAX-compatible arrays specifically designed for efficient computation.

    Parameters:
    - E: List of edge vertex indices
    - rotations_data: Dictionary with rotation data between edge pairs
    - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices

    Returns:
    - Dictionary with JAX arrays containing pre-processed edge pair data ready for computation
    """
    
    e1_normal_indices_dict = {}
    e2_normal_indices_dict = {}

    # curve edge pairwise 
    pairwise = set()

    if mode == 'one':
        pairwise = extract_pairwise_weight(V, E, distances)
    else:
    # if True:

        for vertex_index, edge_indices in vertex_to_edges_map.items():
            if len(edge_indices) == 2:
                ei0, ei1 = edge_indices
                
                e0 = E[ei0]
                e1 = E[ei1]

                e0_vec = compute_edge_tangent(V, e0)
                e1_vec = compute_edge_tangent(V, e1)

                dot_product = np.abs(np.dot(e0_vec, e1_vec))
                weight = dot_product * 0.5 + 0.5
                # weight = 1 # not much difference 
                pairwise.add((ei0, ei1, weight))
            
            elif len(edge_indices) > 2:

                sorted_edges = compute_edge_circulation_graph_laplacian(edge_indices, vertex_index, E, V)
                n_sorted_edges = len(sorted_edges)
                sorted_pairs = [(sorted_edges[i], sorted_edges[(i + 1) % n_sorted_edges]) for i in range(n_sorted_edges)]


                for ei0, ei1 in sorted_pairs:
                    e0 = E[ei0]
                    e1 = E[ei1]

                    # e0_vec = compute_edge_tangent(V, e0)
                    # e1_vec = compute_edge_tangent(V, e1)

                    # dot_product = np.abs(np.dot(e0_vec, e1_vec))
                    # # weight = dot_product * 0.5 + 0.5
                    weight = 1
                    pairwise.add((ei0, ei1, weight))
                    # Set up the high valence circulation order normal correspondence.
                    # The first edge in the circulation order pair compares its 0-th normal to the second edge's 1-th normal.
                    e1_normal_indices_dict[ (ei0, ei1) ] = 0
                    e2_normal_indices_dict[ (ei0, ei1) ] = 1

    # Extract pairwise data
    e1_indices = np.array([p[0] for p in pairwise])
    e2_indices = np.array([p[1] for p in pairwise])
    weights = np.array([p[2] for p in pairwise])

    # Pre-compute information about shared vertices and curve edges
    curve_edges = np.zeros(len(pairwise), dtype=bool)
    e1_normal_indices = np.zeros(len(pairwise), dtype=int)
    e2_normal_indices = np.zeros(len(pairwise), dtype=int)
    
    for i, (e1, e2, _) in enumerate(pairwise):
        # Unpack the circulation normal pairing information into a flat array
        if (e1, e2) in e1_normal_indices_dict:
            # print('e1_normal_indices_dict', e1_normal_indices_dict)
            assert (e1, e2) in e2_normal_indices_dict
            e1_normal_indices[i] = e1_normal_indices_dict[ (e1, e2) ]
            e2_normal_indices[i] = e2_normal_indices_dict[ (e1, e2) ]
        # Set up the valence 2 information.
        # TODO: Is this redundant with the above check?
        shared_vertex = frozenset(E[e1]) & frozenset(E[e2])
        if len(shared_vertex) == 1:
            vertex = next(iter(shared_vertex))
            if len(vertex_to_edges_map[vertex]) == 2:
                curve_edges[i] = True
        else:
            if mode == 'two':
                assert (e1, e2) in e1_normal_indices_dict


    # Extract rotation data for the specific edge pairs in pairwise
    has_rotation = np.array([rotations_data['has_rotation'][e1, e2] for e1, e2 in zip(e1_indices, e2_indices)])
    rotation_matrices = np.array([rotations_data['matrices'][e1, e2] for e1, e2 in zip(e1_indices, e2_indices)])

    return {
        'e1_indices': jnp.array(e1_indices),
        'e2_indices': jnp.array(e2_indices),
        'weights': jnp.array(weights),
        'e1_normal_indices': jnp.array(e1_normal_indices),
        'e2_normal_indices': jnp.array(e2_normal_indices),
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
    ## These two elements encode the circulation order for valence > 2 (when curve_edges[i] is false).
    e1_normal_indices = data['e1_normal_indices']
    e2_normal_indices = data['e2_normal_indices']

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

        ## High valence
        # Do I need to use the same for high valence pairs 

        e1_normal_index = e1_normal_indices[i]
        e2_normal_index = e2_normal_indices[i]
        
        # Curve edges: use min of diagonal/antidiagonal sum
        # Non-curve edges: use global minimum
        # potential problem : normal0 is being constrained and also being paired, they might be conflicting
        pair_energy = jnp.where(
            is_curve,
            weight * jnp.minimum(diagonal_sum, antidiagonal_sum),
            weight * costs[ e1_normal_index, e2_normal_index ]
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
    # E_constraint = compute_two_normal_constraint_energy(normals0, normals1, constraints)
    # E_constraint = compute_one_normal_constraint_energy(normals0, constraints) * 0.5
    # E_constraint += compute_one_normal_constraint_energy(normals1, constraints) * 0.5
    E_constraint = compute_one_normal_constraint_energy(normals0, constraints['normal0']) * 0.5
    E_constraint += compute_one_normal_constraint_energy(normals1, constraints['normal1']) * 0.5

    # ===== One Normal Energy =====
    
    # ===== Pairwise Energy =====
    E_pairwise = compute_two_normal_pairwise_energy(
        normals0, 
        normals1, 
        data
    )
    
    # Return weighted sum of energies
    # return 1e-2 * E_constraint + 1e4 * E_pairwise + 1e6 * E_one_normal
    return 1e0 * E_constraint + 1e0 * E_pairwise 

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
            

            # filename = f"mouse_figs/frame_{current_iter:04d}.png"

            # Update the plot for one-normal case
            plot_edge_constraints(
                V, E, P, normals_now, 
                unconstrained_polylines_indices=None,
                scale=0.05,
                str=f"One-Normal Optimization: Iteration {current_iter}",
                # filename= filename,
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
            
            
            # filename = f"mouse_figs/frame_{current_iter:04d}.png"

            # Update the plot for two-normal case
            plot_edge_constraints_two_normals(
                V, E, P, normals_now, 
                unconstrained_polylines_indices=None,
                scale=0.05,
                str=f"Two-Normal Optimization: Iteration {current_iter}",
                # filename= filename,
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
def optimize_normal_angles(thetas0, Us, Vs, edge_constraints, rotations_data, vertex_to_edges_map, distances, 
                      mode='two', one_normal=None, callback_fn=None):
    """
    Unified JAX-based optimization function for both one-normal and two-normal cases
    
    Parameters:
    - thetas0: Initial theta values for each edge
    - Us, Vs: Frame vectors for each edge
    - edge_constraints: List of tuples (edge_index, desired_normal_vector)
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

    print('mode', mode)
    

    if mode == 'one':
        # Convert edge_constraints to JAX-friendly format
        constraints_jax = prepare_edge_constraints(edge_constraints)
    else:
        # I think for 2, it is already jaxed
        constraints_jax = edge_constraints



    

    
    if mode == 'one':
        # One normal per edge

        # Pre-process data for faster computation using improved rotations
        data = preprocess_edge_pair_data(distances, rotations_data, vertex_to_edges_map, mode)

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
        data = preprocess_edge_pair_data(distances, rotations_data, vertex_to_edges_map, mode)

        num_edges = len(thetas0)
        thetas_2d = np.column_stack((thetas0, thetas0))
        # thetas_2d[:, 0] -= 1e-3  # Small perturbation for second normal
        thetas_2d[:, 1] += 1e-3  # Small perturbation for second normal
        initial_thetas = thetas_2d.flatten()
        
        # # TODO
        # # If an edge has zero constrained normals, the initial guess is fine.
        # # If an edge has two constrained normals, the initial guess is essentially overridden and doesn't matter.
        # # If an edge has exactly one constrained normal, let's update the initial guess so that the perturbation
        # # implies positive rather than negative curvature.
        # # This only applies to high-valence junctions where there is a "straight across" edge.
        # for edge_index in len(edges):
        #     e0_is_constrained = ( edge_index, 0 ) in corner_constraints
        #     e1_is_constrained = ( edge_index, 1 ) in corner_constraints
        #     if e0_is_constrained and not e1_is_constrained:
        #         # Extract the initial guess directly from the constraint for the constrained edge.
        #         thetas_2d[ edge_index, 0 ] = corner_constraints[ ( edge_index, 0 ) ]
        #         ## UPDATE: We need circulation order here. We need to know whether normal 1 is to the "right" or "left" or normal 0.
        #         ## UPDATE 2: Do we know that because we always circulate 0 to 1?
        #         thetas_2d[ edge_index, 1 ] = thetas_2d[ edge_index, 0 ] + 1e-3
        #     elif e1_is_constrained and not e0_is_constrained:
        #         # Extract the initial guess directly from the constraint for the constrained edge.
        #         thetas_2d[ edge_index, 1 ] = corner_constraints[ ( edge_index, 1 ) ]
        #         ## UPDATE: We need circulation order here. We need to know whether normal 1 is to the "right" or "left" or normal 0.
        #         ## UPDATE 2: Do we know that because we always circulate 0 to 1? No, because we circulated CW or CCW based on
        #         ##           the mean curvature at the vertex. Here, we either want to add or subtract the perturbation
        #         ##           depending on that order so that the edge flap ends up with positive curvature.
        #         ##           The solution is either to reverse the order of normal constraint assignment so that it's always
        #         ##           CCW or else recompute the circulation order here (only one endpoint can be high valence)
        #         ##           and compare it to the one normal guess.
        #         thetas_2d[ edge_index, 0 ] = thetas_2d[ edge_index, 1 ] - 1e-3


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


def vertex_valence_three_constraints(V, E, vertex_to_edges_map, estimate_normals):
    '''
    Computes corner constraints for vertices with valence 3.
    
    Parameters:
        - V: vertices
        - E: edges
        - vertex_to_edges_map: Dictionary mapping vertex indices to edge indices
        - estimate_normals: list of tuples (edge_index, normal_vector) for all edges with computed normals
    
    Returns:
        - Dictionary with separate constraints for normals0 and normals1:
          * 'normal0': Dictionary with 'indices' and 'normals' for first normal of each edge
          * 'normal1': Dictionary with 'indices' and 'normals' for second normal of each edge
    '''
    # Convert normals to dictionary for lookup
    estimate_normals = dict(estimate_normals)
    

    # For plotting: format (edge_idx, which_edge) -> normal
    plotting_normals = {}
    
    # Prepare the output format
    indices0 = []
    normals0 = []
    indices1 = []
    normals1 = []

    # Process each vertex with valence 3
    for vertex, edges in vertex_to_edges_map.items():

        if len(edges) >= 3:
            # I can dn do higher valence here
            # because edges in the circluar order 
            # every edge will have 2 from cross product 

            sorted_edges = compute_edge_circulation_graph_laplacian(edges, vertex, E, V)
            n_sorted_edges = len(sorted_edges)
            sorted_pairs = [(sorted_edges[i], sorted_edges[(i + 1) % n_sorted_edges]) for i in range(n_sorted_edges)]

            # running this to get normals_candidates
            for ei0, ei1 in sorted_pairs:
                e0 = E[ei0]
                e1 = E[ei1]

                e0_vec = compute_edge_tangent(V, e0)
                e1_vec = compute_edge_tangent(V, e1)

                normal = np.cross(e0_vec, e1_vec)
                norm = np.linalg.norm(normal)
                # TODO Q: `norm` is sine of the angle between the tangents. Is this a good parallel threshold?
                # Skip normals whose norm is below threshold.
                if norm < np.sin(np.radians(5)):  # Skip tangents less than N degrees apart
                    # Skip this normal
                    continue
                
                normal = normal / norm

                # Ensure consistent orientation with estimated normals (average ei0 and ei1)
                # TODO Q: The same normal gets assigned to two edges. Is it possible that we negate the normal inconsistently?
                if np.dot(normal, estimate_normals[ei0] + estimate_normals[ei1]) < 0:
                    normal = -normal
                
                indices0.append( ei0 )
                normals0.append( normal )
                plotting_normals[(ei0,0)] = normal

                indices1.append( ei1 )
                normals1.append( normal )
                plotting_normals[(ei1,1)] = normal

    debug.log('indices0', indices0)
    debug.log('normals0', normals0)
    debug.log('indices1', indices1)
    debug.log('normals1', normals1)

    # Return formats for computation and plotting
    return {
        'normal0': {
            'indices': np.array(indices0),
            'normals': np.array(normals0)
        },
        'normal1': {
            'indices': np.array(indices1),
            'normals': np.array(normals1)
        },
        'plotting_normals': plotting_normals
    }

def group_normals_inplace(normals_dict):
    """
    Groups normals by swapping in-place within the dictionary.
    
    Args:
        normals_dict: Dictionary with keys (edge_idx, normal_idx) and normal vectors as values
    
    Returns:
        tuple: (swapped_count, n1_list, n2_list)
            - swapped_count: number of edges where normals were swapped
            - n1_list: list of all first normals (after grouping)
            - n2_list: list of all second normals (after grouping)
    """
    import numpy as np
    
    # Get all unique edge indices from dictionary keys
    edge_indices = set(key[0] for key in normals_dict.keys())
    
    # Get reference from first edge's first normal
    reference = normals_dict[(0, 0)] / np.linalg.norm(normals_dict[(0, 0)])
    
    swapped_count = 0
    n1_list = []
    n2_list = []
    
    for edge_idx in sorted(edge_indices):  # Sort for consistent ordering
        n1 = normals_dict[(edge_idx, 0)]
        n2 = normals_dict[(edge_idx, 1)]
        
        # Calculate similarities
        n1_sim = np.abs(np.dot(n1 / np.linalg.norm(n1), reference))
        n2_sim = np.abs(np.dot(n2 / np.linalg.norm(n2), reference))
        
        # Swap if needed
        if n2_sim > n1_sim:
            normals_dict[(edge_idx, 0)] = n2
            normals_dict[(edge_idx, 1)] = n1
            swapped_count += 1
            # Add swapped normals to lists
            n1_list.append(n2)
            n2_list.append(n1)
        else:
            # Add original normals to lists
            n1_list.append(n1)
            n2_list.append(n2)
    
    return swapped_count, n1_list, n2_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optimize edges to get normals")

    parser.add_argument("curve_file", help="Curve sketch to load")
    parser.add_argument("normal_file", nargs='?', help="Curve sketch with optimized normal information")
    parser.add_argument("gltf_file", nargs="?",default=None,help="Output normal glTF file")
    
    parser.add_argument(
        "-p", "--normal-per-edge", type=int, choices=[1, 2], default=2,
        help="Number of normals per edge (1 or 2)"
    )

    parser.add_argument(
        "--show-plot", action="store_true",
        default=False, # plot default
        help="Show visualization plots"
    )

    parser.add_argument(
        "--save-debug-gltf", action="store_true", 
        default = False, # not sae default
        help="Save debug glTF files"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Print debug logs"
    )

    args = parser.parse_args()

    curve_file = args.curve_file
    normal_file = args.normal_file
    gltf_file = args.gltf_file

    normals_per_edge = {1: "one", 2: "two"}[args.normal_per_edge]

    debug = DebugOptions(
        show_plot=args.show_plot,
        save_gltf=args.save_debug_gltf,
        verbose=args.verbose
    )
    


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
    debug.plot(plot_sketch_data, V, P)
    # debug.plot(plot_edge_info, V, E)
        # plot_edge_info(V, E)

    #####################################
    #endregion


    #region Initial normal estimate
    #####################################

    # edge normal from convex hull
    # edge_constraints : [ (edge_idx, normal) ]
    # only the ones with normal will be in the list 
    edge_constraints = get_sketch_edge_constraints(V, E, tol=2e-2, epsilon=5e-3)
    # print('edge_constraints', edge_constraints)
    debug.log('len(edge_constraints)', len(edge_constraints))
    
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
    debug.plot(plot_edge_constraints, V, E, P, edge_constraints, scale=0.05, str = 'edge constraints from convex hull', block= True)
    debug.save(export_sketch_normal_gltf,V, E, P, edge_constraints, unconstrained_polylines_indices= unconstrained_polylines_indices, filename ='debug_normals_gltf/edge_normals/' + curve_name + '.gltf' )



    debug.log('constrained_polyline_indices', constrained_polyline_indices)
    debug.log('unconstrained_polylines_indices', unconstrained_polylines_indices)

    vertex_to_edges_map = build_vertex_to_edges_map( E )

    debug.log('vertex_to_edges_map', vertex_to_edges_map)
    # compute distance between edges 
    
    distances = edge_distance_matrix(V, E)

    estimate_normals = estimate_initial_normals(V, E, P, polyline_to_edge_map, edge_to_polyline_map, edge_constraints_map, distances, vertex_to_edges_map)

    debug.save(export_sketch_normal_gltf,V, E, P, convert_edge_normals_to_array(estimate_normals, len(E)), unconstrained_polylines_indices , filename = 'debug_normals_gltf/initial_estimate/' + curve_name + '.gltf')




    # computes local coordinate frames for edges by generating parallel transport 
    # frames along polylines and mapping them to the global edge indices
    Us, Vs = create_frames_for_each_polyline( V, E, P )
    
    # show the frame on each edge
    debug.plot(plot_edge_frames, V, E, P, Us, Vs, scale=0.05)
    
    thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)
    # print('thetas0', thetas0)





    #####################################
    #endregion


  

    # print('weight_matrix', weight_matrix)
    # Create JAX-friendly edge rotation map
    rotations_data = precompute_edge_rotation_map(V, E)

    #region Optimization 
    #####################################

    if debug.show_plot:
        callback_fn = create_callback(Us, Vs, E, P, V, mode=normals_per_edge)
    else:
        callback_fn = None

    one_normal = []

    if normals_per_edge =='one':
        result = optimize_normal_angles(
            thetas0, Us, Vs, edge_constraints, rotations_data, vertex_to_edges_map, distances,
            mode= normals_per_edge , one_normal=one_normal, callback_fn=callback_fn
        )
        normals = recover_normals(result, Us, Vs, mode=normals_per_edge)
    # # if there're 2 normal per edge, then use the optimized 1 normal as 
    # # use opt 1 normal as starting normal
    elif normals_per_edge =='two':

        result = optimize_normal_angles(
            thetas0, Us, Vs, edge_constraints, rotations_data, vertex_to_edges_map, distances,
            mode= 'one' , one_normal=one_normal, callback_fn=None
        )

        estimate_normals = recover_normals(result, Us, Vs, mode='one')
        # still want to save this one normal optimization result for debug
        debug.save( export_sketch_normal_gltf, V, E, P, estimate_normals, unconstrained_polylines_indices, filename = f"debug_normals/{curve_name}_1n.gltf")
        debug.save( write_normal_data, V, E, estimate_normals, filename = f'debug_normals/{curve_name}_1n.normal')

        # two_normals_format = {}
        # for idx, normal in estimate_normals.items():
        #     two_normals_format[(idx,0)] = normal
        #     two_normals_format[(idx,1)] = normal

        # write_two_normal(V, E, two_normals_format, filename = f'debug_normals/{curve_name}_1n.normal')

        debug.plot(plot_edge_constraints, 
                V, E, P, estimate_normals, unconstrained_polylines_indices=None, 
                scale=0.05, str="One-Normal Optimization Result", block=True
            )
            
        corner_constraints = vertex_valence_three_constraints(V, E, vertex_to_edges_map, estimate_normals)

        debug.log('corner_constraints', corner_constraints)

        
        debug.plot(plot_edge_constraints_two_normals, V, E, P, corner_constraints['plotting_normals'], unconstrained_polylines_indices=None, str = 'corner constraints', block=True)
            # from utility_plot_viewer import plot_constraints_around_vertex
            # plot_constraints_around_vertex(39, V, E, P, corner_constraints['plotting_normals'], unconstrained_polylines_indices=None, scale=0.08, str_title=None, filename=None, block=True)
        
        
        thetas0 = estimate_initial_thetas(Us, Vs, estimate_normals)
        edge_constraints = corner_constraints


        result = optimize_normal_angles(
            thetas0, Us, Vs, edge_constraints, rotations_data, vertex_to_edges_map,
            normals_per_edge, one_normal=one_normal, callback_fn=callback_fn
        )

        normals = recover_normals(result, Us, Vs, mode=normals_per_edge)







    normals_str = f"{args.normal_per_edge}n"  # 1n or 2n for normals

    filename = f"{curve_name}_{normals_str}"
    # Create filename
    filename = f"{curve_name}_{normals_str}"

    if not gltf_file:
        gltf_file = 'data/normal/' + filename + '.gltf'
    if not normal_file:
        normal_file = 'data/normal/' + filename + '.normal'

    
    

    if normals_per_edge == 'one':
        debug.plot( plot_edge_constraints, 
                V, E, P, normals, unconstrained_polylines_indices=None, 
                scale=0.05, str="One-Normal Optimization Result", block=True
            )
        
        debug.save( export_sketch_normal_gltf, V, E, P, normals, unconstrained_polylines_indices, filename = gltf_file)
        debug.save( write_normal_data, V, E, normals, normal_file)

    else:
        # swap the normals
        _, n1, n2 = group_normals_inplace(normals)
        debug.plot( plot_edge_constraints_two_normals,
                V, E, P, normals, unconstrained_polylines_indices=None, 
                scale=0.05, str="Two-Normal Optimization Result", block=True
            )


        debug.plot( plot_edge_constraints_two_normals,
                V, E, P, normals, unconstrained_polylines_indices=None, 
                scale=0.05, str="Grouped normal", block=True
            )

        debug.plot( plot_edge_constraints_two_normals,
                V, E, P, normals, unconstrained_polylines_indices=None, 
                scale=0.05, block=True, filename="sewing_machine_normal", 
            )

        normals_view_0 = {k: v if k[1] == 0 else np.zeros(3) for k, v in normals.items()}
        normals_view_1 = {k: v if k[1] == 1 else np.zeros(3) for k, v in normals.items()}



            # plot_edge_constraints_two_normals(
            #     V, E, P, normals_view_0, unconstrained_polylines_indices=None, 
            #     scale=0.05, str="Grouped normal", block=True
            # )

            # plot_edge_constraints_two_normals(
            #     V, E, P, normals_view_1, unconstrained_polylines_indices=None, 
            #     scale=0.05, str="Grouped normal", block=True
            # )

            # plot_two_normals(V, E, normals)

        debug.save(export_sketch_normal_gltf, V, E, P, n1, unconstrained_polylines_indices=None , filename='debug_normals/' + curve_name + '_n0.gltf', arrow_color=(0, 1, 0) )
        debug.save(export_sketch_normal_gltf, V, E, P, n2, unconstrained_polylines_indices=None , filename='debug_normals/' + curve_name + '_n1.gltf', arrow_color= (1, 0, 0) )
        



        debug.save(export_sketch_two_normal_gltf, V, E, P, normals, unconstrained_polylines_indices, filename='debug_normals/' + curve_name + '_2n.gltf')

        write_two_normal(V, E, normals , normal_file)

        # if save_debug_gltf:
        #     export_sketch_two_normal_gltf(V, E, P, normals, unconstrained_polylines_indices, filename='debug_normals_gltf/final_optimize_two_normals/' + curve_name + '.gltf')


        # if save_debug_gltf:
        #     export_sketch_normal_gltf(V, E, P, N_normalized, unconstrained_polylines_indices, filename ='debug_normals_gltf/final_optimize/' + curve_name + '.gltf')
        #     write_normal_data(V, E, N_normalized , 'debug_normals_gltf/final_optimize/' + curve_name + '.normal')

        # if show_plot:
        #     plot_edge_constraints(V, E, P,  opt_normals, unconstrained_polylines_indices, scale=0.08, str = "optimize result")

    #####################################
    #endregion
    


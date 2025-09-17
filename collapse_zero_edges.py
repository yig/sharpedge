# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "libigl",
#     "numpy",
#     "polyscope",
#     "scipy",
# ]
# ///
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import igl
from pprint import pprint
# Optional dependency
# import polyscope as ps

from utility_io import load_mesh_obj

def collapse_zero_edges( V, F, threshold=1e-6, return_max_displacement=False ):
    """
    Collapse edges in the mesh that are shorter than the specified threshold.
    
    Parameters:
    V : np.ndarray
        Vertex positions of the mesh.
    F : np.ndarray
        Faces of the mesh.
    threshold : float
        Length below which edges will be collapsed. Defaults to 1e-6.
    Returns:
    V_new : np.ndarray
        New vertex positions after collapsing edges.
    F_new : np.ndarray
        New faces after collapsing edges.
    """

    # Compute edge lengths
    edges = igl.edges(F)
    edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)

    # Find edges shorter than the threshold
    short_edges = edge_lengths < threshold

    # Create a mapping for vertices of short edges
    # Because we may collapse multiple edges that share vertices,
    # we need to ensure that we only keep one representative vertex
    # for each collection of collapsed edges.
    # We might end up in a situation where we collapse v2->v1->v0 and v3->v4->v5,
    # and then we need to collase the edge (v2,v3) as well.
    # To do this, we will use a union-find approach to ensure that
    # we always map to the root representative vertex.
    vertex_sets = DisjointSet()
    for edge in edges[short_edges]:
        v0, v1 = edge
        assert v0 != v1, "Edges must have distinct vertices."
        # It is a no-op to add the same vertex multiple times.
        vertex_sets.add(v0)
        vertex_sets.add(v1)
        vertex_sets.merge(v0, v1)

    # Re-map vertices based on the mapping.
    # Average subsets.
    max_displacement = 0.0
    vertex_map = np.arange(len(V))
    for subset in vertex_sets.subsets():
        assert len(subset) != 0, "Subset should not be empty."

        center = np.mean(V[list(subset)], axis=0)
        subset_max_displacement = np.max(np.linalg.norm(V[list(subset)] - center, axis=1))
        max_displacement = max(max_displacement, subset_max_displacement)

        ## Map all vertices in the subset to the root vertex index.
        for i in subset: vertex_map[i] = vertex_sets[i]
        ## Update the root vertex position to the average of the subset.
        V[vertex_sets[i]] = center
    
    # Remove unused vertices and re-map again to reflect that.
    vertices_unique, vertex_map2 = np.unique( vertex_map, return_inverse=True )

    collapsed_V = V[vertices_unique]
    
    # Create new faces by remapping vertices
    remapped_F = np.vectorize(lambda x: vertex_map2[x])(F)
    
    # Skip collapsed faces
    collapsed_F = np.asarray([ face for face in remapped_F if len(frozenset(face)) == 3 ])
    # Handle the case where all edges are collapsed
    if len(collapsed_F) == 0: collapsed_F = np.empty((0, 3), dtype=int)

    if return_max_displacement:
        return collapsed_V, collapsed_F, max_displacement
    else:
        return collapsed_V, collapsed_F

def degenerate_statistics( V, F, short_edge_threshold=1e-6 ):
    """
    Compute statistics about degenerate faces in the mesh.
    
    Parameters:
    V : np.ndarray
        Vertex positions of the mesh.
    F : np.ndarray
        Faces of the mesh.
    
    Returns:
    stats : dict
        Dictionary containing statistics about degenerate faces.
    """
    E = igl.edges(F)
    edge_lengths = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    
    # Count degenerate edges
    short_edges = np.sum(edge_lengths <= short_edge_threshold)
    
    # Euler characteristic (V - E + F)
    euler = len(V) - len(E) + len(F)

    # Count edges with more than two faces
    edge_flap_counts = {}
    # Count faces with repeated vertices
    faces_with_repeated_vertices = 0
    for face in F:
        vs = frozenset(face)

        assert len(face) == 3, "Only triangular faces are supported."

        # Check for repeated vertices
        if len( vs ) != 3: faces_with_repeated_vertices += 1

        i, j, k = sorted(face)

        edge_flap_counts.setdefault((i, j), 0)
        edge_flap_counts.setdefault((j, k), 0)
        edge_flap_counts.setdefault((i, k), 0)

        edge_flap_counts[(i, j)] += 1
        edge_flap_counts[(j, k)] += 1
        edge_flap_counts[(i, k)] += 1
    
    edge_flap_histo = {}
    for x in edge_flap_counts.values():
        edge_flap_histo.setdefault(x, 0)
        edge_flap_histo[x] += 1

    return {
        'short_edge_threshold': short_edge_threshold,
        'short_edges': short_edges,
        'total_edges': len(E),
        'short_edge_ratio': short_edges / len(E) if len(E) > 0 else 0.0,
        'euler_characteristic': euler,
        'faces_with_repeated_vertices': faces_with_repeated_vertices,
        'edge_flap_histo': edge_flap_histo,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description = 'Collapse zero-length edges' )
    parser.add_argument( 'mesh', help = 'Path to mesh .obj file' )
    ## An optional threshold for edge length below which edges will be collapsed
    parser.add_argument( '--threshold', type=float, default=1e-6, help='Threshold for edge length below which edges will be collapsed.' )
    ## An optional argument to specify the output file name
    parser.add_argument( '--output', type=str, help='Output file name for the decimated mesh. Default: input-collapsed.obj' )
    ## An optional argument to show with polyscope
    parser.add_argument( '--show', action='store_true', help='Show the mesh with Polyscope.' )
    args = parser.parse_args()
    
    V, F = load_mesh_obj( args.mesh )
    
    V = np.asarray( V )
    F = np.asarray( F )

    print( "Loaded mesh with {} vertices and {} faces.".format( len(V), len(F) ) )
    print( "Degenerate statistics before collapsing:" )
    pprint( degenerate_statistics( V, F, short_edge_threshold=args.threshold ) )
    
    collapsed_V, collapsed_F, max_displacement = collapse_zero_edges(V, F, args.threshold, return_max_displacement=True)

    print( "Collapsed mesh has {} vertices and {} faces.".format(
        len(collapsed_V), len(collapsed_F) ) )
    print( "Maximum displacement of collapsed vertices:", max_displacement )
    print( "Degenerate statistics after collapsing:" )
    pprint( degenerate_statistics( collapsed_V, collapsed_F, short_edge_threshold=args.threshold ) )

    if( len( collapsed_F ) == 0 ): print( "WARNING: No faces left after collapsing edges." )

    if not args.output: args.output = args.mesh.rsplit('.obj',1)[0] + '_collapsed.obj'
    igl.writeOBJ(args.output, collapsed_V, collapsed_F)
    print(f"Collapsed mesh saved to: {args.output}")

    if args.show:
        # Optional: Show the mesh with Polyscope
        import polyscope as ps
        ps.init()
        ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)
        ps_mesh = ps.register_surface_mesh("Collapsed Mesh", collapsed_V, collapsed_F)
        ps.show()

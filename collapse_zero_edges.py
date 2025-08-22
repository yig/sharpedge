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
# Optional dependency
# import polyscope as ps

from utility_io import load_mesh_obj

def collapse_zero_edges( V, F, threshold=1e-6 ):
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
    vertex_map = np.arange(len(V))
    for i in range(len(V)):
        try:
            vertex_map[i] = vertex_sets[i]
        except KeyError:
            # If the vertex is not in the disjoint set, it wasn't part of a short edge.
            pass
    
    # Remove unused vertices and re-map again to reflect that.
    vertices_unique, vertex_map2 = np.unique( vertex_map, return_inverse=True )

    collapsed_V = V[vertices_unique]
    
    # Create new faces by remapping vertices
    remapped_F = np.vectorize(lambda x: vertex_map2[x])(F)
    
    # Skip collapsed faces
    collapsed_F = np.asarray([ face for face in remapped_F if len(frozenset(face)) == 3 ])

    return collapsed_V, collapsed_F

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
    
    collapsed_V, collapsed_F = collapse_zero_edges(V, F, args.threshold)

    print( "Collapsed from {} vertices and {} faces to {} vertices and {} faces.".format(
        len(V), len(F), len(collapsed_V), len(collapsed_F) ) )

    if not args.output: args.output = args.mesh.rsplit('.obj',1)[0] + '-collapsed.obj'
    igl.writeOBJ(args.output, collapsed_V, collapsed_F)
    print(f"Collapsed mesh saved to: {args.output}")

    if args.show:
        # Optional: Show the mesh with Polyscope
        import polyscope as ps
        ps.init()
        ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)
        ps_mesh = ps.register_surface_mesh("Collapsed Mesh", collapsed_V, collapsed_F)
        ps.show()

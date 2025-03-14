import numpy as np

def check_duplicates(V, E, P):
    # Check for duplicate vertices
    # Round vertices to some precision to handle floating point comparisons
    V_rounded = np.round(V, decimals=6)
    unique_V, inverse_indices = np.unique(V_rounded, axis=0, return_inverse=True)
    duplicate_vertices = len(V) != len(unique_V)
    
    # Check for duplicate edges (assuming E is already sorted)
    unique_E = np.unique(E, axis=0)
    duplicate_edges = len(E) != len(unique_E)
    
    # Check for duplicate polylines
    duplicate_polylines = False
    unique_polylines = set()
    for polyline in P:
        # Convert to tuple for hashing
        polyline_tuple = tuple(polyline)
        if polyline_tuple in unique_polylines:
            duplicate_polylines = True
            break
        unique_polylines.add(polyline_tuple)
    
    return {
        'duplicate_vertices': duplicate_vertices,
        'duplicate_edges': duplicate_edges,
        'duplicate_polylines': duplicate_polylines
    }


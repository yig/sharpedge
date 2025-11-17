import numpy as np

def compute_edge_tangent(V, edge):
    # Compute normalized tangent vector for an edge
    e0, e1 = edge 
    tangent = V[e1] - V[e0]
    assert np.linalg.norm(tangent) != 0
    return tangent / np.linalg.norm(tangent)


def are_parallel_cos(v1, v2, cos_threshold=np.cos(np.radians(10))):
    """Check if two vectors are parallel using cosine threshold"""
    # Compute normalized dot product (cosine of angle between vectors)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Check if angle is less than 15 degrees (cosine value greater than threshold)
    return abs(cos_angle) >= cos_threshold


def random_normal_for_edge( U, V) :
    '''
    create a random normal for edge, given the frame U and V.
    This normal will always be perpendicular to the edge
    '''
    return normal_for_edge( np.random.uniform(0, 2 * np.pi), U, V )   


def normal_for_edge( theta, U, V):  return np.cos( theta ) * U + np.sin( theta ) * V

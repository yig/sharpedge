import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import polyscope as ps
from utility_io import write_two_normal
import argparse
from utility_io import read_two_normal
from utility_viewer_ps import plot_two_normals

# Convert the dictionary format normals to arrays for plotting
def convert_normals_for_plotting(E, normals):
    """
    Convert normals from dictionary format to two arrays for plotting.
    
    Args:
        E: (m,2) array of edge vertex pairs
        normals: Dictionary with keys (edge_idx, which_edge)
        
    Returns:
        N1: (m,3) array of first normal vectors
        N2: (m,3) array of second normal vectors
    """
    m = len(E)
    N1 = np.zeros((m, 3))
    N2 = np.zeros((m, 3))
    
    for i in range(m):
        if (i, 0) in normals:
            N1[i] = normals[(i, 0)]
        if (i, 1) in normals:
            N2[i] = normals[(i, 1)]
    
    return N1, N2


# Even simpler one-liner version:
def one_liner_grouping_return_stats(N1, N2):
    """
    Version that returns the statistics as well as the arrays.
    """
    reference = N1[0] / np.linalg.norm(N1[0])
    N1_new, N2_new = N1.copy(), N2.copy()
    
    swapped_count = 0
    
    for i in range(len(N1)):
        n1_sim = np.abs(np.dot(N1[i] / np.linalg.norm(N1[i]), reference))
        n2_sim = np.abs(np.dot(N2[i] / np.linalg.norm(N2[i]), reference))
        
        if n2_sim > n1_sim:  # N2[i] is closer to reference, swap
            N1_new[i], N2_new[i] = N2[i], N1[i]
            swapped_count += 1
    
    stats = {
        'total_edges': len(N1),
        'swapped': swapped_count,
        'kept': len(N1) - swapped_count,
        'swap_percentage': swapped_count/len(N1)*100
    }
    
    return N1_new, N2_new, stats


parser = argparse.ArgumentParser(description='Edge normal file to point normal file')

# Add arguments
parser.add_argument('normal_file', nargs='?',
                    help='Input file containing normal data (.normal)')

args = parser.parse_args()

normal_file = args.normal_file


V, E, N = read_two_normal(normal_file)
N0, N1 = convert_normals_for_plotting(E, N)

# Show both plots for comparison
print("Original plot (by order):")
plot_two_normals(V, E, N0, N1)

N0_grouped, N1_grouped, stats = one_liner_grouping_return_stats(N0, N1)
    
print(f"Swapped {stats['swapped']} out of {stats['total_edges']} edges")
print(f"That's {stats['swap_percentage']:.1f}% of all edges")
plot_two_normals(V, E, N0_grouped, N1_grouped)
import argparse


from utility_io import load_normal_data
from utility_viewer_ps import plot_two_normals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To view normal difference.')
    parser.add_argument('normal_file_1', nargs='?', help='The curve sketch with optimized normal information.')
    parser.add_argument('normal_file_2', nargs='?', help='The curve sketch with optimized normal information.')

    args = parser.parse_args()

    normal_file_1 = args.normal_file_1
    normal_file_2 = args.normal_file_2

    V, E, N1 = load_normal_data(normal_file_1)
    V, E, N2 = load_normal_data(normal_file_2)

    plot_two_normals(V, E, N1, N2)

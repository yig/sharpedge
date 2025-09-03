#pragma once

/// Cuts specified edges of a mesh.
/// The function cuts the mesh along those edges by duplicating the vertices at either end of the edge and attaches the triangles on one side to the new vertices.
/// The number of faces does not change.
/// The number of vertices will increase (unless no edges are cut).
/// The same vertex can appear in multiple edges.
///
/// @param[in] V  #V by dim vertex positions
/// @param[in] F  #F by 3 triangle indices into V
/// @param[in] cut_edges  #edges by 2 pairs of vertex indices defining the edges to cut along
/// @param[out] V_out  #V' by dim vertex positions. The rows of V and V_out refer to the same vertices unless a vertex was duplicated. In that case, the new vertex appears at the end of V_out.
/// @param[out] F_out  #F by #3 triangle indices into V_out. The rows of F and F_out refer to the same triangles with vertices in the same order. Some of the vertex indices in F_out may refer to new vertices in V_out.

#include <Eigen/Core>
#include <unordered_set>

void cut_mesh_along_edges(
    // Each row is a d-dimensional vertex.
    const Eigen::MatrixXd& V,
    // Each row is a triplet of vertex indices defining a triangle.
    const Eigen::MatrixXi& F,
    // Each row is a pair of vertex indices defining an edge to cut along.
    const Eigen::MatrixXi& cut_edges,
    // The rows of V and V_out refer to the same vertices unless a vertex was duplicated. In that case, the new vertex appears at the end of V_out.
    Eigen::MatrixXd& V_out,
    // The rows of F and F_out refer to the same triangles with vertices in the same order.
    Eigen::MatrixXi& F_out
);

namespace {
/// A union-find data structure.
// MIT-licensed code source:
// <https://stackoverflow.com/questions/8300125/union-find-data-structure>
// and, originally
// <https://github.com/kartikkukreja/blog-codes/blob/master/src/Union%20Find%20%28Disjoint%20Set%29%20Data%20Structure.cpp>
class UF {
  int *id{}, cnt{}, *sz{};
  public:
// Create an empty union find data structure with N isolated sets.
UF(int N) {
    cnt = N; id = new int[N]; sz = new int[N];
    for (int i = 0; i<N; i++)  id[i] = i, sz[i] = 1; }
~UF() { delete[] id; delete[] sz; }

// Return the id of component corresponding to object p.
int find(int p) {
    int root = p;
    while (root != id[root])    root = id[root];
    while (p != root) { int newp = id[p]; id[p] = root; p = newp; }
    return root;
}
// Replace sets containing x and y with their union.
void merge(int x, int y) {
    int i = find(x); int j = find(y); if (i == j) return;
    // make smaller root point to larger one
    if (sz[i] < sz[j]) { id[i] = j, sz[j] += sz[i]; }
    else { id[j] = i, sz[i] += sz[j]; }
    cnt--;
}
// Are objects x and y in the same set?
bool connected(int x, int y) { return find(x) == find(y); }
// Return the number of disjoint sets.
int count() { return cnt; }
};
}

void cut_mesh_along_edges(
    // Each row is a d-dimensional vertex.
    const Eigen::MatrixXd& V,
    // Each row is a triplet of vertex indices defining a triangle.
    const Eigen::MatrixXi& F,
    // Each row is a pair of vertex indices defining an edge to cut along.
    const Eigen::MatrixXi& cut_edges,
    // The rows of V and V_out refer to the same vertices unless a vertex was duplicated. In that case, the new vertex appears at the end of V_out.
    Eigen::MatrixXd& V_out,
    // The rows of F and F_out refer to the same triangles with vertices in the same order.
    Eigen::MatrixXi& F_out
)
{
    /// The function is implemented using a union-find data structure.
    /// 1 First, we create a set of the cut edges for fast lookup.
    /// 2 We create a vertex set for each corner of the triangle.
    /// 3 We iterate over the edges and for each edge we find the triangles that share that edge.
    /// 4 We union the vertex sets of the corners of those triangles, so long as the edge is not a cut edge.
    /// 5 Finally, we create a new vertex for each vertex set and update the triangle indices to point to the new vertices. At least one of the triangle corners will use the original vertex, so we need to keep track of which vertex sets have been assigned a new vertex and which have not.

    assert(F.cols() == 3);
    assert(cut_edges.cols() == 2);
    
    // I don't think we require manifold-ness, since we completely split the mesh and re-build it.
    // There is a place below for a manifold assertion if we want it.
    // assert( is_manifold(V, F) );
    
    auto vertex_pair_hash = [&]( const std::pair<int, int>& edge ) {
        return edge.first * V.rows() + edge.second;
    };
    
    // 1 Create a set of edges as pairs of indices with the smaller index first.
    std::unordered_set<std::pair<int, int>, decltype(vertex_pair_hash)> cut_edge_set( cut_edges.rows(), vertex_pair_hash );
    for (int cut_edge_index = 0; cut_edge_index < cut_edges.rows(); ++cut_edge_index) {
        int v0 = cut_edges(cut_edge_index, 0);
        int v1 = cut_edges(cut_edge_index, 1);
        if (v0 > v1) std::swap(v0, v1);
        cut_edge_set.insert({v0, v1});
    }
    
    // 2 Create a union-find data structure with 3 * F.rows() elements, one for each corner of each triangle.
    UF uf(3 * F.rows());
    auto corner = [&](int f, int c) { return 3 * f + c; }; // corner index for face f and corner c (0, 1, or 2)
    auto vertex_of_corner = [&](int corner) { return F(corner / 3, corner % 3); }; // vertex index for corner index
    auto face_of_corner = [&](int corner) { return corner / 3; }; // face index for corner index
    auto corner_in_face = [&](int corner) { return corner % 3; }; // corner index (0, 1, or 2) for corner index

    // 3 Make a map from edges to faces that contain them.
    std::unordered_map<std::pair<int, int>, std::vector<int>, decltype(vertex_pair_hash)> edge_to_faces( cut_edges.rows(), vertex_pair_hash );
    // Iterate over triangles.
    for (int f = 0; f < F.rows(); ++f) {
        // Iterate over edges of the triangle.
        for (int c = 0; c < 3; ++c) {
            // Get the two vertices of the edge in canonical order (v0 < v1).
            int v0 = F(f, c);
            int v1 = F(f, (c + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            // Add the face to the list of faces for this edge. For a manifold mesh, this list should have length 2 (or 1 for boundaries).
            // We could assert that the length is at most 2, but I think our approach doesn't require manifold-ness.
            edge_to_faces[{v0, v1}].push_back(f);
        }
    }

    // 4 Merge corners of triangles that are not separated by a cut edge.
    // Iterate over the edges in the edge_to_faces map.
    for (const auto& item : edge_to_faces) {
        const auto& edge = item.first;
        const auto& faces = item.second;
        // If the edge is a cut edge, skip it.
        if (cut_edge_set.find(edge) != cut_edge_set.end()) { continue; }
        // If the edge is not a cut edge, union the corners of the faces that are not on the edge.
        // Get the two vertices of the edge.
        auto v0 = edge.first;
        auto v1 = edge.second;
        // Find a representative corner for v0 and v1.
        int v0_rep = -1;
        int v1_rep = -1;
        // Iterate over triangles incident to the edge.
        for (int f : faces) {
            // Iterate over corners of the triangle.
            for (int c = 0; c < 3; ++c) {
                // If the corner is v0 or v1, set the representative if not already set, otherwise merge the corner with the representative.
                int v = F(f, c);
                if (v == v0) {
                    if( v0_rep != -1 ) {
                        uf.merge(v0_rep, corner(f, c));
                    } else {
                        v0_rep = corner(f, c);
                    }
                } else if (v == v1) {
                    if( v1_rep != -1 ) {
                        uf.merge(v1_rep, corner(f, c));
                    } else {
                        v1_rep = corner(f, c);
                    }
                }
            }
            // We should have found representatives for both vertices after the first face.
            assert(v0_rep != -1 && v1_rep != -1);
        }
    }

    // 5 Create new vertices and update F_out.
    // Start with the original vertices.
    V_out = V;
    // Start counting new vertices after the original vertices.
    int new_vertex_count = V.rows();
    // We can create a maximum of two new vertices per cut edge, so reserve space for that.
    V_out.conservativeResize(V.rows() + 2 * cut_edges.rows(), Eigen::NoChange );
    
    // Update F_out as we go.
    F_out = F;
    
    // Create a map from root corner to new vertex index.
    std::unordered_map<int, int> root_to_vertex_index;
    // Since we want the original vertices to be used where possible, let's track which vertices have been used once.
    Eigen::VectorXi original_vertex_used = Eigen::VectorXi::Zero(V.rows());

    // Iterate over corners.
    for (int corner_index = 0; corner_index < 3 * F.rows(); ++corner_index) {
        int root = uf.find(corner_index);
        // If the root is not in the map, add it with a possibly new vertex index.
        if (root_to_vertex_index.find(root) == root_to_vertex_index.end()) {
            int v = vertex_of_corner(root);
            // If the original vertex has not been used yet, use it.
            if (original_vertex_used(v) == 0) {
                root_to_vertex_index[root] = v;
                original_vertex_used(v) = 1;
            } else {
                // Otherwise, create a new vertex.
                root_to_vertex_index[root] = new_vertex_count;
                V_out.row(new_vertex_count) = V.row(v);
                new_vertex_count++;
            }
        }
        // Update the face to use the new vertex index.
        int f = face_of_corner(corner_index);
        int c = corner_in_face(corner_index);
        F_out(f, c) = root_to_vertex_index[root];
    }
    // Resize V_out to the actual number of new vertices.
    V_out.conservativeResize(new_vertex_count, Eigen::NoChange );
}

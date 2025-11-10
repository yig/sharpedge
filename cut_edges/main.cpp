#include "cut_mesh_along_edges.h"
#include "edge_dual_normal.h"
#include "simple_obj_reader.h"

#include <iostream>
#include <set>

// Add a suffix before the file extension
std::string add_suffix_before_extension(const std::string& filename, const std::string& suffix) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return filename + suffix; // No file extension
    }
    return filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
}

// Map geometry edges (from EdgeDualNormalGeometry) to mesh edges based on vertex proximity
Eigen::MatrixXi mapGeometryEdgesToMesh(
    const EdgeDualNormalGeometry& geometry,
    const Eigen::MatrixXd& meshVertices,
    const Eigen::MatrixXi& meshFaces,
    double tolerance = 1e-5) {
    
    std::cout << "Mapping geometry edges to mesh with tolerance: " << tolerance << std::endl;
    
    // Build the edge list of the mesh
    std::vector<std::pair<int, int>> meshEdgeList;
    std::set<std::pair<int, int>> edgeSet;
    
    for (int f = 0; f < meshFaces.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int v1 = meshFaces(f, e);
            int v2 = meshFaces(f, (e + 1) % 3);
            
            // Ensure vertex indices are in ascending order
            if (v1 > v2) std::swap(v1, v2);
            
            if (edgeSet.find({v1, v2}) == edgeSet.end()) {
                edgeSet.insert({v1, v2});
                meshEdgeList.push_back({v1, v2});
            }
        }
    }
    
    std::cout << "Found " << meshEdgeList.size() << " unique mesh edges" << std::endl;
    
    // Store successfully mapped edges
    std::vector<std::pair<int, int>> mappedEdges;
    
    const auto& geomVertices = geometry.getVertices();
    const auto& geomEdges = geometry.getEdges();
    
    // Try to find a matching mesh edge for each geometry edge
    for (size_t i = 0; i < geomEdges.size(); ++i) {
        const auto& geomEdge = geomEdges[i];
        const Point& geomV1 = geomVertices[geomEdge.first];
        const Point& geomV2 = geomVertices[geomEdge.second];
        
        // Search for the closest mesh edge
        int bestEdgeIdx = -1;
        double bestDistance = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < meshEdgeList.size(); ++j) {
            int meshV1 = meshEdgeList[j].first;
            int meshV2 = meshEdgeList[j].second;
            
            Point meshP1 = meshVertices.row(meshV1);
            Point meshP2 = meshVertices.row(meshV2);
            
            // Compute distance for both endpoint orderings
            double dist1 = (geomV1 - meshP1).norm() + (geomV2 - meshP2).norm();
            double dist2 = (geomV1 - meshP2).norm() + (geomV2 - meshP1).norm();
            
            double minDist = std::min(dist1, dist2);
            
            if (minDist < bestDistance) {
                bestDistance = minDist;
                bestEdgeIdx = j;
            }
        }
        
        // Check if the match is within tolerance
        if (bestEdgeIdx >= 0 && bestDistance <= tolerance * 2.0) {
            mappedEdges.push_back(meshEdgeList[bestEdgeIdx]);
            
            std::cout << "Geometry edge " << i << ": ("
                      << geomV1.transpose() << ") -> ("
                      << geomV2.transpose() << ") mapped to mesh edge ("
                      << meshEdgeList[bestEdgeIdx].first << ", "
                      << meshEdgeList[bestEdgeIdx].second << ") with distance "
                      << bestDistance << std::endl;
        } else {
            std::cout << "Geometry edge " << i << ": No good match found (best distance: "
                      << bestDistance << ")" << std::endl;
        }
    }
    
    // Convert to Eigen matrix
    if (mappedEdges.empty()) {
        std::cout << "Warning: No edges were successfully mapped!" << std::endl;
        return Eigen::MatrixXi(0, 2);
    }
    
    Eigen::MatrixXi cutEdges(mappedEdges.size(), 2);
    for (size_t i = 0; i < mappedEdges.size(); ++i) {
        cutEdges(i, 0) = mappedEdges[i].first;
        cutEdges(i, 1) = mappedEdges[i].second;
    }
    
    std::cout << "Successfully mapped " << mappedEdges.size() << " edges" << std::endl;
    return cutEdges;
}

// Simple unit test
void test_case(){
    Eigen::MatrixXd V(4,3);
    V << 0,0,0,
         1,0,0,
         1,1,0,
         0,1,0;
    Eigen::MatrixXi F(2,3);
    F << 0,1,2,
         0,2,3;
    Eigen::MatrixXi cut_edges(1,2);
    cut_edges << 0,2;
    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;
    
    std::cout << "cut_edges " << cut_edges << std::endl;
    cut_mesh_along_edges(V, F, cut_edges, V_out, F_out);
    std::cout << "V_out:\n" << V_out << std::endl;
    std::cout << "F_out:\n" << F_out << std::endl;
    SimpleOBJReader::writeOBJ("cut_mesh.obj", V_out, F_out);
}

// Main function
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <normal_file.normal> <surface_file.obj> [-t targetEdgeLength]"
                  << std::endl;
        return 1;
    }

    const char* normal_file = argv[1];
    const char* surface_file = argv[2];

    EdgeDualNormalGeometry geometry;
    readEdgeDualNormal(normal_file, geometry);

    // Default value : 0.04f
    
    // This is used because when generate the surfaces.
    // The edges are resampled when creating the mesh, so need to resample again to match the mesh edges.
    float targetEdgeLength = 0.04f;
    
    // Parse optional argument
    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            targetEdgeLength = std::atof(argv[i + 1]);
            i++; // Skip next argument
        }
    }
    
    std::cout << "Using targetEdgeLength = " << targetEdgeLength << std::endl;

    EdgeDualNormalGeometry resampled;
    resampleEdgeDualNormalGeometry(geometry, resampled, targetEdgeLength);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    SimpleOBJReader::readOBJ(surface_file, V, F);

    Eigen::MatrixXi cut_edges = mapGeometryEdgesToMesh(resampled, V, F);

    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;

    cut_mesh_along_edges(V, F, cut_edges, V_out, F_out);
    std::cout << "V_out:\n" << V_out << std::endl;
    std::cout << "F_out:\n" << F_out << std::endl;

    std::string output_file = add_suffix_before_extension(surface_file, "_cut");
    SimpleOBJReader::writeOBJ(output_file, V_out, F_out);

    // test_case();
    return 0;
}

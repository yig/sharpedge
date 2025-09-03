#include "cut_mesh_along_edges.h"
#include "edge_dual_normal.h"
#include "simple_obj_reader.h"

// Example usage:
#include <iostream>
#include <set>

std::string add_suffix_before_extension(const std::string& filename, const std::string& suffix) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return filename + suffix; // 没有扩展名
    }
    return filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
}

Eigen::MatrixXi mapGeometryEdgesToMesh(
    const EdgeDualNormalGeometry& geometry,
    const Eigen::MatrixXd& meshVertices,
    const Eigen::MatrixXi& meshFaces,
    double tolerance = 1e-5) {
    
    std::cout << "Mapping geometry edges to mesh with tolerance: " << tolerance << std::endl;
    
    // 构建网格的边列表
    std::vector<std::pair<int, int>> meshEdgeList;
    std::set<std::pair<int, int>> edgeSet;
    
    for (int f = 0; f < meshFaces.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int v1 = meshFaces(f, e);
            int v2 = meshFaces(f, (e + 1) % 3);
            
            // 确保边的顶点索引按升序排列
            if (v1 > v2) std::swap(v1, v2);
            
            if (edgeSet.find({v1, v2}) == edgeSet.end()) {
                edgeSet.insert({v1, v2});
                meshEdgeList.push_back({v1, v2});
            }
        }
    }
    
    std::cout << "Found " << meshEdgeList.size() << " unique mesh edges" << std::endl;
    
    // 存储成功映射的边
    std::vector<std::pair<int, int>> mappedEdges;
    
    const auto& geomVertices = geometry.getVertices();
    const auto& geomEdges = geometry.getEdges();
    
    // 对每条几何边寻找匹配
    for (size_t i = 0; i < geomEdges.size(); ++i) {
        const auto& geomEdge = geomEdges[i];
        const Point& geomV1 = geomVertices[geomEdge.first];
        const Point& geomV2 = geomVertices[geomEdge.second];
        
        // 寻找最佳匹配的网格边
        int bestEdgeIdx = -1;
        double bestDistance = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < meshEdgeList.size(); ++j) {
            int meshV1 = meshEdgeList[j].first;
            int meshV2 = meshEdgeList[j].second;
            
            Point meshP1 = meshVertices.row(meshV1);
            Point meshP2 = meshVertices.row(meshV2);
            
            // 计算两种端点对应方式的距离
            double dist1 = (geomV1 - meshP1).norm() + (geomV2 - meshP2).norm();
            double dist2 = (geomV1 - meshP2).norm() + (geomV2 - meshP1).norm();
            
            double minDist = std::min(dist1, dist2);
            
            if (minDist < bestDistance) {
                bestDistance = minDist;
                bestEdgeIdx = j;
            }
        }
        
        // 检查是否在容差范围内
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
    
    // 转换为Eigen矩阵
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


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <normal_file> <surface_file.obj>" << std::endl;
        return 1;
    }

    const char* normal_file = argv[1];
    const char* surface_file = argv[2];


    EdgeDualNormalGeometry geometry;
    readEdgeDualNormal(normal_file, geometry);

    EdgeDualNormalGeometry resampled;
    resampleEdgeDualNormalGeometry(geometry, resampled, 0.05);

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    SimpleOBJReader::readOBJ(surface_file, V, F);


    Eigen::MatrixXi cut_edges = mapGeometryEdgesToMesh(resampled, V, F);

//    Eigen::MatrixXi cut_edges;
//    cut_edges << 6,7;
//
//    std::cout << cut_edges << std::endl;
    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;


    cut_mesh_along_edges(V, F, cut_edges, V_out, F_out);
    std::cout << "V_out:\n" << V_out << std::endl;
    std::cout << "F_out:\n" << F_out << std::endl;

    std::string output_file = add_suffix_before_extension(surface_file, "_cut");    
    SimpleOBJReader::writeOBJ(output_file, V_out, F_out);


//    test_case();
    return 0;
}

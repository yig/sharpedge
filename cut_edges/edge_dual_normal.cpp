// edge_dual_normal.cpp
#include "edge_dual_normal.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>

EdgeDualNormalGeometry::EdgeDualNormalGeometry() {}

bool EdgeDualNormalGeometry::isValid() const {
    return (edges_.size() == normals1_.size()) &&
           (edges_.size() == normals2_.size()) &&
           !vertices_.empty();
}

Scalar EdgeDualNormalGeometry::getEdgeLength(size_t edgeIdx) const {
    if (edgeIdx >= edges_.size()) {
        throw std::out_of_range("Edge index out of range");
    }
    
    const auto& edge = edges_[edgeIdx];
    const Point& v1 = vertices_[edge.first];
    const Point& v2 = vertices_[edge.second];
    
    return (v2 - v1).norm();  // Eigen's norm() function
}

Scalar EdgeDualNormalGeometry::getMeanEdgeLength() const {
    if (edges_.empty()) return 0.0;
    
    Scalar total = 0.0;
    for (size_t i = 0; i < edges_.size(); ++i) {
        total += getEdgeLength(i);
    }
    
    return total / edges_.size();
}

std::pair<Point, Point> EdgeDualNormalGeometry::getEdgeNormals(size_t edgeIdx) const {
    if (edgeIdx >= edges_.size()) {
        throw std::out_of_range("Edge index out of range");
    }
    return std::make_pair(normals1_[edgeIdx], normals2_[edgeIdx]);
}

void EdgeDualNormalGeometry::clear() {
    vertices_.clear();
    edges_.clear();
    normals1_.clear();
    normals2_.clear();
}

void EdgeDualNormalGeometry::printSummary() const {
    std::cout << "EdgeDualNormalGeometry Summary:" << std::endl;
    std::cout << "- " << nVertices() << " vertices" << std::endl;
    std::cout << "- " << nEdges() << " edges" << std::endl;
    std::cout << "- Valid: " << (isValid() ? "Yes" : "No") << std::endl;
    
    if (!edges_.empty()) {
        Scalar minLen = std::numeric_limits<Scalar>::max();
        Scalar maxLen = 0.0;
        Scalar meanLen = getMeanEdgeLength();
        
        for (size_t i = 0; i < edges_.size(); ++i) {
            Scalar len = getEdgeLength(i);
            minLen = std::min(minLen, len);
            maxLen = std::max(maxLen, len);
        }
        
        std::cout << "- Edge lengths: min=" << minLen
                  << ", max=" << maxLen
                  << ", mean=" << meanLen << std::endl;
    }
}

bool readEdgeDualNormal(const std::string& filename, EdgeDualNormalGeometry& geometry) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<Point> vertices;
    std::vector<std::pair<size_t, size_t>> edges;
    std::vector<Point> normals;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Scalar x, y, z;
            if (iss >> x >> y >> z) {
                vertices.push_back(Point(x, y, z));  // Eigen::Vector3d constructor
            }
        }
        else if (prefix == "l") {
            size_t i, j;
            if (iss >> i >> j) {
                edges.emplace_back(i - 1, j - 1); // Convert to 0-based
            }
        }
        else if (prefix == "vn") {
            Scalar nx, ny, nz;
            if (iss >> nx >> ny >> nz) {
                normals.push_back(Point(nx, ny, nz));  // Eigen::Vector3d constructor
            }
        }
    }

    file.close();

    // Validate
    if (edges.size() * 2 != normals.size()) {
        std::cerr << "Error: Expected " << edges.size() * 2
                  << " normals for " << edges.size() << " edges, but got "
                  << normals.size() << std::endl;
        return false;
    }

    // Split normals
    std::vector<Point> normals1, normals2;
    for (size_t i = 0; i < edges.size(); ++i) {
        normals1.push_back(normals[2 * i]);
        normals2.push_back(normals[2 * i + 1]);
    }

    // Set data
    geometry.setVertices(vertices);
    geometry.setEdges(edges);
    geometry.setNormals1(normals1);
    geometry.setNormals2(normals2);

    std::cout << "Successfully read from " << filename << std::endl;
    geometry.printSummary();

    return true;
}

bool resampleEdgeDualNormalGeometry(
    const EdgeDualNormalGeometry& source,
    EdgeDualNormalGeometry& target,
    Scalar targetEdgeLength) {
    
    const auto& sourceVertices = source.getVertices();
    const auto& sourceEdges = source.getEdges();
    const auto& sourceNormals1 = source.getNormals1();
    const auto& sourceNormals2 = source.getNormals2();
    
    if (sourceEdges.empty() || sourceVertices.empty()) {
        std::cerr << "Error: Source geometry is empty" << std::endl;
        return false;
    }
    
    if (targetEdgeLength <= 0.0) {
        std::cerr << "Error: Target edge length must be positive" << std::endl;
        return false;
    }
    
    std::vector<Point> newVertices = sourceVertices;
    std::vector<std::pair<size_t, size_t>> newEdges;
    std::vector<Point> newNormals1;
    std::vector<Point> newNormals2;
    
    for (size_t edgeIdx = 0; edgeIdx < sourceEdges.size(); ++edgeIdx) {
        const auto& edge = sourceEdges[edgeIdx];
        const Point& startVertex = sourceVertices[edge.first];
        const Point& endVertex = sourceVertices[edge.second];
        const Point& normal1 = sourceNormals1[edgeIdx];
        const Point& normal2 = sourceNormals2[edgeIdx];
        
        Scalar currentEdgeLength = (endVertex - startVertex).norm();  // Eigen's norm()
        
        if (currentEdgeLength < 1e-6) {
            continue; // Skip degenerate edges
        }
        
        int numSegments = static_cast<int>(std::ceil(currentEdgeLength / targetEdgeLength));
        numSegments = std::max(1, numSegments);
        
        size_t currentVertexIdx = edge.first;
        
        for (int i = 0; i < numSegments; ++i) {
            size_t nextVertexIdx;
            
            if (i == numSegments - 1) {
                nextVertexIdx = edge.second;
            } else {
                Scalar t = static_cast<Scalar>(i + 1) / numSegments;
                Point interpolatedVertex = startVertex + t * (endVertex - startVertex);  // Eigen vector arithmetic
                newVertices.push_back(interpolatedVertex);
                nextVertexIdx = newVertices.size() - 1;
            }
            
            newEdges.emplace_back(currentVertexIdx, nextVertexIdx);
            newNormals1.push_back(normal1);
            newNormals2.push_back(normal2);
            
            currentVertexIdx = nextVertexIdx;
        }
    }
    
    target.setVertices(newVertices);
    target.setEdges(newEdges);
    target.setNormals1(newNormals1);
    target.setNormals2(newNormals2);
    
    std::cout << "Resampling complete: "
              << sourceVertices.size() << " -> " << newVertices.size() << " vertices, "
              << sourceEdges.size() << " -> " << newEdges.size() << " edges" << std::endl;
    
    return true;
}

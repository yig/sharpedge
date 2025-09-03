// edge_dual_normal.h
#pragma once

#include <vector>
#include <string>
#include <map>
#include <Eigen/Core>

// Type aliases for easier migration
using Point = Eigen::Vector3d;
using Scalar = double;

class EdgeDualNormalGeometry {
public:
    EdgeDualNormalGeometry();
    ~EdgeDualNormalGeometry() = default;

    // Basic accessors
    const std::vector<Point>& getVertices() const { return vertices_; }
    const std::vector<std::pair<size_t, size_t>>& getEdges() const { return edges_; }
    const std::vector<Point>& getNormals1() const { return normals1_; }
    const std::vector<Point>& getNormals2() const { return normals2_; }

    // Basic info
    size_t nVertices() const { return vertices_.size(); }
    size_t nEdges() const { return edges_.size(); }
    bool isValid() const;
    
    // Utility functions
    Scalar getEdgeLength(size_t edgeIdx) const;
    Scalar getMeanEdgeLength() const;
    std::pair<Point, Point> getEdgeNormals(size_t edgeIdx) const;
    
    // Data manipulation
    void setVertices(const std::vector<Point>& vertices) { vertices_ = vertices; }
    void setEdges(const std::vector<std::pair<size_t, size_t>>& edges) { edges_ = edges; }
    void setNormals1(const std::vector<Point>& normals1) { normals1_ = normals1; }
    void setNormals2(const std::vector<Point>& normals2) { normals2_ = normals2; }
    void clear();
    
    // Debug
    void printSummary() const;

private:
    std::vector<Point> vertices_;
    std::vector<std::pair<size_t, size_t>> edges_;
    std::vector<Point> normals1_;
    std::vector<Point> normals2_;
};

// Helper functions
bool readEdgeDualNormal(const std::string& filename, EdgeDualNormalGeometry& geometry);

bool resampleEdgeDualNormalGeometry(
    const EdgeDualNormalGeometry& source,
    EdgeDualNormalGeometry& target,
    Scalar targetEdgeLength);

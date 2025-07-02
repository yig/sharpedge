#pragma once

#include "geometrycentral/numerical/linear_solvers.h"
#include "polyscope/volume_grid.h"

#include "signed_heat_3d.h"
#include "point_position_dual_normal_geometry.h"
#include "edge_dual_normal_geometry.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

class SignedHeatGridSolver {

  public:
    SignedHeatGridSolver();

    Vector<double> computeDistance(VertexPositionGeometry& geometry,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());

    Vector<double> computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                   const SignedHeat3DOptions& options = SignedHeat3DOptions());
    
    Vector<double> computeDistance(pointcloud::PointPositionDualNormalGeometry& pointGeom,
                                  const SignedHeat3DOptions& options);
    
    Vector<double> computeDistance(EdgeDualNormalGeometry& edgeGeom,
                                   const SignedHeat3DOptions& options);


    bool VERBOSE = true;

  private:
    size_t nx = 0;
    size_t ny, nz; // number of vertices on x/y/z side of grid
    Vector3 bboxMin, bboxMax;

    double shortTime, cellSize;

    SparseMatrix<double> laplaceMat;
    FaceData<double> faceAreas;    // of the source geometry
    FaceData<Vector3> faceNormals; // of the source geometry

    SparseMatrix<double> laplacian() const;
    SparseMatrix<double> gradient() const;
    Vector<double> integrateGreedily(const Eigen::VectorXd& Yt);
    double evaluateFunction(const Vector<double>& u, const Vector3& q) const;
    void trilinearCoefficients(const Vector3& q, std::vector<size_t>& nodeIndices, std::vector<double>& coeffs) const;
    double evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry, const Vector<double>& u) const;
    double evaluateAverageAlongSourceGeometry(pointcloud::PointPositionGeometry& pointGeom,
                                              const Vector<double>& u) const;
    Vector3 barycenter(VertexPositionGeometry& geometry, const Face& f) const;
    size_t indicesToNodeIndex(const size_t& i, const size_t& j, const size_t& k) const;
    Vector3 indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const;
    void exportData(const Vector<double>& phi, const SignedHeat3DOptions& options) const;
    
    
    
    // For edge dual normal
    Vector3 centroidFromEdges(const EdgeDualNormalGeometry& edgeGeom);
    double radiusFromEdges(const EdgeDualNormalGeometry& edgeGeom, const Vector3& center);
    double calculateAverageEdgeLength(const EdgeDualNormalGeometry& edgeGeom);
    double evaluateAverageAlongEdgeGeometry(const EdgeDualNormalGeometry& edgeGeom, const Vector<double>& phi);
    double evaluateAtPoint(const Vector3& point, const Vector<double>& phi);

    
    
};

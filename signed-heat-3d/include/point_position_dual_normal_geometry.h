// Created by Xue
#pragma once

#include "geometrycentral/pointcloud/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_normal_geometry.h"

namespace geometrycentral {
namespace pointcloud {

class PointPositionDualNormalGeometry : public PointPositionNormalGeometry {
public:
  // Constructor
  PointPositionDualNormalGeometry(PointCloud& cloud, PointData<Vector3>& positions,
                                  PointData<Vector3>& normals0, PointData<Vector3>& normals1);

  // Destructor
  virtual ~PointPositionDualNormalGeometry() = default;

  // The second set of normals
  PointData<Vector3> secondNormals;

protected:
  // Override any methods from base class as needed
};

// Constructor implementation
inline PointPositionDualNormalGeometry::PointPositionDualNormalGeometry(
    PointCloud& cloud, PointData<Vector3>& positions, PointData<Vector3>& normals0, PointData<Vector3>& normals1)
    : PointPositionNormalGeometry(cloud, positions, normals0), secondNormals(normals1) {}

} // namespace pointcloud
} // namespace geometrycentral

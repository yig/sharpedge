#include "signed_heat_grid_solver.h"

SignedHeatGridSolver::SignedHeatGridSolver() {}

Vector<double> SignedHeatGridSolver::computeDistance(VertexPositionGeometry& geometry,
                                                     const SignedHeat3DOptions& options) {

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        Vector3 c = centroid(geometry);
        double r = radius(geometry, c);
        double s = r * options.scale;
        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    SurfaceMesh& mesh = geometry.mesh;
    double h = meanEdgeLength(geometry);
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    setFaceVectorAreas(geometry, faceAreas, faceNormals);
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 x = indicesToNodePosition(i, j, k);
                for (Face f : mesh.faces()) {
                    Vector3 N = faceNormals[f];
                    Vector3 y = barycenter(geometry, f);
                    double A = faceAreas[f];
                    Vector3 source = N * A * yukawaPotential(x, y, lambda);
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) += source[p];
                }
                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                X /= X.norm();
                for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    for (size_t i = 0; i < divYt.size(); i++) {
        if (std::isinf(divYt[i]) || std::isnan(divYt[i])) divYt[i] = 0.;
    }
    // No level set constraints implemented for grid.
    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;
        for (Face f : mesh.faces()) {
            Vector3 b = barycenter(geometry, f);
            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSize);
            size_t j = std::floor(d[1] / cellSize);
            size_t k = std::floor(d[2] / cellSize);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }
        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(geometry, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (options.exportData) exportData(phi, options);
    return phi;
}

Vector<double> SignedHeatGridSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                     const SignedHeat3DOptions& options) {
    
    std::cout << "PointPositionNormalGeometry SINGLE NORMAL" << std::endl;

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        Vector3 c = centroid(pointGeom);
        double r = radius(pointGeom, c);
        double s = r * options.scale;
        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();
    double h = meanEdgeLength(*(pointGeom.tuftedGeom));
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    size_t P = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 y = indicesToNodePosition(i, j, k);
                for (size_t pIdx = 0; pIdx < P; pIdx++) {
                    Vector3 x = pointGeom.positions[pIdx];
                    Vector3 n = pointGeom.normals[pIdx];
                    double A = pointGeom.tuftedGeom->vertexDualAreas[pIdx];
                    Vector3 source = n * A * yukawaPotential(x, y, lambda);
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) += source[p];
                }
                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                X /= X.norm();
                for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    // No level set constraints implemented for grid.
    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 b = pointGeom.positions[pIdx];
            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSize);
            size_t j = std::floor(d[1] / cellSize);
            size_t k = std::floor(d[2] / cellSize);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }
        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(pointGeom, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    pointGeom.unrequireTuftedTriangulation();
    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    if (options.exportData) exportData(phi, options);
    return phi;
}


// Xue : Try to make it work with dual normal
Vector<double> SignedHeatGridSolver::computeDistance(pointcloud::PointPositionDualNormalGeometry& pointGeom,
                                                     const SignedHeat3DOptions& options) {
    
    
    std::cout << "PointPositionDualNormalGeometry" << std::endl;

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        Vector3 c = centroid(pointGeom);
        double r = radius(pointGeom, c);
        double s = r * options.scale;
//        std::cout << "options.scale " << options.scale << std::endl;
        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    // With direct convolution in R^n, it's not clear what we should pick as our timestep. Use the
    // input mesh as a heuristic.
    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();
    double h = meanEdgeLength(*(pointGeom.tuftedGeom));
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);
    size_t P = pointGeom.cloud.nPoints();
    
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 q = indicesToNodePosition(i, j, k);
                
                for (size_t pIdx = 0; pIdx < P; pIdx++) {
                    Vector3 p = pointGeom.positions[pIdx];
                    Vector3 n = pointGeom.normals[pIdx];
                    Vector3 n_prime = pointGeom.secondNormals[pIdx];
                    double A = pointGeom.tuftedGeom->vertexDualAreas[pIdx];
                    
          
                    // Implement Nicole's logic for choosing which normal to use
                    Vector3 direction = q - p;
                    
                    // Calculate dot products to determine which side of each plane the query point is on
                    double dot1 = dot(direction, n);
                    double dot2 = dot(direction, n_prime);
                    

#define NICOLE 0
                    Vector3 normalToUse;
                    
                    // Logic as described in Nicole's email:
                    if (dot1 > 0 && dot2 < 0) {
                        normalToUse = n;
                    } else if (dot1 < 0 && dot2 > 0) {
                        normalToUse = n_prime;
                    } else if (dot1 > 0 && dot2 > 0) {
#if NICOLE
                        // If outside both planes, use normalized direction vector
                        normalToUse = direction.normalize();
#else
                        // Outside the n1 and n2
                        // Use 3 cases now
                        // close to n1
                        // close to n2
                        
                        Vector3 bisector = ( n_prime + n ) / 2;
                        bisector.normalize();
                        
                        double dot_bisector = dot(direction, bisector);
                        
                        assert(dot_bisector > 0);
                        
                        if (dot_bisector > dot1 and dot_bisector > dot2)
                        {
                            normalToUse = direction.normalize();
                        }
                        else if (dot1 < dot2)
                        {
                            normalToUse = n_prime;
                        }
                        else{
                            normalToUse = n;
                        }
#endif
                    } else {

#if NICOLE
                        // If inside both planes, use first normal
                        normalToUse = n;
                        // Select the normal with the smaller negative dot product
#else
                        if (dot1 > dot2) {
                            normalToUse = n;
                        } else {
                            normalToUse = n_prime;
                        }
#endif
                    }
                    
                    
        
                    Vector3 source = normalToUse * A * yukawaPotential(p, q, lambda);
                    for (int d = 0; d < 3; d++) Y(3 * idx + d) += source[d];
                }
                
                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                double norm = X.norm();
                if (norm > 0) {
                    X /= norm;
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
                }
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    // No level set constraints implemented for grid.
    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 b = pointGeom.positions[pIdx];
            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSize);
            size_t j = std::floor(d[1] / cellSize);
            size_t k = std::floor(d[2] / cellSize);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }
        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }
    double shift = evaluateAverageAlongSourceGeometry(pointGeom, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;
    pointGeom.unrequireTuftedTriangulation();
    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    if (options.exportData) exportData(phi, options);
    return phi;
}



Vector<double> SignedHeatGridSolver::integrateGreedily(const Eigen::VectorXd& Yt) {

    Vector<double> phi = Vector<double>::Zero(nx * ny * nz);
    Vector<bool> visited = Vector<bool>::Zero(nx * ny * nz);
    std::queue<std::array<size_t, 3>> queue;
    queue.push({0, 0, 0});
    visited[0] = true;
    std::array<size_t, 3> dims = {nx, ny, nz};
    std::array<size_t, 3> curr, next;
    while (!queue.empty()) {
        curr = queue.front();
        Vector3 p = indicesToNodePosition(curr[0], curr[1], curr[2]);
        size_t currIdx = indicesToNodeIndex(curr[0], curr[1], curr[2]);
        Eigen::Vector3d Yp = {Yt(3 * currIdx), Yt(3 * currIdx + 1), Yt(3 * currIdx + 2)};
        queue.pop();
        for (int i = 0; i < 3; i++) {
            if (curr[i] > 0) {
                next = curr;
                next[i] -= 1;
                size_t nextIdx = indicesToNodeIndex(next[0], next[1], next[2]);
                if (!visited[nextIdx]) {
                    Vector3 q = indicesToNodePosition(next[0], next[1], next[2]);
                    Vector3 edge = q - p;
                    Eigen::Vector3d Yq = {Yt(3 * nextIdx), Yt(3 * nextIdx + 1), Yt(3 * nextIdx + 2)};
                    Eigen::Vector3d Y_avg = (Yq + Yp);
                    Y_avg /= Y_avg.norm();
                    Vector3 Y = {Y_avg[0], Y_avg[1], Y_avg[2]};
                    phi[nextIdx] = phi[currIdx] + dot(Y, edge);
                    visited[nextIdx] = true;
                    queue.push(next);
                }
            }
            if (curr[i] < dims[i] - 1) {
                next = curr;
                next[i] += 1;
                size_t nextIdx = indicesToNodeIndex(next[0], next[1], next[2]);
                if (!visited[nextIdx]) {
                    Vector3 q = indicesToNodePosition(next[0], next[1], next[2]);
                    Vector3 edge = q - p;
                    Eigen::Vector3d Yq = {Yt(3 * nextIdx), Yt(3 * nextIdx + 1), Yt(3 * nextIdx + 2)};
                    Eigen::Vector3d Y_avg = (Yq + Yp);
                    Y_avg /= Y_avg.norm();
                    Vector3 Y = {Y_avg[0], Y_avg[1], Y_avg[2]};
                    phi[nextIdx] = phi[currIdx] + dot(Y, edge);
                    visited[nextIdx] = true;
                    queue.push(next);
                }
            }
        }
    }
    return phi;
}



// Modified computeDistance function for EdgeDualNormalGeometry
Vector<double> SignedHeatGridSolver::computeDistance(EdgeDualNormalGeometry& edgeGeom,
                                                     const SignedHeat3DOptions& options) {
    
    std::cout << "EdgeDualNormalGeometry with dual normals per edge" << std::endl;

    if (options.rebuild) {
        if (VERBOSE) std::cerr << "Building grid..." << std::endl;
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();

        // Calculate centroid and radius from edge vertices
        Vector3 c = centroidFromEdges(edgeGeom);
        double r = radiusFromEdges(edgeGeom, c);
        double s = r * options.scale;

        // clang-format off
        bboxMin = {-s, -s, -s}; bboxMax = {s, s, s};
        bboxMin += c; bboxMax += c;
        glm::vec3 boundMin, boundMax;
        for (int i = 0; i < 3; i++) {
            boundMin[i] = bboxMin[i];
            boundMax[i] = bboxMax[i];
        }
        nx = 2 * std::pow(2, options.hCoef + 3); ny = nx; nz = nx;
        // clang-format on
        cellSize = 2. * s / (nx - 1);
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = laplacian();
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
        polyscope::VolumeGrid* psGrid = polyscope::registerVolumeGrid("domain", {nx, ny, nz}, boundMin, boundMax);
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;

    // Calculate timestep based on average edge length
    double h = calculateAverageEdgeLength(edgeGeom);
    shortTime = options.tCoef * h * h;
    double lambda = std::sqrt(1. / shortTime);
    size_t totalNodes = nx * ny * nz;
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(3 * totalNodes);

    const auto& edges = edgeGeom.getEdges();
    const auto& vertices = edgeGeom.getVertices();
    const auto& normals1 = edgeGeom.getNormals1();
    const auto& normals2 = edgeGeom.getNormals2();
    size_t numEdges = edges.size();

    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t idx = indicesToNodeIndex(i, j, k);
                Vector3 x = indicesToNodePosition(i, j, k);  // Grid point

                // Process each edge
                for (size_t edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
                    // Get edge endpoints
                    size_t v0Idx = edges[edgeIdx].first;
                    size_t v1Idx = edges[edgeIdx].second;
                    Vector3 v0 = vertices[v0Idx];
                    Vector3 v1 = vertices[v1Idx];

                    // Calculate edge midpoint (sample point on edge)
                    Vector3 y = (v0 + v1) * 0.5;  // Edge midpoint

                    // Get dual normals for this edge
                    Vector3 n = normals1[edgeIdx];
                    Vector3 n_prime = normals2[edgeIdx];

                    // Calculate edge length as area weight
                    double edgeLength = (v1 - v0).norm();
                    double A = edgeLength; // Use edge length as weight

                    // Direction from edge midpoint to grid point
                    Vector3 direction = x - y;

                    // Calculate dot products to determine which side of each plane the query point is on
                    double dot1 = dot(direction, n);
                    double dot2 = dot(direction, n_prime);
                    
                    Vector3 normalToUse;

                    // Logic for choosing which normal to use (same as your point-based logic)
                    if (dot1 > 0 && dot2 < 0) {
                        normalToUse = n;
                    } else if (dot1 < 0 && dot2 > 0) {
                        normalToUse = n_prime;
                    } else if (dot1 > 0 && dot2 > 0) {
#if NICOLE
                        // If outside both planes, use normalized direction vector
                        normalToUse = direction.normalize();
#else
                        // Outside both n1 and n2
                        Vector3 bisector = (n_prime + n) / 2;
                        bisector = bisector.normalize();

                        double dot_bisector = dot(direction, bisector);

                        assert(dot_bisector > 0);

                        if (dot_bisector > dot1 && dot_bisector > dot2) {
                            normalToUse = direction.normalize();
                        } else if (dot1 < dot2) {
                            normalToUse = n_prime;
                        } else {
                            normalToUse = n;
                        }
#endif
                    } else {
#if NICOLE
                        // If inside both planes, use first normal
                        normalToUse = n;
#else
                        if (dot1 > dot2) {
                            normalToUse = n;
                        } else {
                            normalToUse = n_prime;
                        }
#endif
                    }

                    Vector3 source = normalToUse * A * yukawaPotential(x, y, lambda);  // x, y order
                    for (int p = 0; p < 3; p++) Y(3 * idx + p) += source[p];
                }

                Vector3 X = {Y(3 * idx + 0), Y(3 * idx + 1), Y(3 * idx + 2)};
                X /= X.norm();  // Simplified like in VertexPositionGeometry
                for (int p = 0; p < 3; p++) Y(3 * idx + p) = X[p];
            }
        }
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    // Integrate gradient to get distance.
    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    SparseMatrix<double> D = gradient(); // 3N x N
    Vector<double> divYt = D.transpose() * Y;
    
    // Add NaN/inf checking like in VertexPositionGeometry version
    for (size_t i = 0; i < divYt.size(); i++) {
        if (std::isinf(divYt[i]) || std::isnan(divYt[i])) divYt[i] = 0.;
    }

    Vector<double> phi;
    if (options.fastIntegration) {
        phi = integrateGreedily(Y);
    } else {
        SparseMatrix<double> A;
        size_t m = 0;
        std::vector<size_t> nodeIndices;
        std::vector<double> coeffs;
        std::vector<bool> hasCellBeenUsed(totalNodes, false);
        std::vector<Eigen::Triplet<double>> tripletList;

        // Sample constraint points from edge midpoints
        for (size_t edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
            size_t v0Idx = edges[edgeIdx].first;
            size_t v1Idx = edges[edgeIdx].second;
            Vector3 v0 = vertices[v0Idx];
            Vector3 v1 = vertices[v1Idx];
            Vector3 b = (v0 + v1) * 0.5; // Edge midpoint

            Vector3 d = b - bboxMin;
            size_t i = std::floor(d[0] / cellSize);
            size_t j = std::floor(d[1] / cellSize);
            size_t k = std::floor(d[2] / cellSize);
            size_t nodeIdx = indicesToNodeIndex(i, j, k);
            if (hasCellBeenUsed[nodeIdx]) continue;
            trilinearCoefficients(b, nodeIndices, coeffs);
            for (size_t i = 0; i < nodeIndices.size(); i++) tripletList.emplace_back(m, nodeIndices[i], coeffs[i]);
            hasCellBeenUsed[nodeIdx] = true;
            m++;
        }

        A.resize(m, totalNodes);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(totalNodes + m);
        RHS.head(totalNodes) = divYt;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = -soln.head(totalNodes);
    }

    double shift = evaluateAverageAlongEdgeGeometry(edgeGeom, phi);
    phi -= shift * Vector<double>::Ones(totalNodes);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (options.exportData) exportData(phi, options);
    return phi;
}


// Helper functions you'll need to implement:

Vector3 SignedHeatGridSolver::centroidFromEdges(const EdgeDualNormalGeometry& edgeGeom) {
    const auto& vertices = edgeGeom.getVertices();
    Vector3 centroid = {0, 0, 0};
    for (const auto& v : vertices) {
        centroid += v;
    }
    return centroid / vertices.size();
}

double SignedHeatGridSolver::radiusFromEdges(const EdgeDualNormalGeometry& edgeGeom, const Vector3& center) {
    const auto& vertices = edgeGeom.getVertices();
    double maxDist = 0;
    for (const auto& v : vertices) {
        double dist = (v - center).norm();
        if (dist > maxDist) maxDist = dist;
    }
    return maxDist;
}

double SignedHeatGridSolver::calculateAverageEdgeLength(const EdgeDualNormalGeometry& edgeGeom) {
    const auto& edges = edgeGeom.getEdges();
    const auto& vertices = edgeGeom.getVertices();
    double totalLength = 0;

    for (const auto& edge : edges) {
        Vector3 v0 = vertices[edge.first];
        Vector3 v1 = vertices[edge.second];
        totalLength += (v1 - v0).norm();
    }

    return totalLength / edges.size();
}

double SignedHeatGridSolver::evaluateAverageAlongEdgeGeometry(const EdgeDualNormalGeometry& edgeGeom,
                                                              const Vector<double>& phi) {
    const auto& edges = edgeGeom.getEdges();
    const auto& vertices = edgeGeom.getVertices();
    double sum = 0;
    size_t count = 0;

    for (const auto& edge : edges) {
        Vector3 v0 = vertices[edge.first];
        Vector3 v1 = vertices[edge.second];
        Vector3 midpoint = (v0 + v1) * 0.5;

        // Evaluate phi at edge midpoint
        double value = evaluateAtPoint(midpoint, phi);
        sum += value;
        count++;
    }

    return sum / count;
}


double SignedHeatGridSolver::evaluateAtPoint(const Vector3& point, const Vector<double>& phi) {
    // Convert world coordinates to grid indices
    Vector3 d = point - bboxMin;
    double fi = d[0] / cellSize;
    double fj = d[1] / cellSize;
    double fk = d[2] / cellSize;
    
    // Get integer parts and fractional parts
    size_t i0 = std::floor(fi); size_t i1 = i0 + 1;
    size_t j0 = std::floor(fj); size_t j1 = j0 + 1;
    size_t k0 = std::floor(fk); size_t k1 = k0 + 1;
    
    double alpha = fi - i0;
    double beta = fj - j0;
    double gamma = fk - k0;
    
    // Bounds checking
    if (i1 >= nx || j1 >= ny || k1 >= nz) return 0.0;
    
    // Trilinear interpolation
    double c000 = phi[indicesToNodeIndex(i0, j0, k0)];
    double c001 = phi[indicesToNodeIndex(i0, j0, k1)];
    double c010 = phi[indicesToNodeIndex(i0, j1, k0)];
    double c011 = phi[indicesToNodeIndex(i0, j1, k1)];
    double c100 = phi[indicesToNodeIndex(i1, j0, k0)];
    double c101 = phi[indicesToNodeIndex(i1, j0, k1)];
    double c110 = phi[indicesToNodeIndex(i1, j1, k0)];
    double c111 = phi[indicesToNodeIndex(i1, j1, k1)];
    
    double c00 = c000 * (1 - alpha) + c100 * alpha;
    double c01 = c001 * (1 - alpha) + c101 * alpha;
    double c10 = c010 * (1 - alpha) + c110 * alpha;
    double c11 = c011 * (1 - alpha) + c111 * alpha;
    
    double c0 = c00 * (1 - beta) + c10 * beta;
    double c1 = c01 * (1 - beta) + c11 * beta;
    
    return c0 * (1 - gamma) + c1 * gamma;
}




/* Builds negative-definite Laplace */
SparseMatrix<double> SignedHeatGridSolver::laplacian() const {

    // Use 5-point stencil (well, I guess 7-point in 3D)
    size_t N = nx * ny * nz;
    SparseMatrix<double> L(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
                size_t currX = currIdx;
                size_t currY = currIdx;
                size_t currZ = currIdx;
                size_t nextX = indicesToNodeIndex(i + 1, j, k);
                size_t nextY = indicesToNodeIndex(i, j + 1, k);
                size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                size_t prevX = indicesToNodeIndex(i - 1, j, k);
                size_t prevY = indicesToNodeIndex(i, j - 1, k);
                size_t prevZ = indicesToNodeIndex(i, j, k - 1);

                // Use mirroring for differences along boundary.
                if (i == nx - 1) {
                    nextX = currIdx;
                    currX = indicesToNodeIndex(i - 1, j, k);
                } else if (i == 0) {
                    prevX = currX;
                    currX = nextX;
                }
                if (j == ny - 1) {
                    nextY = currIdx;
                    currY = indicesToNodeIndex(i, j - 1, k);
                } else if (j == 0) {
                    prevY = currIdx;
                    currY = nextY;
                }
                if (k == nz - 1) {
                    nextZ = currIdx;
                    currZ = indicesToNodeIndex(i, j, k - 1);
                } else if (k == 0) {
                    prevZ = currIdx;
                    currZ = nextZ;
                }

                triplets.emplace_back(currIdx, nextX, 1);
                triplets.emplace_back(currIdx, nextY, 1);
                triplets.emplace_back(currIdx, nextZ, 1);
                triplets.emplace_back(currIdx, prevX, 1);
                triplets.emplace_back(currIdx, prevY, 1);
                triplets.emplace_back(currIdx, prevZ, 1);
                triplets.emplace_back(currIdx, currIdx, -6);
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L / (cellSize * cellSize);
}

SparseMatrix<double> SignedHeatGridSolver::gradient() const {

    size_t N = nx * ny * nz;
    SparseMatrix<double> D(3 * N, N);
    std::vector<Eigen::Triplet<double>> tripletList;
    for (size_t i = 0; i < nx; i++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t k = 0; k < nz; k++) {
                size_t currIdx = indicesToNodeIndex(i, j, k);
                // if (i < nx - 1) {
                //     size_t nextX = indicesToNodeIndex(i + 1, j, k);
                //     tripletList.emplace_back(3 * currIdx, nextX, 1);
                // }
                // if (i > 0) {
                //     size_t prevX = indicesToNodeIndex(i - 1, j, k);
                //     tripletList.emplace_back(3 * currIdx, prevX, -1);
                // }
                // if (j < ny - 1) {
                //     size_t nextY = indicesToNodeIndex(i, j + 1, k);
                //     tripletList.emplace_back(3 * currIdx + 1, nextY, 1);
                // }
                // if (j > 0) {
                //     size_t prevY = indicesToNodeIndex(i, j - 1, k);
                //     tripletList.emplace_back(3 * currIdx + 1, prevY, -1);
                // }
                // if (k < nz - 1) {
                //     size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                //     tripletList.emplace_back(3 * currIdx + 2, nextZ, 1);
                // }
                // if (k > 0) {
                //     size_t prevZ = indicesToNodeIndex(i, j, k - 1);
                //     tripletList.emplace_back(3 * currIdx + 2, prevZ, -1);
                // }

                // forward differences
                size_t currX = currIdx;
                size_t currY = currIdx;
                size_t currZ = currIdx;
                size_t nextX = indicesToNodeIndex(i + 1, j, k);
                size_t nextY = indicesToNodeIndex(i, j + 1, k);
                size_t nextZ = indicesToNodeIndex(i, j, k + 1);
                // Use mirroring for differences along boundary.
                if (i == nx - 1) {
                    nextX = currIdx;
                    currX = indicesToNodeIndex(i - 1, j, k);
                }
                if (j == ny - 1) {
                    nextY = currIdx;
                    currY = indicesToNodeIndex(i, j - 1, k);
                }
                if (k == nz - 1) {
                    nextZ = currIdx;
                    currZ = indicesToNodeIndex(i, j, k - 1);
                }
                tripletList.emplace_back(3 * currIdx, nextX, 1);
                tripletList.emplace_back(3 * currIdx, currX, -1);
                tripletList.emplace_back(3 * currIdx + 1, nextY, 1);
                tripletList.emplace_back(3 * currIdx + 1, currY, -1);
                tripletList.emplace_back(3 * currIdx + 2, nextZ, 1);
                tripletList.emplace_back(3 * currIdx + 2, currZ, -1);
            }
        }
    }
    D.setFromTriplets(tripletList.begin(), tripletList.end());

    return D / cellSize;
}

/* Evaluate a function at position q, interpolating trilinearly inside grid cells. */
double SignedHeatGridSolver::evaluateFunction(const Vector<double>& u, const Vector3& q) const {

    Vector3 d = q - bboxMin;
    int i = static_cast<int>(std::floor(d[0] / cellSize));
    int j = static_cast<int>(std::floor(d[1] / cellSize));
    int k = static_cast<int>(std::floor(d[2] / cellSize));
    Vector3 p000 = indicesToNodePosition(i, j, k);
    double v000 = u[indicesToNodeIndex(i, j, k)];
    double v100 = u[indicesToNodeIndex(i + 1, j, k)];
    double v010 = u[indicesToNodeIndex(i, j + 1, k)];
    double v001 = u[indicesToNodeIndex(i, j, k + 1)];
    double v110 = u[indicesToNodeIndex(i + 1, j + 1, k)];
    double v101 = u[indicesToNodeIndex(i + 1, j, k + 1)];
    double v011 = u[indicesToNodeIndex(i, j + 1, k + 1)];
    double v111 = u[indicesToNodeIndex(i + 1, j + 1, k + 1)];
    double tx = (q[0] - p000[0]) / cellSize;
    double ty = (q[1] - p000[1]) / cellSize;
    double tz = (q[2] - p000[2]) / cellSize;
    double v00 = v000 * (1. - tx) + v100 * tx;
    double v01 = v001 * (1. - tx) + v101 * tx;
    double v10 = v010 * (1. - tx) + v110 * tx;
    double v11 = v011 * (1. - tx) + v111 * tx;
    double v0 = v00 * (1. - ty) + v10 * ty;
    double v1 = v01 * (1. - ty) + v11 * ty;
    double v = v0 * (1. - tz) + v1 * tz;
    return v;
}

void SignedHeatGridSolver::trilinearCoefficients(const Vector3& q, std::vector<size_t>& nodeIndices,
                                                 std::vector<double>& coeffs) const {

    double h = cellSize;
    Vector3 d = q - bboxMin;
    size_t i = std::floor(d[0] / h);
    size_t j = std::floor(d[1] / h);
    size_t k = std::floor(d[2] / h);
    Vector3 p000 = indicesToNodePosition(i, j, k);
    size_t i000 = indicesToNodeIndex(i, j, k);
    size_t i100 = indicesToNodeIndex(i + 1, j, k);
    size_t i010 = indicesToNodeIndex(i, j + 1, k);
    size_t i001 = indicesToNodeIndex(i, j, k + 1);
    size_t i110 = indicesToNodeIndex(i + 1, j + 1, k);
    size_t i101 = indicesToNodeIndex(i + 1, j, k + 1);
    size_t i011 = indicesToNodeIndex(i, j + 1, k + 1);
    size_t i111 = indicesToNodeIndex(i + 1, j + 1, k + 1);
    nodeIndices = {i000, i100, i010, i001, i110, i101, i011, i111};
    double tx = (q[0] - p000[0]) / h;
    double ty = (q[1] - p000[1]) / h;
    double tz = (q[2] - p000[2]) / h;
    coeffs = {
        (1. - tx) * (1. - ty) * (1. - tz), // 000
        tx * (1. - ty) * (1. - tz),        // 100
        (1. - tx) * ty * (1. - tz),        // 010
        (1. - tx) * (1. - ty) * tz,        // 001
        tx * ty * (1. - tz),               // 110
        tx * (1. - ty) * tz,               // 101
        (1. - tx) * ty * tz,               // 011
        tx * ty * tz                       // 111
    };
}

double SignedHeatGridSolver::evaluateAverageAlongSourceGeometry(VertexPositionGeometry& geometry,
                                                                const Vector<double>& u) const {

    // Again integrate (approximately) using 1-pt quadrature.
    SurfaceMesh& mesh = geometry.mesh;
    double shift = 0.;
    double normalization = 0.;
    for (Face f : mesh.faces()) {
        double A = faceAreas[f];
        Vector3 x = barycenter(geometry, f);
        shift += A * evaluateFunction(u, x);
        normalization += A;
    }
    shift /= normalization;
    return shift;
}

double SignedHeatGridSolver::evaluateAverageAlongSourceGeometry(pointcloud::PointPositionGeometry& pointGeom,
                                                                const Vector<double>& u) const {

    double shift = 0.;
    double normalization = 0.;
    size_t P = pointGeom.cloud.nPoints();
    for (size_t i = 0; i < P; i++) {
        double A = pointGeom.tuftedGeom->vertexDualAreas[i];
        shift += A * evaluateFunction(u, pointGeom.positions[i]);
        normalization += A;
    }
    shift /= normalization;
    return shift;
}

Vector3 SignedHeatGridSolver::barycenter(VertexPositionGeometry& geometry, const Face& f) const {
    Vector3 c = {0, 0, 0};
    for (Vertex v : f.adjacentVertices()) c += geometry.vertexPositions[v];
    c /= f.degree();
    return c;
}

size_t SignedHeatGridSolver::indicesToNodeIndex(const size_t& i, const size_t& j, const size_t& k) const {
    // return i * (ny * nz) + j * nz + k;
    return i + j * ny + k * (nx * ny);
}

Vector3 SignedHeatGridSolver::indicesToNodePosition(const size_t& i, const size_t& j, const size_t& k) const {
    Vector3 pos = {i * cellSize, j * cellSize, k * cellSize};
    pos += bboxMin;
    return pos;
}

/*
 * Write CSV file, where each row is a node of the grid.
 * The grid positions are defined in the computeDistance() functions.
 * Columns: xCoord, yCoord, zCoord, SDF
 * The first three columns record the (x,y,z) position of the node of the grid.
 * "SDF" records the SDF value at the node.
 */
void SignedHeatGridSolver::exportData(const Vector<double>& phi, const SignedHeat3DOptions& options) const {

    std::string filename = "../export/" + options.meshname + ".csv";
    std::fstream f;
    f.open(filename, std::ios::out | std::ios::trunc);
    if (f.is_open()) {
        f << "xCoord,yCoord,zCoord,SDF" << "\n";
        for (size_t i = 0; i < nx; i++) {
            for (size_t j = 0; j < ny; j++) {
                for (size_t k = 0; k < nz; k++) {
                    Vector3 x = indicesToNodePosition(i, j, k);
                    size_t idx = indicesToNodeIndex(i, j, k);
                    f << x[0] << "," << x[1] << "," << x[2] << "," << phi[idx] << "\n";
                }
            }
        }
        f.close();
        if (VERBOSE) std::cerr << "File " << filename << " written succesfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export '" << filename << "'!" << std::endl;
    }
}

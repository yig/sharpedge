#include "signed_heat_tet_solver.h"

#define USETetgen 0
SignedHeatTetSolver::SignedHeatTetSolver() {}

// =============== ALGORITHM

Vector<double> SignedHeatTetSolver::computeDistance(VertexPositionGeometry& geometry,
                                                    const SignedHeat3DOptions& options) {

    bool isConforming = false;
    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building tet mesh..." << std::endl;
        double meanFaceArea = 0.;
        SurfaceMesh& mesh = geometry.mesh;
        setFaceVectorAreas(geometry, surfaceFaceAreas, surfaceFaceNormals);
        for (Face f : mesh.faces()) meanFaceArea += surfaceFaceAreas[f];
        meanFaceArea /= mesh.nFaces();
        double areaScale = std::pow(2, -options.hCoef);
        TETFLAGS = TET_PREFIX + std::to_string(areaScale * meanFaceArea);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(areaScale * meanFaceArea) + "Y";
        if (mesh.isTriangular()) isConforming = tetmeshDomain(geometry);
        if (!isConforming) {
            size_t nPts = mesh.nVertices();
            cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
            pointcloud::PointData<Vector3> pointPositions = pointcloud::PointData<Vector3>(*cloud);
            for (size_t i = 0; i < nPts; i++) pointPositions[i] = geometry.vertexPositions[i];
            pointPolyGeom = std::unique_ptr<pointcloud::PointPositionGeometry>(
                new pointcloud::PointPositionGeometry(*cloud, pointPositions));
            tetmeshPointCloud(*pointPolyGeom);
        }
        // With direct convolution in R^n, it's not clear what we should pick as our timestep. Just use the
        // tetmesh/trimesh as a proxy.
        if (VERBOSE) std::cerr << "Computing tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = dualLaplacian();
        if (VERBOSE) std::cerr << "Tet mesh (re)built" << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    Eigen::MatrixXd Yt = Eigen::MatrixXd::Zero(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    SurfaceMesh& mesh = geometry.mesh;
    size_t F = mesh.nFaces();
    // Integrate contributions (single-point quadrature)
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions (single-point quadrature)
        Vector3 X = {0, 0, 0};
        for (Face f : mesh.faces()) {
            Vector3 p = {0, 0, 0};
            for (Vertex v : f.adjacentVertices()) p += geometry.vertexPositions[v];
            p /= f.degree();
            Vector3 n = surfaceFaceNormals[f];
            X += yukawaPotential(p, q, lambda) * n * surfaceFaceAreas[f];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    Vector<double> phi;
    if (isConforming) {
        phi = options.fastIntegration ? integrateVectorFieldGreedily(geometry, Yt, options)
                                      : integrateVectorField(geometry, Yt, options);
    } else {
        pointPolyGeom->requireTuftedTriangulation();
        pointPolyGeom->tuftedGeom->requireVertexDualAreas();
        phi = options.fastIntegration ? integrateVectorFieldGreedily(*pointPolyGeom, Yt, options)
                                      : integrateVectorField(*pointPolyGeom, Yt, options);
        pointPolyGeom->unrequireTuftedTriangulation();
        pointPolyGeom->tuftedGeom->unrequireVertexDualAreas();
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    return phi;
}

Vector<double> SignedHeatTetSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                    const SignedHeat3DOptions& options) {

    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();

    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building tet mesh..." << std::endl;
        double meanArea = 0.;
        for (size_t i = 0; i < pointGeom.cloud.nPoints(); i++) meanArea += pointGeom.tuftedGeom->vertexDualAreas[i];
        meanArea /= pointGeom.cloud.nPoints();
        double areaScale = std::pow(2, -options.hCoef);
        TETFLAGS = TET_PREFIX + std::to_string(areaScale * meanArea);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(areaScale * meanArea) + "Y";
        tetmeshPointCloud(pointGeom);
        // With direct convolution in R^n, it's not clear what we should pick as our timestep. Just use the
        // tetmesh/trimesh as a proxy.
        if (VERBOSE) std::cerr << "Computing tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = dualLaplacian();
        if (VERBOSE) std::cerr << "Tet mesh (re)built" << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;

    // Evaluate vectors at tet barycenters.
    size_t P = pointGeom.cloud.nPoints();
    Eigen::MatrixXd Yt(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions.
        Vector3 X = {0, 0, 0};
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 p = pointGeom.positions[pIdx];
            Vector3 n = pointGeom.normals[pIdx];
            X += yukawaPotential(p, q, lambda) * n * pointGeom.tuftedGeom->vertexDualAreas[pIdx];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    Vector<double> phi = options.fastIntegration ? integrateVectorFieldGreedily(pointGeom, Yt, options)
                                                 : integrateVectorField(pointGeom, Yt, options);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    pointGeom.unrequireTuftedTriangulation();

    return phi;
}

// Modified computeDistance function for EdgeDualNormalGeometry
// 这是用来处理 dual normal per edge 的函数
//
Vector<double> SignedHeatTetSolver::computeDistance(EdgeDualNormalGeometry& edgeGeom,
                                                    const SignedHeat3DOptions& options) {
    
    bool VERBOSE = true;
    
    std::cout << "SignedHeatTetSolver with dual normals per edge" << std::endl;

    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building tet mesh..." << std::endl;
        
        /*
          Xue: Below are the steps using TetGen to tetrahedralize the input geometry:
               1. Calculate mesh quality parameters and set TetGen FLAGS
               2. Convert input edge geometry to point cloud format (required by TetGen)
               3. Call tetmeshPointCloud() which handles the actual TetGen tetrahedralization
               4. Post-process the generated tetrahedral mesh for numerical computation
        */



        // Calculate mean edge length for area scaling
        double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
        double meanArea = meanEdgeLength; // Use edge length as proxy for area
        double areaScale = std::pow(2, -options.hCoef);
        TETFLAGS = TET_PREFIX + std::to_string(areaScale * meanArea);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(areaScale * meanArea) + "Y";

        // Create point cloud from edge vertices for tetmesh generation
        const auto& vertices_data = edgeGeom.getVertices();
        size_t nPts = vertices_data.size();
        cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
        pointcloud::PointData<Vector3> pointPositions = pointcloud::PointData<Vector3>(*cloud);
        for (size_t i = 0; i < nPts; i++) {
            pointPositions[i] = vertices_data[i];
        }
        pointPolyGeom = std::unique_ptr<pointcloud::PointPositionGeometry>(
            new pointcloud::PointPositionGeometry(*cloud, pointPositions));
#if USETetgen
        tetmeshPointCloud(*pointPolyGeom);

#else
        /*
         Xue : switch to CDT
         */
        
        tetmeshEdgeGeometryCDT(edgeGeom, options);
        
#endif
        
        
        if (VERBOSE) std::cerr << "Computing tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = dualLaplacian();
        if (VERBOSE) std::cerr << "Tet mesh (re)built" << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    
    // Evaluate vectors at tet barycenters
    Eigen::MatrixXd Yt = Eigen::MatrixXd::Zero(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    
    const auto& edges = edgeGeom.getEdges();
    const auto& vertices_data = edgeGeom.getVertices();
    const auto& normals1 = edgeGeom.getNormals1();
    const auto& normals2 = edgeGeom.getNormals2();
    size_t numEdges = edges.size();
    
    for (size_t i = 0; i < nTets; i++) {
        // Compute tet barycenter (query point)
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
        }
        q /= 4.;
        
        // Integrate contributions from all edges
        Vector3 X = {0, 0, 0};
        for (size_t edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
            // Get edge endpoints
            size_t v0Idx = edges[edgeIdx].first;
            size_t v1Idx = edges[edgeIdx].second;
            Vector3 v0 = vertices_data[v0Idx];
            Vector3 v1 = vertices_data[v1Idx];
            
            // Calculate edge midpoint (sample point on edge)
            Vector3 p = (v0 + v1) * 0.5;
            
            // Get dual normals for this edge
            Vector3 n = normals1[edgeIdx];
            Vector3 n_prime = normals2[edgeIdx];
            
            // Calculate edge length as area weight
            double edgeLength = (v1 - v0).norm();
            double A = edgeLength;
            
            // Direction from edge midpoint to query point
            Vector3 direction = q - p;
            
            // Calculate dot products to determine which side of each plane the query point is on
            double dot1 = dot(direction, n);
            double dot2 = dot(direction, n_prime);
            
            Vector3 normalToUse;
            
            // Logic for choosing which normal to use
            if (dot1 > 0 && dot2 < 0) {
                normalToUse = n;
            } else if (dot1 < 0 && dot2 > 0) {
                normalToUse = n_prime;
            } else if (dot1 > 0 && dot2 > 0) {
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
            } else {
                if (dot1 > dot2) {
                    normalToUse = n;
                } else {
                    normalToUse = n_prime;
                }
            }
            
            X += yukawaPotential(p, q, lambda) * normalToUse * A;
        }
        
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    
    // Use point cloud geometry for integration since we're working with non-conforming mesh
    pointPolyGeom->requireTuftedTriangulation();
    pointPolyGeom->tuftedGeom->requireVertexDualAreas();
    
    Vector<double> phi = options.fastIntegration ? integrateVectorFieldGreedily(*pointPolyGeom, Yt, options)
                                                 : integrateVectorField(*pointPolyGeom, Yt, options);
    
    pointPolyGeom->tuftedGeom->unrequireVertexDualAreas();
    pointPolyGeom->unrequireTuftedTriangulation();
    
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    return phi;
}


Vector<double> SignedHeatTetSolver::integrateVectorField(VertexPositionGeometry& geometry, const Eigen::MatrixXd& Yt,
                                                         const SignedHeat3DOptions& options) {

    if (options.useCrouzeixRaviart) return integrateVectorFieldToFaces(geometry, Yt, options);

    SurfaceMesh& mesh = geometry.mesh;
    Vector<double> div = vertexDivergence(Yt);
    Vector<double> phi;
    if (options.levelSetConstraint == LevelSetConstraint::ZeroSet) {
        // Since the tet mesh conforms to the surface, preserving zero can be done via Dirichlet boundary conditions.
        Vector<bool> setAMembership = Vector<bool>::Ones(nVertices);
        for (size_t i = 0; i < mesh.nVertices(); i++) setAMembership[i] = false;
        int nB = nVertices - setAMembership.cast<int>().sum();
        Vector<double> bcVals = Vector<double>::Zero(nB);
        BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceMat, setAMembership, true);
        Vector<double> rhsValsA, rhsValsB;
        decomposeVector(decomp, div, rhsValsA, rhsValsB);
        Vector<double> combinedRHS = rhsValsA;
        Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
        phi = reassembleVector(decomp, Aresult, bcVals);
    } else if (options.levelSetConstraint == LevelSetConstraint::Multiple) {
        // Determine the connected components of the mesh. Do simple depth-first search.
        std::vector<Eigen::Triplet<double>> triplets;
        SparseMatrix<double> A;
        size_t m = 0;
        size_t V = mesh.nVertices();
        VertexData<bool> marked(mesh, false);
        geometry.requireVertexIndices();
        for (Vertex v : mesh.vertices()) {
            if (marked[v]) continue;
            marked[v] = true;
            std::vector<Vertex> queue = {v};
            size_t v0 = geometry.vertexIndices[v];
            Vertex curr;
            while (!queue.empty()) {
                curr = queue.back();
                queue.pop_back();
                for (Vertex w : curr.adjacentVertices()) {
                    if (marked[w]) continue;
                    triplets.emplace_back(m, geometry.vertexIndices[w], -1);
                    triplets.emplace_back(m, v0, 1);
                    marked[w] = true;
                    queue.push_back(w);
                    m++;
                }
            }
        }
        geometry.unrequireVertexIndices();
        A.resize(m, nVertices);
        A.setFromTriplets(triplets.begin(), triplets.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(nVertices + m);
        RHS.head(nVertices) = div;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = soln.head(nVertices);
        double shift = averageVertexDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nVertices);
    } else {
        if (options.rebuild || poissonSolver == nullptr) {
            if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
            poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
        }
        phi = poissonSolver->solve(div);
        double shift = averageVertexDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nVertices);
    }

    return phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorFieldToFaces(VertexPositionGeometry& geometry,
                                                                const Eigen::MatrixXd& Yt,
                                                                const SignedHeat3DOptions& options) {

    geometry.requireFaceIndices();

    SurfaceMesh& mesh = geometry.mesh;
    Vector<double> div = faceDivergence(Yt);
    Vector<double> phi;
    laplaceCR = buildCrouzeixRaviartLaplacian();
    if (options.levelSetConstraint == LevelSetConstraint::ZeroSet) {
        // Since the tet mesh conforms to the surface, preserving zero can be done via Dirichlet boundary conditions.
        Vector<bool> setAMembership = Vector<bool>::Ones(nFaces);
        for (const int& fIdx : surfaceFaces) setAMembership[abs(fIdx)] = false;
        int nB = nFaces - setAMembership.cast<int>().sum();
        Vector<double> bcVals = Vector<double>::Zero(nB);
        BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceCR, setAMembership, true);
        Vector<double> rhsValsA, rhsValsB;
        decomposeVector(decomp, div, rhsValsA, rhsValsB);
        Vector<double> combinedRHS = rhsValsA;
        Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
        phi = reassembleVector(decomp, Aresult, bcVals);
    } else if (options.levelSetConstraint == LevelSetConstraint::Multiple) {
        // Determine the connected components of the mesh. Do simple depth-first search.
        std::vector<Eigen::Triplet<double>> triplets;
        SparseMatrix<double> A;
        size_t m = 0;
        size_t F = mesh.nFaces();
        FaceData<bool> marked(mesh, false);
        geometry.requireFaceIndices();
        for (Face f : mesh.faces()) {
            if (marked[f]) continue;
            marked[f] = true;
            std::vector<Face> queue = {f};
            size_t f0 = geometry.faceIndices[f];
            Face curr;
            while (!queue.empty()) {
                curr = queue.back();
                queue.pop_back();
                for (Face g : curr.adjacentFaces()) {
                    if (marked[g]) continue;
                    triplets.emplace_back(m, geometry.faceIndices[g], -1);
                    triplets.emplace_back(m, f0, 1);
                    marked[g] = true;
                    queue.push_back(g);
                    m++;
                }
            }
        }
        geometry.unrequireFaceIndices();
        A.resize(m, nFaces);
        A.setFromTriplets(triplets.begin(), triplets.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceCR, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(nFaces + m);
        RHS.head(nFaces) = div;
        Vector<double> soln = solveSquare(LHS, RHS);
        phi = soln.head(nFaces);
        double shift = averageFaceDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nFaces);
    } else {
        if (options.rebuild || poissonSolverCR == nullptr) {
            if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
            poissonSolverCR.reset(new PositiveDefiniteSolver<double>(laplaceCR));
        }
        phi = poissonSolverCR->solve(div);
        double shift = averageFaceDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nFaces);
    }

    if (options.rebuild || projectionSolver == nullptr) {
        massMat = buildCrouzeixRaviartMassMatrix();
        avgMat = buildAveragingMatrix();
        SparseMatrix<double> P = avgMat.transpose() * massMat * avgMat;
        projectionSolver.reset(new SquareSolver<double>(P));
    }
    phi = projectOntoVertices(phi);

    geometry.unrequireFaceIndices();

    return -phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorField(pointcloud::PointPositionGeometry& pointGeom,
                                                         const Eigen::MatrixXd& Yt,
                                                         const SignedHeat3DOptions& options) {

    Vector<double> phi;
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            if (options.rebuild || poissonSolver == nullptr) {
                if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
                poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
            }
            Vector<double> div = vertexDivergence(Yt);
            phi = poissonSolver->solve(div);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            Vector<double> div = vertexDivergence(Yt);
            size_t P = pointGeom.cloud.nPoints();
            Vector<bool> setAMembership = Vector<bool>::Ones(nVertices);
            for (size_t i = 0; i < P; i++) setAMembership[i] = false;
            int nB = nVertices - setAMembership.cast<int>().sum();
            Vector<double> bcVals = Vector<double>::Zero(nB);
            BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceMat, setAMembership, true);
            Vector<double> rhsValsA, rhsValsB;
            decomposeVector(decomp, div, rhsValsA, rhsValsB);
            Vector<double> combinedRHS = rhsValsA;
            // shiftDiagonal(decomp.AA, 1e-8);
            Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
            phi = reassembleVector(decomp, Aresult, bcVals);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            Vector<double> div = vertexDivergence(Yt);
            std::vector<Eigen::Triplet<double>> triplets;
            SparseMatrix<double> A;
            size_t m = 0;
            size_t P = pointGeom.cloud.nPoints();
            VertexData<bool> marked(pointGeom.tuftedGeom->mesh, Vector<bool>::Zero(P));
            pointGeom.tuftedGeom->requireVertexIndices();
            for (Vertex v : pointGeom.tuftedGeom->mesh.vertices()) {
                if (marked[v]) continue;
                marked[v] = true;
                std::vector<Vertex> queue = {v};
                size_t v0 = pointGeom.tuftedGeom->vertexIndices[v];
                Vertex curr;
                while (!queue.empty()) {
                    curr = queue.back();
                    queue.pop_back();
                    for (Vertex w : curr.adjacentVertices()) {
                        if (marked[w]) continue;
                        triplets.emplace_back(m, pointGeom.tuftedGeom->vertexIndices[w], -1);
                        triplets.emplace_back(m, v0, 1);
                        marked[w] = true;
                        queue.push_back(w);
                        m++;
                    }
                }
            }
            pointGeom.tuftedGeom->unrequireVertexIndices();
            A.resize(m, nVertices);
            A.setFromTriplets(triplets.begin(), triplets.end());
            SparseMatrix<double> Z(m, m);
            SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
            SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
            SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
            Vector<double> RHS = Vector<double>::Zero(nVertices + m);
            RHS.head(nVertices) = div;
            // shiftDiagonal(LHS, 1e-16);
            Vector<double> soln = solveSquare(LHS, RHS);
            phi = soln.head(nVertices);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
    }
    return phi;
}

/* Integrate using breadth-first search. */
Vector<double> SignedHeatTetSolver::integrateVectorFieldGreedily(VertexPositionGeometry& geometry,
                                                                 const Eigen::MatrixXd& Yt,
                                                                 const SignedHeat3DOptions& options) {

    Vector<double> phi(nVertices);
    SurfaceMesh& mesh = geometry.mesh;
    size_t V = mesh.nVertices();
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            phi[0] = 0;
            visited[0] = true;
            integrateGreedily(Yt, visited, phi);
            double shift = averageVertexDataOnSource(geometry, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            // Fix solution values on source geometry.
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            for (size_t i = 0; i < V; i++) {
                phi[i] = 0;
                visited[i] = true;
            }
            integrateGreedily(Yt, visited, phi);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            phi = integrateGreedilyMultipleLevelSets(geometry, Yt);
            break;
        }
    }
    return phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorFieldGreedily(pointcloud::PointPositionGeometry& pointGeom,
                                                                 const Eigen::MatrixXd& Yt,
                                                                 const SignedHeat3DOptions& options) {

    Vector<double> phi(nVertices);
    size_t P = pointGeom.cloud.nPoints();
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            phi[0] = 0;
            visited[0] = true;
            integrateGreedily(Yt, visited, phi);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            for (size_t i = 0; i < P; i++) {
                phi[i] = 0;
                visited[i] = true;
            }
            integrateGreedily(Yt, visited, phi);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            phi = integrateGreedilyMultipleLevelSets(*(pointGeom.tuftedGeom), Yt);
            break;
        }
    }
    return phi;
}

void SignedHeatTetSolver::integrateGreedily(const Eigen::MatrixXd& Yt, Vector<bool>& visited,
                                            Vector<double>& phi) const {

    // Start queue with one of the surface vertices; we're assuming that the tetmesh domain is connected.
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        Eigen::Vector3d p = vertices.row(curr);
        queue.pop();
        for (size_t tIdx : vertexTet[curr]) {
            for (int j = 0; j < 4; j++) {
                size_t neighbor = tets(tIdx, j);
                if (visited[neighbor]) continue;
                Eigen::Vector3d q = vertices.row(neighbor);
                Eigen::Vector3d edge = q - p;
                Eigen::Vector3d Y = Yt.row(tIdx);
                phi[neighbor] = phi[curr] + Y.dot(edge);
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
}

Vector<double> SignedHeatTetSolver::integrateGreedilyMultipleLevelSets(IntrinsicGeometryInterface& geometry,
                                                                       const Eigen::MatrixXd& Yt) const {

    // Determine mesh components.
    SurfaceMesh& mesh = geometry.mesh;
    geometry.requireVertexIndices();
    std::vector<int> meshComponent(mesh.nVertices(), -1);
    Vector<bool> visited = Vector<bool>::Zero(nVertices);
    Vector<double> phi(nVertices);
    size_t cptIdx = 0;
    for (Vertex v : mesh.vertices()) {
        size_t vIdx = geometry.vertexIndices[v];
        if (meshComponent[vIdx] != -1) continue;
        meshComponent[vIdx] = cptIdx;
        std::vector<Vertex> queue = {v};
        if (cptIdx == 0) phi[vIdx] = 0;
        while (!queue.empty()) {
            Vertex curr = queue.back();
            queue.pop_back();
            for (Vertex w : curr.adjacentVertices()) {
                size_t wIdx = geometry.vertexIndices[w];
                if (meshComponent[wIdx] != -1) continue;
                meshComponent[wIdx] = cptIdx;
                if (cptIdx == 0) phi[wIdx] = 0;
                queue.push_back(w);
            }
        }
        cptIdx++;
    }
    geometry.unrequireVertexIndices();

    // integrate
    size_t V = mesh.nVertices();
    std::vector<bool> componentVisited(cptIdx, false);
    std::vector<double> componentValue(cptIdx);
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        Eigen::Vector3d p = vertices.row(curr);
        queue.pop();
        for (size_t tIdx : vertexTet[curr]) {
            for (int j = 0; j < 4; j++) {
                size_t neighbor = tets(tIdx, j);
                if (visited[neighbor]) continue;
                if ((neighbor < V) && componentVisited[meshComponent[neighbor]]) {
                    phi[neighbor] = componentValue[meshComponent[neighbor]];
                } else {
                    Eigen::Vector3d q = vertices.row(neighbor);
                    Eigen::Vector3d edge = q - p;
                    Eigen::Vector3d Y = Yt.row(tIdx);
                    phi[neighbor] = phi[curr] + Y.dot(edge);
                    if (neighbor < V) {
                        componentVisited[meshComponent[neighbor]] = true;
                        componentValue[meshComponent[neighbor]] = phi[neighbor];
                    }
                }
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
    return phi;
}

double SignedHeatTetSolver::averageFaceDataOnSource(VertexPositionGeometry& geometry, const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0.;
    for (const auto& fIdx : surfaceFaces) {
        size_t i = abs(fIdx);
        Eigen::Vector3d a = vertices.row(faces(i, 0));
        Eigen::Vector3d b = vertices.row(faces(i, 1));
        Eigen::Vector3d c = vertices.row(faces(i, 2));
        double A = 0.5 * ((a - c).cross(b - c)).norm();
        shift += A * phi[i];
        totalArea += A;
    }
    shift /= totalArea;
    return shift;
}

double SignedHeatTetSolver::averageVertexDataOnSource(VertexPositionGeometry& geometry,
                                                      const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0.;
    geometry.requireVertexDualAreas();
    for (size_t i = 0; i < geometry.mesh.nVertices(); i++) {
        double A = geometry.vertexDualAreas[i];
        shift += A * phi[i];
        totalArea += A;
    }
    shift /= totalArea;
    geometry.unrequireVertexDualAreas();
    return shift;
}

double SignedHeatTetSolver::averageVertexDataOnSource(pointcloud::PointPositionGeometry& pointGeom,
                                                      const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0;
    size_t P = pointGeom.cloud.nPoints();
    for (size_t pIdx = 0; pIdx < P; pIdx++) {
        double A = pointGeom.tuftedGeom->vertexDualAreas[pIdx];
        shift += A * phi[pIdx];
        totalArea += A;
    }
    shift /= totalArea;
    return shift;
}

/*
 * Given a piecewise-constant vector field defined on tets, compute FEM integrated divergence per face.
 */
Vector<double> SignedHeatTetSolver::faceDivergence(const Eigen::MatrixXd& X) const {

    Vector<double> divX = Vector<double>::Zero(nFaces);
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            int sfIdx = tetFace(i, j);
            int fIdx = abs(sfIdx);
            Eigen::Vector3d N = areaWeightedNormalVector(sfIdx);
            divX[fIdx] += N.dot(X.row(i));
        }
    }
    return divX;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartLaplacian() const {

    SparseMatrix<double> L(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        for (int j = 0; j < 4; j++) {
            int sfA = tetFace(i, j);
            int fA = abs(sfA);
            Eigen::Vector3d nA = areaWeightedNormalVector(sfA);
            for (int k = j + 1; k < 4; k++) {
                int sfB = tetFace(i, k);
                int fB = abs(sfB);
                Eigen::Vector3d nB = areaWeightedNormalVector(sfB);
                double w = (nA.dot(nB)) / vol;
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
                triplets.emplace_back(fA, fA, -w);
                triplets.emplace_back(fB, fB, -w);
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartMassMatrix() const {

    SparseMatrix<double> M(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        // Iterate over all pairs of adjacent faces.
        double w = -0.05 * vol;
        for (int j = 0; j < 4; j++) {
            int fA = abs(tetFace(i, j));
            for (int k = j + 1; k < 4; k++) {
                int fB = abs(tetFace(i, k));
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
            }
            triplets.emplace_back(fA, fA, 0.4 * vol);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

/*
 * Compute the circumcenter of a tetrahedron, given its vertex positions.
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
void tetCircumcenter(const Eigen::Matrix<double, 4, 3>& t, Eigen::Vector3d& c) {

    Eigen::Matrix3d A;
    Eigen::Vector3d b;

    const double n0 = t.row(0).squaredNorm();

    for (int k = 0; k < 3; ++k) {
        A.row(k) = t.row(k + 1) - t.row(0);
        b(k) = t.row(k + 1).squaredNorm() - n0;
    }

    c = 0.5 * A.fullPivHouseholderQr().solve(b);
}

/*
 * Compute the circumcenter of a face, given its vertex positions.
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
void faceCircumcenter(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                      Eigen::Vector3d& cc) {

    const double l[3]{(b - c).squaredNorm(), (a - c).squaredNorm(), (a - b).squaredNorm()};

    const double ba[3]{l[0] * (l[1] + l[2] - l[0]), l[1] * (l[2] + l[0] - l[1]), l[2] * (l[0] + l[1] - l[2])};
    const double sum = ba[0] + ba[1] + ba[2];

    cc = (ba[0] / sum) * a + (ba[1] / sum) * b + (ba[2] / sum) * c;
}

/*
 * Build the dual Laplacian for the tet mesh from Alexa et al. 2020 (https://igl.ethz.ch/projects/LB3D/LB3D.pdf).
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
SparseMatrix<double> SignedHeatTetSolver::dualLaplacian() const {

    SparseMatrix<double> L(nVertices, nVertices);

    const int turn[4][4]{{-1, 2, 3, 1}, {3, -1, 0, 2}, {1, 3, -1, 0}, {2, 0, 1, -1}};

    auto getTet = [&](const int i, Eigen::Matrix<double, 4, 3>& t) {
        for (int k = 0; k < 4; ++k) {
            t.row(k) = vertices.row(tets(i, k));
        }
    };

    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Vector3d cc;
    Eigen::Matrix<double, 4, 3> t;

    for (size_t k = 0; k < nTets; k++) {
        // Compute the circumcenter of the tet.
        getTet(k, t);
        tetCircumcenter(t, cc);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    Eigen::Vector3d cf;
                    faceCircumcenter(t.row(i), t.row(j), t.row(turn[i][j]), cf);

                    const Eigen::Vector3d ce = 0.5 * (t.row(i) + t.row(j));

                    const double vol = tetVolume(t.row(i), ce, cf, cc);
                    const double wij = 6. * vol / (t.row(i) - t.row(j)).squaredNorm();

                    triplets.emplace_back(tets(k, i), tets(k, j), wij);
                    triplets.emplace_back(tets(k, j), tets(k, i), wij);
                    triplets.emplace_back(tets(k, i), tets(k, i), -wij);
                    triplets.emplace_back(tets(k, j), tets(k, j), -wij);
                }
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

Vector<double> SignedHeatTetSolver::vertexDivergence(const Eigen::MatrixXd& X) const {

    const int turn[4][4]{{-1, 2, 3, 1}, {3, -1, 0, 2}, {1, 3, -1, 0}, {2, 0, 1, -1}};
    auto getTet = [&](const int i, Eigen::Matrix<double, 4, 3>& t) {
        for (int k = 0; k < 4; ++k) {
            t.row(k) = vertices.row(tets(i, k));
        }
    };
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Vector3d cc;
    Eigen::Matrix<double, 4, 3> t;
    Vector<double> div = Vector<double>::Zero(nVertices);
    for (size_t k = 0; k < nTets; k++) {
        getTet(k, t);
        tetCircumcenter(t, cc);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    Eigen::Vector3d cf;
                    faceCircumcenter(t.row(i), t.row(j), t.row(turn[i][j]), cf);
                    int vA = tets(k, i);
                    int vB = tets(k, j);
                    Eigen::Vector3d a = vertices.row(vA);
                    Eigen::Vector3d b = vertices.row(vB);
                    Eigen::Vector3d e = b - a;
                    const Eigen::Vector3d ce = 0.5 * (t.row(i) + t.row(j));
                    const double vol = tetVolume(t.row(i), ce, cf, cc);
                    const double wij = 6. * vol / (t.row(i) - t.row(j)).squaredNorm();
                    div[vA] += e.dot(X.row(k)) * wij;
                    div[vB] -= e.dot(X.row(k)) * wij;
                }
            }
        }
    }
    return div;
}

Vector<double> SignedHeatTetSolver::projectOntoVertices(const Vector<double>& u) const {

    SparseMatrix<double> At = avgMat.transpose();
    Vector<double> RHS = At * massMat * u;
    Vector<double> w = projectionSolver->solve(RHS);
    return w;
}

SparseMatrix<double> SignedHeatTetSolver::buildAveragingMatrix() const {

    SparseMatrix<double> A(nFaces, nVertices);
    std::vector<Eigen::Triplet<double>> triplets;
    double w = 1. / 3.;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            triplets.emplace_back(i, faces(i, j), w);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

void SignedHeatTetSolver::isosurface(std::unique_ptr<SurfaceMesh>& isoMesh,
                                     std::unique_ptr<VertexPositionGeometry>& isoGeom, const Vector<double>& phi,
                                     double isoval) const {

    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi J;
    Eigen::SparseMatrix<double> BC;
    igl::marching_tets(vertices, tets, phi, isoval, SV, SF, J, BC);
    std::tie(isoMesh, isoGeom) = makeSurfaceMeshAndGeometry(SV, SF);
}

// =============== TET UTILITIES

Eigen::VectorXd SignedHeatTetSolver::computeTetVolumes() const {

    Eigen::VectorXd volumes(nTets);
    for (size_t i = 0; i < nTets; i++) {
        volumes(i) = computeTetVolume(i);
    }
    return volumes;
}

double SignedHeatTetSolver::computeTetVolume(size_t tIdx) const {

    return tetVolume(vertices.row(tets(tIdx, 0)), vertices.row(tets(tIdx, 1)), vertices.row(tets(tIdx, 2)),
                     vertices.row(tets(tIdx, 3)));
}

double SignedHeatTetSolver::tetVolume(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                                      const Eigen::Vector3d& d) const {
    Eigen::Matrix3d A;
    A.col(0) = b - a;
    A.col(1) = c - a;
    A.col(2) = d - a;
    return A.determinant() / 6.;
}

/*
 * Return the area-weighted normal vector of the face with index abs(fIdx). The sign of `fIdx` gives the orientation of
 * the face relative to its (arbitrary but fixed) global orientation.
 */
Eigen::Vector3d SignedHeatTetSolver::areaWeightedNormalVector(int fIdx) const {

    int idx = abs(fIdx);
    Eigen::Vector3d a = vertices.row(faces(idx, 0));
    Eigen::Vector3d b = vertices.row(faces(idx, 1));
    Eigen::Vector3d c = vertices.row(faces(idx, 2));
    Eigen::Vector3d n = 0.5 * (a - c).cross(b - c);
    if (fIdx < 0) n *= -1;
    return n;
}

Eigen::Vector3d SignedHeatTetSolver::faceBarycenter(size_t fIdx) const {

    return (vertices.row(faces(fIdx, 0)) + vertices.row(faces(fIdx, 1)) + vertices.row(faces(fIdx, 2))) / 3.;
}

// =============== TET-MESHING

/*
 * Tetmesh the interior and exterior of the given surface inside a bounding box, s.t. the vertices of the surface are
 * preserved.
 *
 * TetGen allows you to tetmesh while preserving the input faces; this allows us to construct a correspondence between
 * vertices in the original surface, and vertices in the tetmesh. However, there's no way to preserve only some faces
 * and not others. This is a problem if we want to generate a tetmesh within a particular bounding cube. (Without
 * specifying a bounding box, TetGen will just tetmesh a convex hull.) The faces of the cube are incredibly large,
 * leading to a terribly coarse tetrahedralization. So first we triangulate the surface of the bounding cube. Then we
 * generate a tetmesh, with the faces of the bounding cube and the surface constrained, with the command that they
 * should all be preserved. However, the faces of the bounding cube should be sufficiently refined from the first step
 * that the resulting tets are small enough and of similar size to the ones everywhere else.
 */
bool SignedHeatTetSolver::tetmeshDomain(VertexPositionGeometry& geometry) {

    SurfaceMesh& mesh = geometry.mesh;

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    Vector3 geomCentroid = centroid(geometry);
    double geomRadius = radius(geometry, geomCentroid);
    triangulateCube(cubeSurface, geomCentroid, geomRadius);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    // Create a constrained tetmesh of the surface, without changing any of the input faces itself.
    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    in.firstnumber = 0;
    in.numberofpoints = mesh.nVertices() + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < mesh.nVertices(); i++) {
        Vector3 pos = geometry.inputVertexPositions[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[mesh.nVertices() + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * mesh.nVertices() + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = mesh.nFaces() + cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy faces from input surface mesh.
    geometry.requireVertexIndices();
    for (size_t i = 0; i < mesh.nFaces(); i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        int j = 0;
        for (Vertex v : mesh.face(i).adjacentVertices()) {
            p->vertexlist[j] = geometry.vertexIndices[v];
            in.trifacelist[3 * i + j] = geometry.vertexIndices[v];
            j++;
        }
    }
    geometry.unrequireVertexIndices();
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[mesh.nFaces() + i] = 0;
        f = &in.facetlist[mesh.nFaces() + i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * mesh.nFaces() + 3 * i + j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
        }
    }

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
        return false;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        return false;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
        return false;
    }
    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);

    // Determine the face ids in the tetmesh corresponding to the original input surface.
    // The indices of marked faces are not preserved in the final tet mesh. However, indices of marked points
    // (vertices) are. So we can match faces in the tetmesh to faces in the input surface mesh by comparing their
    // vertex indices.
    surfaceFaces.clear();
    int nConstraints = 0;
    geometry.requireVertexIndices();
    for (size_t i = 0; i < nFaces; i++) {
        if (out.trifacemarkerlist[i]) {
            // Determine orientation.
            int sign = 1;
            Vertex vA = mesh.vertex(faces(i, 0));
            for (Halfedge he : vA.outgoingHalfedges()) {
                size_t vBIdx = geometry.vertexIndices[he.tipVertex()];
                size_t vCIdx = geometry.vertexIndices[he.next().tipVertex()];
                if (vBIdx == faces(i, 1) && vCIdx == faces(i, 2)) {
                    sign = 1;
                    break;
                }
                if (vBIdx == faces(i, 2) && vCIdx == faces(i, 1)) {
                    sign = -1;
                    break;
                }
            }
            surfaceFaces.push_back(sign * i);
            nConstraints++;
        }
    }
    geometry.unrequireVertexIndices();

    // Display the tetmesh in the GUI.
    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("domain", vertices, tets);
    return true;
}

/*
 Xue: This is the original/actual tetrahedralize function
 */
void SignedHeatTetSolver::tetmeshPointCloud(pointcloud::PointPositionGeometry& pointGeom) {

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    Vector3 geomCentroid = centroid(pointGeom);
    double geomRadius = radius(pointGeom, geomCentroid);
    triangulateCube(cubeSurface, geomCentroid, geomRadius);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    size_t P = pointGeom.cloud.nPoints();
    in.firstnumber = 0;
    in.numberofpoints = P + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < pointGeom.cloud.nPoints(); i++) {
        Vector3 pos = pointGeom.positions[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[P + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * P + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[i] = 0;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = P + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * i + j] = P + cubeSurface.trifacelist[3 * i + j];
        }
    }

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
    }

    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);

    // Display the tetmesh in the GUI.
    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("domain", vertices, tets);
//    polyscope::show();

    if (VERBOSE) std::cout << "tetmeshPointCloud tetrahedralization completed" << std::endl;
    
    
}


/*
 * Generate a constrained Delaunay tetrahedralization of a cube surrounding the input surface mesh.
 * Return only the boundary of the cube.
 */
void SignedHeatTetSolver::triangulateCube(tetgenio& cubeSurface, const Vector3& centroid, const double& radius,
                                          double scale) const {

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    tetmeshCube(in, out, centroid, radius, scale);

    // Determine which faces/vertices lie on the boundary.
    std::vector<int> fIdx; // indices of boundary faces in tetmesh
    Eigen::VectorXi vMap =
        -1 * Eigen::VectorXi::Ones(out.numberofpoints); // Map tet mesh vertex indices to new indexing.
    std::set<int> vSet;                                 // Map surface mesh vertex indices to tetmesh indices
    for (int i = 0; i < out.numberoftrifaces; i++) {
        if (out.trifacemarkerlist[i] == 1) {
            fIdx.push_back(i);
            for (int j = 0; j < 3; j++) {
                // have to do this way, because vertices added along edges don't inherit the boundary marker... argh
                vSet.insert(out.trifacelist[3 * i + j]);
            }
        }
    }
    std::vector<int> vIdx;
    for (int i : vSet) {
        vMap(i) = vIdx.size();
        vIdx.push_back(i);
    }

    cubeSurface.firstnumber = 0;
    cubeSurface.numberofpoints = vIdx.size();
    cubeSurface.pointlist = new REAL[cubeSurface.numberofpoints * 3];
    cubeSurface.pointmarkerlist = new int[cubeSurface.numberofpoints];
    cubeSurface.numberoffacets = fIdx.size();
    cubeSurface.facetlist = new tetgenio::facet[cubeSurface.numberoffacets];
    cubeSurface.facetmarkerlist = new int[cubeSurface.numberoffacets];
    cubeSurface.numberoftrifaces = fIdx.size();
    cubeSurface.trifacelist = new int[cubeSurface.numberoftrifaces * 3];
    // Define nodes.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        for (int j = 0; j < 3; j++) {
            cubeSurface.pointlist[3 * i + j] = out.pointlist[3 * vIdx[i] + j];
        }
    }

    // Define faces.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        f = &cubeSurface.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
            cubeSurface.trifacelist[3 * i + j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
        }
    }
}

void SignedHeatTetSolver::tetmeshCube(tetgenio& in, tetgenio& out, const Vector3& centroid, const double& radius,
                                      double scale) const {

    tetgenio::facet* f;
    tetgenio::polygon* p;

    // All indices start from 0.
    in.firstnumber = 0;
    in.numberofpoints = 8; // there are 8 vertices of a cube
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];

    // Define nodes.
    std::vector<Vector3> cubeCorners = buildCubeAroundSurface(centroid, radius, scale);

    for (int i = 0; i < in.numberofpoints; i++) {
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = cubeCorners[i][j];
        }
    }

    // Define facets.
    in.numberoffacets = 6;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];

    int cubeIndices[6][4] = {
        {0, 1, 2, 3}, // bottom face
        {4, 5, 6, 7}, // top face
        {0, 1, 5, 4}, // left face
        {3, 2, 6, 7}, // right face
        {0, 3, 7, 4}, // front face
        {1, 2, 6, 5}  // back face
    };

    for (int i = 0; i < in.numberoffacets; i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 4;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 4; j++) {
            p->vertexlist[j] = cubeIndices[i][j];
        }
    }

    tetrahedralize(const_cast<char*>(TETFLAGS.c_str()), &in, &out);
}

/*
 * Construct a cube around the input surface mesh.
 * Returns the 3D positions of the 8 corners of the cube.
 */
std::vector<Vector3> SignedHeatTetSolver::buildCubeAroundSurface(const Vector3& centroid, const double& radius,
                                                                 double scale) const {

    // make the side length of the cube big enough to surround the entire mesh.
    double s = radius * scale;

    std::vector<Vector3> cubeCorners = {
        {-s, -s, -s}, // bottom lower left corner
        {-s, -s, s},  // bottom upper left
        {s, -s, s},   // bottom upper right
        {s, -s, -s},  // bottom lower right
        {-s, s, -s},  // upper lower left corner
        {-s, s, s},   // upper upper left
        {s, s, s},    // upper upper right
        {s, s, -s}    // upper lower right
    };
    for (size_t i = 0; i < 8; i++) cubeCorners[i] += centroid;

    return cubeCorners;
}

void SignedHeatTetSolver::getTetmeshData(tetgenio& out) {

    nVertices = out.numberofpoints;
    nTets = out.numberoftetrahedra;
    nFaces = out.numberoftrifaces;
    nEdges = out.numberofedges;
    // out.numberofcorners is 4
    if (VERBOSE) std::cerr << "# of vertices: " << nVertices << std::endl;
    if (VERBOSE) std::cerr << "# of tets: " << nTets << std::endl;
    if (VERBOSE) std::cerr << "# of facets: " << out.numberoffacets << std::endl;
    if (VERBOSE) std::cerr << "# of tri-faces: " << out.numberoftrifaces << std::endl; // # of constrained faces
    if (VERBOSE) std::cerr << "# of edges: " << nEdges << std::endl;
    vertices.resize(nVertices, 3);
    tets.resize(nTets, 4);
    faces.resize(nFaces, 3);

    // Determine element-vertex matrices.
    for (size_t i = 0; i < nVertices; i++) {
        for (int j = 0; j < 3; j++) {
            vertices(i, j) = out.pointlist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`vertices` constructed" << std::endl;
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            tets(i, j) = out.tetrahedronlist[4 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`tets` constructed" << std::endl;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            faces(i, j) = out.trifacelist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`faces` constructed" << std::endl;

    // Determine adjacency info.
    tetFace.resize(nTets, 4);
    for (size_t i = 0; i < nTets; i++) {
        // All tets should already be positively oriented.
        Eigen::MatrixXi tetFaces(4, 3); // oriented faces in the tet
        tetFaces.row(0) << tets(i, 0), tets(i, 1), tets(i, 2);
        tetFaces.row(1) << tets(i, 0), tets(i, 3), tets(i, 1);
        tetFaces.row(2) << tets(i, 0), tets(i, 2), tets(i, 3);
        tetFaces.row(3) << tets(i, 1), tets(i, 3), tets(i, 2);
        for (int j = 0; j < 4; j++) {
            int fIdx = out.tet2facelist[4 * i + j];
            // Determine orientation (slow way)
            int s = -1;
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 3; l++) {
                    if (faces(fIdx, 0) == tetFaces(k, (0 + l) % 3) && faces(fIdx, 1) == tetFaces(k, (1 + l) % 3) &&
                        faces(fIdx, 2) == tetFaces(k, (2 + l) % 3)) {
                        s = 1;
                        break;
                    }
                }
            }
            tetFace(i, j) = s * fIdx;
        }
    }
    vertexTet.clear();
    vertexTet.resize(nVertices);
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            vertexTet[tets(i, j)].insert(i);
        }
    }
    if (VERBOSE) std::cerr << "Adjacency structures constructed" << std::endl;
}

double SignedHeatTetSolver::computeMeanNodeSpacing() const {

    double h = 0.;
    for (size_t i = 0; i < nTets; i++) {
        Eigen::MatrixXd faceBarycenters(4, 3);
        for (int j = 0; j < 4; j++) {
            faceBarycenters.row(j) = faceBarycenter(abs(tetFace(i, j)));
        }
        for (int j = 0; j < 4; j++) {
            for (int k = j + 1; k < 4; k++) {
                h += (faceBarycenters.row(j) - faceBarycenters.row(k)).norm();
            }
        }
    }
    h /= 6 * nTets;
    return h;
}

double SignedHeatTetSolver::calculateAverageEdgeLength(const EdgeDualNormalGeometry& edgeGeom) {
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

/*
 Xue: Trying to switch to CDT to tetrahedralize
 */
void SignedHeatTetSolver::tetmeshEdgeGeometryCDT(EdgeDualNormalGeometry& edgeGeom, const SignedHeat3DOptions& options) {
    
    if (VERBOSE) std::cout << "Using CDT for EdgeDualNormalGeometry tetrahedralization..." << std::endl;
    
    // Step 1: Extract vertices and edges from EdgeDualNormalGeometry
    const auto& vertices_data = edgeGeom.getVertices();   // std::vector<Vector3>
    const auto& edges_data = edgeGeom.getEdges();        // std::vector<std::pair<size_t, size_t>>
    
    size_t nVertices = vertices_data.size();
    size_t nEdges = edges_data.size();
    
    if (VERBOSE) std::cout << "Input geometry: " << nVertices << " vertices, "
                          << nEdges << " edges" << std::endl;
    
    // Step 2: Create PLC object and initialize from dual edge object
    inputPLC plc;
    bool success = convertEdgeGeomToPLC(edgeGeom, plc, true);
    
    if (success) {
        printf("Conversion successful!\n");
        printf("PLC has %u vertices and %u edges\n",
               plc.numVertices(), plc.numEdges());
    } else {
        printf("Conversion failed!\n");
    }
    
    // Step 3: Add bounding box
    // there's conversion between 
    double bbox_expansion = options.scale - 1;
//    plc.addBoundingBoxVertices(bbox_expansion);
//
    double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
    // Add face grid points
        
    
    
    if (VERBOSE) {
        std::cout << "Final PLC: "
                  << plc.numVertices() << " vertices, "
                  << plc.numEdges() << " edges" << std::endl;
    }
    
    // Step 4: Set CDT options and execute tetrahedralization
    std::string cdtOptions = "v";  // bounding box + verbose
    

    TetMesh* tetMesh = nullptr;
    try {
        tetMesh = createSteinerCDT(plc, cdtOptions, std::to_string(bbox_expansion));
        std::cout << "CDT tetrahedralization completed " << std::endl;
        if (VERBOSE) std::cout << "CDT tetrahedralization completed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CDT failed: " << e.what() << std::endl;
        throw;
    }
    
    tetMesh->saveTET("cdt.tet", false);

    
    convertTetMeshForVisualization(tetMesh);
    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("domain", vertices, tets);
    polyscope::show();
    
//    tetgenio out;
//    convertTetMeshToTetgenio(tetMesh, out);
//    getTetmeshData(out);

    
    
//   *** 新增部分：使用 TetGen 后处理 CDT 结果生成 pointPolyGeom ***
   if (VERBOSE) std::cout << "Post-processing CDT result with TetGen..." << std::endl;
   
   // 使用与 USETetgen 分支相同的约束计算
   double meanArea = meanEdgeLength; // Use edge length as proxy for area
   double areaScale = std::pow(2, -options.hCoef);
   
   // 使用 TetGen 优化 CDT 结果
   feedTetMeshToTetGenOptimized(*tetMesh, edgeGeom, options, areaScale * meanArea);
   

   // *** 新增部分结束 ***

    
    psVolumeMesh = polyscope::registerTetMesh("domain", vertices, tets);
//    polyscope::show();
    
    if (VERBOSE) std::cout << "CDT EdgeDualNormalGeometry tetrahedralization completed" << std::endl;
   
    delete tetMesh;
}




/*
 Xue :
 */
//bool SignedHeatTetSolver::convertEdgeGeomToPLC(const geometrycentral::EdgeDualNormalGeometry& edgeGeom, inputPLC& plc, bool verbose = false) {
//    // 获取几何数据
//    const auto& vertices = edgeGeom.getVertices();
//    const auto& edges = edgeGeom.getEdges();
//
//    // 验证输入
//    if (vertices.empty()) {
//        if (verbose) printf("Error: No vertices in EdgeDualNormalGeometry\n");
//        return false;
//    }
//
//    // 转换顶点数据
//    size_t npts = vertices.size();
//    double* vertex_p = (double*)malloc(npts * 3 * sizeof(double));
//    for (size_t i = 0; i < npts; i++) {
//        vertex_p[i * 3 + 0] = vertices[i].x;
//        vertex_p[i * 3 + 1] = vertices[i].y;
//        vertex_p[i * 3 + 2] = vertices[i].z;
//    }
//
//    // 转换边数据
//    size_t nedges = edges.size();
//    uint32_t* edge_vertices_p = nullptr;
//    if (nedges > 0) {
//        edge_vertices_p = (uint32_t*)malloc(nedges * 2 * sizeof(uint32_t));
//        for (size_t i = 0; i < nedges; i++) {
//            edge_vertices_p[i * 2 + 0] = (uint32_t)edges[i].first;
//            edge_vertices_p[i * 2 + 1] = (uint32_t)edges[i].second;
//        }
//    }
//
//    // 初始化 PLC
//    plc.input_file_name = "";
//    plc.postProcess(vertex_p, (uint32_t)npts, nullptr, 0, edge_vertices_p, (uint32_t)nedges, verbose);
//
//    // 清理内存
//    free(vertex_p);
//    if (edge_vertices_p) free(edge_vertices_p);
//
//    return true;
//}

bool SignedHeatTetSolver::convertEdgeGeomToPLC(const geometrycentral::EdgeDualNormalGeometry& edgeGeom, inputPLC& plc, bool verbose = false) {
    // 获取几何数据
    const auto& vertices = edgeGeom.getVertices();
    const auto& edges = edgeGeom.getEdges();
    
    // 验证输入
    if (vertices.empty()) {
        if (verbose) printf("Error: No vertices in EdgeDualNormalGeometry\n");
        return false;
    }
    
    // 转换顶点数据
    size_t npts = vertices.size();
    double* vertex_p = (double*)malloc(npts * 3 * sizeof(double));
    for (size_t i = 0; i < npts; i++) {
        vertex_p[i * 3 + 0] = vertices[i].x;
        vertex_p[i * 3 + 1] = vertices[i].y;
        vertex_p[i * 3 + 2] = vertices[i].z;
    }
    
    // 转换边数据
    size_t nedges = edges.size();
    uint32_t* edge_vertices_p = nullptr;
    if (nedges > 0) {
        edge_vertices_p = (uint32_t*)malloc(nedges * 2 * sizeof(uint32_t));
        for (size_t i = 0; i < nedges; i++) {
            edge_vertices_p[i * 2 + 0] = (uint32_t)edges[i].first;
            edge_vertices_p[i * 2 + 1] = (uint32_t)edges[i].second;
        }
    }
    
    // 初始化 PLC
    plc.input_file_name = "";
    plc.postProcess(vertex_p, (uint32_t)npts, nullptr, 0, edge_vertices_p, (uint32_t)nedges, verbose);
    
    // 清理内存
    free(vertex_p);
    if (edge_vertices_p) free(edge_vertices_p);
    
    // 添加边界框
    double bbox_expansion = 0.5; // 或者从参数传入
    plc.addBoundingBoxVertices(bbox_expansion);
    
    // 添加网格点
    double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
    
    // Calculate bounding box bounds (similar to addBoundingBoxVertices logic)
    double bbmin[3] = { DBL_MAX, DBL_MAX, DBL_MAX };
    double bbmax[3] = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
    for (uint32_t i = 0; i < npts; i++) { // use original vertices only
        const auto& v = vertices[i];
        if (v.x < bbmin[0]) bbmin[0] = v.x;
        if (v.x > bbmax[0]) bbmax[0] = v.x;
        if (v.y < bbmin[1]) bbmin[1] = v.y;
        if (v.y > bbmax[1]) bbmax[1] = v.y;
        if (v.z < bbmin[2]) bbmin[2] = v.z;
        if (v.z > bbmax[2]) bbmax[2] = v.z;
    }
    const double bbox[3] = { bbmax[0] - bbmin[0], bbmax[1] - bbmin[1], bbmax[2] - bbmin[2] };
    for (int j = 0; j < 3; j++) {
        bbmin[j] -= bbox[j] * bbox_expansion;
        bbmax[j] += bbox[j] * bbox_expansion;
    }
    
    // Calculate grid divisions based on mean edge length
    double box_length = std::max({bbox[0], bbox[1], bbox[2]}); // use largest dimension
    int N = std::max(2, (int)(box_length / meanEdgeLength * 0.8));

    if (verbose) {
        printf("Grid divisions: %d (box_length: %f, meanEdgeLength: %f)\n",
               N, box_length, meanEdgeLength);
    }

    double x_grid_size = (bbmax[0] - bbmin[0]) / N;
    double y_grid_size = (bbmax[1] - bbmin[1]) / N;
    double z_grid_size = (bbmax[2] - bbmin[2]) / N;
//
//    int added_grid_points = 0;
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            for (int k = 0; k < N; k++) {
//                // Calculate grid cell bounds
//                double cell_x_min = bbmin[0] + i * x_grid_size;
//                double cell_y_min = bbmin[1] + j * y_grid_size;
//                double cell_z_min = bbmin[2] + k * z_grid_size;
//                double cell_x_max = bbmin[0] + (i + 1) * x_grid_size;
//                double cell_y_max = bbmin[1] + (j + 1) * y_grid_size;
//                double cell_z_max = bbmin[2] + (k + 1) * z_grid_size;
//
//                // Check if any existing vertex is in this cell
//                bool has_existing_point = false;
//                for (const auto& vertex : vertices) {
//                    if (vertex.x >= cell_x_min && vertex.x <= cell_x_max &&
//                        vertex.y >= cell_y_min && vertex.y <= cell_y_max &&
//                        vertex.z >= cell_z_min && vertex.z <= cell_z_max) {
//                        has_existing_point = true;
//                        break;
//                    }
//                }
//
//                // If no existing point in this cell, add a point at cell center
//                if (!has_existing_point) {
//                    double grid_x = cell_x_min + (cell_x_max - cell_x_min) * 0.5;
//                    double grid_y = cell_y_min + (cell_y_max - cell_y_min) * 0.5;
//                    double grid_z = cell_z_min + (cell_z_max - cell_z_min) * 0.5;
//
//                    // Add vertex to PLC coordinates directly
//                    plc.coordinates.push_back(grid_x);
//                    plc.coordinates.push_back(grid_y);
//                    plc.coordinates.push_back(grid_z);
//                    added_grid_points++;
//                }
//            }
//        }
//    }
//
//    if (verbose) {
//        printf("Added %d grid points to bounding box\n", added_grid_points);
//    }

    return true;
}


TetMesh* SignedHeatTetSolver::createSteinerCDT(inputPLC& plc, const std::string& options, const std::string& bbox_expansion_fraction ) {
    bool log = false, bbox = false, verbose = false, snap = false, logscreen = false;
    //bool optimize = false;

    for (int i = 0; i < options.size(); i++) switch (options[i]) {
    case 'b':
        bbox = true; break;
    case 'v':
        verbose = true; break;
    //case 'o':
    //    optimize = true; break;
    } // Just ignore unknown options
    
    bbox = false;
    if (bbox) plc.addBoundingBoxVertices( std::stod(bbox_expansion_fraction) );

    if (logscreen) {
        log = true;
    }

    // Build a delaunay tetrahedrization of the vertices
    TetMesh  *tin = new TetMesh;
    tin->init_vertices(plc.coordinates.data(), plc.numVertices());
    tin->tetrahedrize();

    if (verbose) printf("DT of the vertices built\n");

    // Build a structured PLC linked to the Delaunay tetrahedrization
    PLCx Steiner_plc(
          *tin,
          plc.triangle_vertices.data(),
          plc.numTriangles(),
          plc.edge_vertices.data(),
          plc.numEdges());

    // Recover segments by inserting Steiner points in both the PLC and the tetrahedrization
    Steiner_plc.segmentRecovery_HSi(!verbose);


    // Recover PLC faces by locally remeshing the tetrahedrization
    bool sisMethodWorks = Steiner_plc.faceRecovery(!verbose);


    // Mark the tets which are bounded by the PLC.
    // If the PLC is not a valid polyhedron (i.e. it has odd-valency edges)
    // all the tets but the ghosts are marked as "internal".
    uint32_t num_inner_tets = (uint32_t)Steiner_plc.markInnerTets();



    if (snap) {
        if (!tin->optimizeNearDegenerateTets(verbose)) {
            std::cerr << "Could not force FP representability.\n";
        }
    }

    //if (optimize) tin->optimizeMesh();

    return tin;
}

/*
 Xue:
 */

void SignedHeatTetSolver::convertTetMeshToTetgenio(TetMesh* tetMesh, tetgenio& out) {
    if (!tetMesh) {
        std::cerr << "Error: TetMesh is null" << std::endl;
        return;
    }
    
    VERBOSE = true;
    
    if (VERBOSE) std::cout << "Converting TetMesh to tetgenio format..." << std::endl;
    
    // Initialize tetgenio
    out.initialize();
    
    // Include necessary headers for map and algorithm
    #include <map>
    #include <algorithm>
    #include <tuple>
    #include <array>
    
    // Get basic counts
    uint32_t nVertices = tetMesh->numVertices();
    uint32_t nTets = tetMesh->countNonGhostTets();
    
    if (VERBOSE) {
        std::cout << "TetMesh contains:" << std::endl;
        std::cout << "  Vertices: " << nVertices << std::endl;
        std::cout << "  Non-ghost Tetrahedra: " << nTets << std::endl;
    }
    
    // Set vertex information
    out.firstnumber = 0;  // 0-based indexing
    out.numberofpoints = nVertices;
    out.pointlist = new REAL[nVertices * 3];
    
    // Extract vertices from TetMesh
    for (uint32_t i = 0; i < nVertices; i++) {
        // Get vertex coordinates - need to check TetMesh API for correct method
        double coords[3];
        tetMesh->vertices[i]->getApproxXYZCoordinates(coords[0], coords[1], coords[2], true);
        
        out.pointlist[i * 3 + 0] = coords[0];
        out.pointlist[i * 3 + 1] = coords[1];
        out.pointlist[i * 3 + 2] = coords[2];
    }
    
    // Set tetrahedra information
    out.numberoftetrahedra = nTets;
    out.tetrahedronlist = new int[nTets * 4];
    
    // Extract tetrahedra from TetMesh
    uint32_t tetIdx = 0;
    for (uint32_t i = 0; i < tetMesh->numTets(); i++) {
        if (!tetMesh->isGhost(i)) {  // Only include non-ghost tets
            // Get tetrahedron vertices - access tet_node array directly
            const uint32_t* tetVertices = tetMesh->tet_node.data() + (i * 4);
            
            // Skip infinite vertices (ghost tetrahedra)
            if (tetVertices[3] != INFINITE_VERTEX) {
                out.tetrahedronlist[tetIdx * 4 + 0] = tetVertices[0];
                out.tetrahedronlist[tetIdx * 4 + 1] = tetVertices[1];
                out.tetrahedronlist[tetIdx * 4 + 2] = tetVertices[2];
                out.tetrahedronlist[tetIdx * 4 + 3] = tetVertices[3];
                tetIdx++;
            }
        }
    }
    
    // Update the actual number of tetrahedra (might be less due to infinite vertices)
    out.numberoftetrahedra = tetIdx;
    
    // Extract faces information
    // We need to build the face list and tet2face mapping
    std::map<std::tuple<int,int,int>, int> faceMap;  // face vertices -> face index
    std::vector<std::array<int,3>> faceList;        // list of face vertices
    std::vector<std::array<int,4>> tet2faceList;    // tet -> faces mapping
    
    // Process each tetrahedron to extract faces
    tetIdx = 0;
    for (uint32_t i = 0; i < tetMesh->numTets(); i++) {
        if (!tetMesh->isGhost(i)) {
            const uint32_t* tetVertices = tetMesh->tet_node.data() + (i * 4);
            
            if (tetVertices[3] != INFINITE_VERTEX) {
                std::array<int,4> tetFaces;
                
                // Each tet has 4 faces: (0,1,2), (0,3,1), (0,2,3), (1,3,2)
                int faceOrders[4][3] = {{0,1,2}, {0,3,1}, {0,2,3}, {1,3,2}};
                
                for (int f = 0; f < 4; f++) {
                    int v0 = tetVertices[faceOrders[f][0]];
                    int v1 = tetVertices[faceOrders[f][1]];
                    int v2 = tetVertices[faceOrders[f][2]];
                    
                    // Sort vertices to create canonical face representation
                    std::array<int,3> sortedFace = {v0, v1, v2};
                    std::sort(sortedFace.begin(), sortedFace.end());
                    auto faceKey = std::make_tuple(sortedFace[0], sortedFace[1], sortedFace[2]);
                    
                    // Check if face already exists
                    auto it = faceMap.find(faceKey);
                    if (it == faceMap.end()) {
                        // New face
                        int faceIdx = faceList.size();
                        faceMap[faceKey] = faceIdx;
                        faceList.push_back({v0, v1, v2});  // Keep original orientation
                        tetFaces[f] = faceIdx;
                    } else {
                        // Existing face
                        tetFaces[f] = it->second;
                    }
                }
                
                tet2faceList.push_back(tetFaces);
                tetIdx++;
            }
        }
    }
    
    // Set face information in tetgenio
    out.numberoftrifaces = faceList.size();
    out.trifacelist = new int[out.numberoftrifaces * 3];
    
    for (size_t i = 0; i < faceList.size(); i++) {
        out.trifacelist[i * 3 + 0] = faceList[i][0];
        out.trifacelist[i * 3 + 1] = faceList[i][1];
        out.trifacelist[i * 3 + 2] = faceList[i][2];
    }
    
    // Set tet2face mapping
    out.tet2facelist = new int[out.numberoftetrahedra * 4];
    for (size_t i = 0; i < tet2faceList.size(); i++) {
        out.tet2facelist[i * 4 + 0] = tet2faceList[i][0];
        out.tet2facelist[i * 4 + 1] = tet2faceList[i][1];
        out.tet2facelist[i * 4 + 2] = tet2faceList[i][2];
        out.tet2facelist[i * 4 + 3] = tet2faceList[i][3];
    }
    
    // Set edge count (approximate, since we don't extract actual edges)
    out.numberofedges = 0;  // You might need to compute this if required
    
    if (VERBOSE) {
        std::cout << "tetgenio conversion completed:" << std::endl;
        std::cout << "  Vertices: " << out.numberofpoints << std::endl;
        std::cout << "  Tetrahedra: " << out.numberoftetrahedra << std::endl;
        std::cout << "  Faces: " << out.numberoftrifaces << std::endl;
    }
}

// Simple conversion function for visualization only

void SignedHeatTetSolver::convertTetMeshForVisualization(TetMesh* tetMesh) {
    if (!tetMesh) {
        std::cerr << "Error: TetMesh is null" << std::endl;
        return;
    }
    
    // Get vertex count
    uint32_t numVerts = tetMesh->numVertices();
    
    // Extract vertices
    vertices.resize(numVerts, 3);
    for (uint32_t i = 0; i < numVerts; i++) {
        double coords[3];
        tetMesh->vertices[i]->getApproxXYZCoordinates(coords[0], coords[1], coords[2], true);
        
        vertices(i, 0) = coords[0];
        vertices(i, 1) = coords[1];
        vertices(i, 2) = coords[2];
    }
    
    // Count non-ghost tetrahedra
    std::vector<std::array<int, 4>> tetList;
    for (uint32_t i = 0; i < tetMesh->numTets(); i++) {
        if (!tetMesh->isGhost(i)) {
            const uint32_t* tetVertices = tetMesh->tet_node.data() + (i * 4);
            
            // Skip infinite vertices
            if (tetVertices[3] != INFINITE_VERTEX) {
                tetList.push_back({
                    (int)tetVertices[0],
                    (int)tetVertices[1],
                    (int)tetVertices[2],
                    (int)tetVertices[3]
                });
            }
        }
    }
    
    // Convert to Eigen matrix
    tets.resize(tetList.size(), 4);
    for (size_t i = 0; i < tetList.size(); i++) {
        tets(i, 0) = tetList[i][0];
        tets(i, 1) = tetList[i][1];
        tets(i, 2) = tetList[i][2];
        tets(i, 3) = tetList[i][3];
    }
    
    if (VERBOSE) {
        std::cout << "Prepared for visualization:" << std::endl;
        std::cout << "  Vertices: " << vertices.rows() << std::endl;
        std::cout << "  Tetrahedra: " << tets.rows() << std::endl;
    }
}

// 专门用于 CDT 后处理的 TetGen 函数
void SignedHeatTetSolver::feedTetMeshToTetGenOptimized(const TetMesh& tetMesh,
                            const EdgeDualNormalGeometry& edgeGeom,
                            const SignedHeat3DOptions& options,
                            double areaConstraint) {
    
    bool VERBOSE = true;
    tetgenio tetgenInput, tetgenOutput;
    

    
    // 1. 设置顶点（使用与 convertTetMeshForVisualization 相同的方法）
    uint32_t numVertices = tetMesh.numVertices();
    tetgenInput.numberofpoints = numVertices;
    tetgenInput.pointlist = new REAL[numVertices * 3];
    tetgenInput.pointmarkerlist = new int[numVertices];
    
    for (uint32_t i = 0; i < numVertices; i++) {
        double coords[3];
        tetMesh.vertices[i]->getApproxXYZCoordinates(coords[0], coords[1], coords[2], true);
        
        tetgenInput.pointlist[i * 3 + 0] = coords[0];
        tetgenInput.pointlist[i * 3 + 1] = coords[1];
        tetgenInput.pointlist[i * 3 + 2] = coords[2];
        
        tetgenInput.pointmarkerlist[i] = tetMesh.isOnBoundary(i) ? 1 : 0;
    }
    
    // 2. 设置四面体（使用与 convertTetMeshForVisualization 相同的逻辑）
    std::vector<std::array<uint32_t, 4>> validTetList;
    
    for (uint32_t i = 0; i < tetMesh.numTets(); i++) {
        if (!tetMesh.isGhost(i)) {
            const uint32_t* tetVertices = tetMesh.tet_node.data() + (i * 4);
            
            // Skip infinite vertices
            if (tetVertices[3] != INFINITE_VERTEX) {
                validTetList.push_back({
                    tetVertices[0],
                    tetVertices[1],
                    tetVertices[2],
                    tetVertices[3]
                });
            }
        }
    }
    
    tetgenInput.numberoftetrahedra = validTetList.size();
    tetgenInput.tetrahedronlist = new int[validTetList.size() * 4];
    tetgenInput.tetrahedronattributelist = new REAL[validTetList.size()];
    tetgenInput.numberoftetrahedronattributes = 1;
    
    for (size_t i = 0; i < validTetList.size(); i++) {
        for (int j = 0; j < 4; j++) {
            tetgenInput.tetrahedronlist[i * 4 + j] = validTetList[i][j];
        }
        
        tetgenInput.tetrahedronattributelist[i] = 1.0;
    }
    
    // 3. 设置边界面（使用相同的四面体遍历逻辑）
    std::vector<std::array<uint32_t, 3>> boundaryFaces;
    
    for (uint32_t t = 0; t < tetMesh.numTets(); t++) {
        if (tetMesh.isGhost(t)) continue;
        
        const uint32_t* nodes = tetMesh.tet_node.data() + (t * 4);
        const uint64_t* neighs = tetMesh.tet_neigh.data() + (t * 4);
        
        // Skip if has infinite vertex
        if (nodes[3] == INFINITE_VERTEX) continue;
        
        for (int face = 0; face < 4; face++) {
            uint64_t neighTet = neighs[face] >> 2;
            
            if (tetMesh.isGhost(neighTet)) {
                std::array<uint32_t, 3> faceNodes;
                
                switch (face) {
                    case 0: faceNodes = {nodes[1], nodes[2], nodes[3]}; break;
                    case 1: faceNodes = {nodes[0], nodes[3], nodes[2]}; break;
                    case 2: faceNodes = {nodes[0], nodes[1], nodes[3]}; break;
                    case 3: faceNodes = {nodes[0], nodes[2], nodes[1]}; break;
                }
                
                boundaryFaces.push_back(faceNodes);
            }
        }
    }
    
    if (!boundaryFaces.empty()) {
        tetgenInput.numberoffacets = boundaryFaces.size();
        tetgenInput.facetlist = new tetgenio::facet[boundaryFaces.size()];
        tetgenInput.facetmarkerlist = new int[boundaryFaces.size()];
        
        for (size_t i = 0; i < boundaryFaces.size(); i++) {
            tetgenio::facet* f = &tetgenInput.facetlist[i];
            f->numberofpolygons = 1;
            f->polygonlist = new tetgenio::polygon[1];
            f->numberofholes = 0;
            f->holelist = nullptr;
            
            tetgenio::polygon* p = &f->polygonlist[0];
            p->numberofvertices = 3;
            p->vertexlist = new int[3];
            
            p->vertexlist[0] = boundaryFaces[i][0];
            p->vertexlist[1] = boundaryFaces[i][1];
            p->vertexlist[2] = boundaryFaces[i][2];
            
            tetgenInput.facetmarkerlist[i] = 1;
        }
    }
    
    // Calculate mean edge length for area scaling (same as tetmeshPointCloud)
    double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
    meanEdgeLength = 0.05f;
    double meanArea = meanEdgeLength; // Use edge length as proxy for area
    double areaScale = std::pow(2, -options.hCoef);
    
    // Build TetGen flags using the same logic as tetmeshPointCloud
//    std::string TETFLAGS = "rfennz";
//    std::string TETFLAGS = "rfennzq1.414";
    std::string TETFLAGS = "rYzfenna"; // 不加体积约束，只做质量改进

//    std::string TETFLAGS = "rq1.414a" + std::to_string(areaScale * meanArea) + "zfennaY";

//    std::string TETFLAGS = "rYq1.414a" + std::to_string(areaScale * meanArea) + "zfenna";

    if (VERBOSE) {
        std::cout << "Using TetGen flags: " << TETFLAGS << std::endl;
        std::cout << "Mean edge length: " << meanEdgeLength << std::endl;
        std::cout << "Area scale: " << areaScale << std::endl;
        std::cout << "Final area constraint: " << (areaScale * meanArea) << std::endl;
    }

    // 4. 调用 TetGen
    try {
        if (VERBOSE) std::cout << "Calling TetGen with constraint flags: " << TETFLAGS << std::endl;
        tetrahedralize(const_cast<char*>(TETFLAGS.c_str()), &tetgenInput, &tetgenOutput);

        if (VERBOSE) {
            std::cout << "TetGen post-processing completed!" << std::endl;
            std::cout << "Input: " << tetgenInput.numberofpoints << " vertices, "
                      << tetgenInput.numberoftetrahedra << " tetrahedra" << std::endl;
            std::cout << "Output: " << tetgenOutput.numberofpoints << " vertices, "
                      << tetgenOutput.numberoftetrahedra << " tetrahedra" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "TetGen post-processing error: " << e.what() << std::endl;
    }

    
    // 5. 转换输出为 PointPositionGeometry
    if (tetgenOutput.numberofpoints == 0) {
        std::cerr << "TetGen produced no output points" << std::endl;
    }

    getTetmeshData(tetgenOutput);
    
    
    // 6. 设置可视化数据 (vertices 和 tets 作为类成员变量)
    if (tetgenOutput.numberofpoints > 0 && tetgenOutput.numberoftetrahedra > 0) {
        // 设置 vertices
        vertices.resize(tetgenOutput.numberofpoints, 3);
        for (int i = 0; i < tetgenOutput.numberofpoints; i++) {
            vertices(i, 0) = tetgenOutput.pointlist[i * 3 + 0];
            vertices(i, 1) = tetgenOutput.pointlist[i * 3 + 1];
            vertices(i, 2) = tetgenOutput.pointlist[i * 3 + 2];
        }
        
        // 设置 tets
        tets.resize(tetgenOutput.numberoftetrahedra, 4);
        for (int i = 0; i < tetgenOutput.numberoftetrahedra; i++) {
            tets(i, 0) = tetgenOutput.tetrahedronlist[i * 4 + 0];
            tets(i, 1) = tetgenOutput.tetrahedronlist[i * 4 + 1];
            tets(i, 2) = tetgenOutput.tetrahedronlist[i * 4 + 2];
            tets(i, 3) = tetgenOutput.tetrahedronlist[i * 4 + 3];
        }
        
        if (VERBOSE) {
            std::cout << "Updated visualization data:" << std::endl;
            std::cout << "  Vertices: " << vertices.rows() << std::endl;
            std::cout << "  Tetrahedra: " << tets.rows() << std::endl;
        }
    }
    
}

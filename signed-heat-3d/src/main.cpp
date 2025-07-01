#include "geometrycentral/pointcloud/point_position_normal_geometry.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_grid.h"
#include "polyscope/volume_mesh.h"

#include "signed_heat_grid_solver.h"
#include "signed_heat_tet_solver.h"
#include "point_position_dual_normal_geometry.h"

#include "args/args.hxx"
#include "imgui.h"

#include <chrono>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
std::chrono::time_point<high_resolution_clock> t1, t2;
std::chrono::duration<double, std::milli> ms_fp;

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;
std::unique_ptr<pointcloud::PointCloud> cloud;
std::unique_ptr<pointcloud::PointPositionNormalGeometry> pointGeom;

// Contouring
float ISOVAL = 0.;
Vector<double> PHI;
std::unique_ptr<SurfaceMesh> isoMesh;
std::unique_ptr<VertexPositionGeometry> isoGeom;

// Polyscope data
polyscope::SurfaceMesh* psMesh;
polyscope::PointCloud* psCloud;
polyscope::VolumeGridNodeScalarQuantity* gridScalarQ;
polyscope::SlicePlane* psPlane;

// Solvers & parameters
float TCOEF = 1.0;
// Xue modified this, set 1.0 by default.
// Maybe 2 is better, but slow.
float HCOEF = 1.0;
std::unique_ptr<SignedHeatTetSolver> tetSolver;
std::unique_ptr<SignedHeatGridSolver> gridSolver;
SignedHeat3DOptions SHM_OPTIONS;
int CONSTRAINT_MODE = static_cast<int>(LevelSetConstraint::ZeroSet);

// Program variables
enum MeshMode { Tet = 0, Grid };
enum InputMode { Mesh = 0, Points };
int MESH_MODE = MeshMode::Tet;
int INPUT_MODE = InputMode::Mesh;
std::string MESHNAME = "input mesh";
std::string OUTPUT_DIR = "../export";
std::string OUTPUT_FILENAME;
int LAST_SOLVER_MODE;
bool VERBOSE = true;
bool HEADLESS;
bool CONTOURED = false;

void solve() {

    SHM_OPTIONS.levelSetConstraint = static_cast<LevelSetConstraint>(CONSTRAINT_MODE);
    SHM_OPTIONS.tCoef = TCOEF;
    SHM_OPTIONS.hCoef = HCOEF;
    std::string cmapName = "viridis";
    if (MESH_MODE == MeshMode::Tet) {
        if (VERBOSE) std::cerr << "\nSolving on tet mesh..." << std::endl;
        t1 = high_resolution_clock::now();
        PHI = (INPUT_MODE == InputMode::Mesh) ? tetSolver->computeDistance(*geometry, SHM_OPTIONS)
                                              : tetSolver->computeDistance(*pointGeom, SHM_OPTIONS);
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;
        if (!HEADLESS) {
            polyscope::getVolumeMesh("domain")
                ->addVertexScalarQuantity("GSD", PHI)
                ->setColorMap(cmapName)
                ->setIsolinesEnabled(true)
                ->setEnabled(true);
            polyscope::getVolumeMesh("domain")->setCullWholeElements(true);
        }
    } else if (MESH_MODE == MeshMode::Grid) {
        t1 = high_resolution_clock::now();
//        PHI = (INPUT_MODE == InputMode::Mesh) ? gridSolver->computeDistance(*geometry, SHM_OPTIONS)
//                                              : gridSolver->computeDistance(*pointGeom, SHM_OPTIONS);
        
        if (INPUT_MODE == InputMode::Mesh) {
                PHI = gridSolver->computeDistance(*geometry, SHM_OPTIONS);
            } else {
                // Check if pointGeom is actually a PointPositionDualNormalGeometry
                auto* dualNormalGeom = dynamic_cast<pointcloud::PointPositionDualNormalGeometry*>(pointGeom.get());
                if (dualNormalGeom != nullptr) {
                    std::cout << "Using dual normal version of computeDistance" << std::endl;
                    PHI = gridSolver->computeDistance(*dualNormalGeom, SHM_OPTIONS);
                } else {
                    PHI = gridSolver->computeDistance(*pointGeom, SHM_OPTIONS);
                }
            }
        
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;
        if (!HEADLESS) {
            gridScalarQ = polyscope::getVolumeGrid("domain")
                              ->addNodeScalarQuantity("GSD", PHI)
                              ->setColorMap(cmapName)
                              ->setIsolinesEnabled(true);
            gridScalarQ->setEnabled(true);
        }
    }
    if (VERBOSE) std::cerr << "min: " << PHI.minCoeff() << "\tmax: " << PHI.maxCoeff() << std::endl;
    if (!HEADLESS) {
        polyscope::removeLastSceneSlicePlane();
        psPlane = polyscope::addSceneSlicePlane();
        psPlane->setDrawPlane(false);
        psPlane->setDrawWidget(true);
        if (MESH_MODE == MeshMode::Tet) psPlane->setVolumeMeshToInspect("domain");
        if (INPUT_MODE == InputMode::Mesh) {
            psMesh->setIgnoreSlicePlane(psPlane->name, true);
        } else {
            psCloud->setIgnoreSlicePlane(psPlane->name, true);
        }
    }
    LAST_SOLVER_MODE = MESH_MODE;
    SHM_OPTIONS.rebuild = false;
}

void contour() {
    if (LAST_SOLVER_MODE == MeshMode::Tet) {
        tetSolver->isosurface(isoMesh, isoGeom, PHI, ISOVAL);
        polyscope::registerSurfaceMesh("isosurface", isoGeom->vertexPositions, isoMesh->getFaceVertexList());
    } else {
        gridScalarQ->setIsosurfaceLevel(ISOVAL);
        gridScalarQ->setIsosurfaceVizEnabled(true);
        gridScalarQ->setSlicePlanesAffectIsosurface(false);
        gridScalarQ->registerIsosurfaceAsMesh("isosurface");
    }
    CONTOURED = true;
    polyscope::getSurfaceMesh("isosurface")->setIgnoreSlicePlane(psPlane->name, true);
}

void callback() {

    if (ImGui::Button("Solve")) {
        solve();
    }
    ImGui::RadioButton("on tet mesh", &MESH_MODE, MeshMode::Tet);
    ImGui::RadioButton("on grid", &MESH_MODE, MeshMode::Grid);

    ImGui::Separator();
    ImGui::Text("Solve options");
    ImGui::Separator();
    if (MESH_MODE == MeshMode::Tet && mesh != nullptr && mesh->isTriangular()) {
        ImGui::Checkbox("Use Crouzeix-Raviart", &SHM_OPTIONS.useCrouzeixRaviart);
    }
    ImGui::InputFloat("tCoef (diffusion time)", &TCOEF);
    if (ImGui::InputFloat("hCoef (mesh spacing)", &HCOEF)) {
        SHM_OPTIONS.rebuild = true;
    }
    if (MESH_MODE != MeshMode::Grid) {
        ImGui::RadioButton("Constrain zero set", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::ZeroSet));
        ImGui::RadioButton("Constrain multiple levelsets", &CONSTRAINT_MODE,
                           static_cast<int>(LevelSetConstraint::Multiple));
        ImGui::RadioButton("No levelset constraints", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::None));
    }

    if (PHI.size() > 0) {
        ImGui::Separator();
        ImGui::Text("Contour options");
        ImGui::Separator();
        if (ImGui::SliderFloat("Contour (drag slider)", &ISOVAL, PHI.minCoeff(), PHI.maxCoeff())) {
            contour();
        }
        if (ImGui::InputFloat("Contour (enter value)", &ISOVAL)) {
            contour();
        }
        if (CONTOURED) {
            if (ImGui::Button("Export isosurface")) {
                if (LAST_SOLVER_MODE == MeshMode::Grid) {
                    // register geometry-central mesh from Polyscope one
                    polyscope::SurfaceMesh* psIsoMesh = polyscope::getSurfaceMesh("isosurface");

                    std::vector<std::vector<size_t>> polygons;
                    std::vector<Vector3> positions;
                    for (size_t i = 0; i < psIsoMesh->nFacesTriangulation(); i++) {
                        std::vector<size_t> face(3);
                        for (int j = 0; j < 3; j++) {
                            face[j] = psIsoMesh->triangleVertexInds.getValue(3 * i + j);
                        }
                        polygons.push_back(face);
                    }
                    for (size_t i = 0; i < psIsoMesh->nVertices(); i++) {
                        Vector3 p;
                        for (int j = 0; j < 3; j++) p[j] = psIsoMesh->vertexPositions.getValue(i)[j];
                        positions.push_back(p);
                    }
                    std::tie(isoMesh, isoGeom) = makeSurfaceMeshAndGeometry(polygons, positions);
                }
                std::string isoFilename = OUTPUT_DIR + "/isosurface.obj";
                writeSurfaceMesh(*isoMesh, *isoGeom, isoFilename);
                std::cerr << "Isosurface written to " << isoFilename << std::endl;
            }
        }
    }
}

std::tuple<std::vector<Vector3>, std::vector<Vector3>> readPointCloud(const std::string& filepath) {

    std::ifstream curr_file(filepath.c_str());
    std::string line;
    std::string X;
    double x, y, z;
    std::vector<Vector3> positions, normals;
    if (curr_file.is_open()) {
        while (!curr_file.eof()) {
            getline(curr_file, line);
            // Ignore any newlines
            if (line == "") {
                continue;
            }
            std::istringstream iss(line);
            iss >> X;
            if (X == "v") {
                iss >> x >> y >> z;
                positions.push_back({x, y, z});
            } else if (X == "vn") {
                iss >> x >> y >> z;
                normals.push_back({x, y, z});
            }
        }
        curr_file.close();
    } else {
        std::cerr << "Could not open file <" << filepath << ">." << std::endl;
    }
    return std::make_tuple(positions, normals);
}


// Reads a point cloud file where each vertex has either:
// - All vertices have exactly one normal (v followed by vn), OR
// - All vertices have exactly two normals (v followed by vn, vn)
// The function detects the format automatically based on the first few lines.
// Returns a tuple: (positions, normals0, normals1)
// normals1 will be empty if only one normal per vertex is provided
std::tuple<std::vector<Vector3>, std::vector<Vector3>, std::vector<Vector3>>
readPointNormals(const std::string& filepath) {
    std::ifstream curr_file(filepath.c_str());
    std::string line;
    std::string X;
    double x, y, z;
    std::vector<Vector3> positions, normals0, normals1;
    if (curr_file.is_open()) {
        bool hasTwoNormals = false;
        bool formatDetermined = false;
        std::streampos lastPos;
        
        while (!curr_file.eof()) {
            getline(curr_file, line);
            // Ignore any newlines
            if (line == "") {
                continue;
            }
            std::istringstream iss(line);
            iss >> X;
            if (X == "v") {
                iss >> x >> y >> z;
                positions.push_back({x, y, z});
                
                // Read the following normal
                if (getline(curr_file, line)) {
                    std::istringstream issNormal(line);
                    issNormal >> X;
                    if (X == "vn") {
                        issNormal >> x >> y >> z;
                        normals0.push_back({x, y, z});
                        
                        // Check for second normal if format not yet determined
                        if (!formatDetermined) {
                            lastPos = curr_file.tellg();
                            if (getline(curr_file, line)) {
                                std::istringstream issCheck(line);
                                issCheck >> X;
                                if (X == "vn") {
                                    hasTwoNormals = true;
                                    issCheck >> x >> y >> z;
                                    normals1.push_back({x, y, z});
                                } else {
                                    // Only one normal, put the stream position back
                                    curr_file.seekg(lastPos);
                                }
                                formatDetermined = true;
                            }
                        }
                        // If format is determined and has two normals, read the second normal
                        else if (hasTwoNormals) {
                            if (getline(curr_file, line)) {
                                std::istringstream issSecond(line);
                                issSecond >> X;
                                if (X == "vn") {
                                    issSecond >> x >> y >> z;
                                    normals1.push_back({x, y, z});
                                }
                            }
                        }
                    }
                }
            }
        }
        curr_file.close();
    } else {
        std::cerr << "Could not open file <" << filepath << ">." << std::endl;
    }
    
    // If we didn't find any second normals, ensure normals1 is empty
    if (normals1.empty()) {
        normals1.clear();
    }
    
    return std::make_tuple(positions, normals0, normals1);
}



int main(int argc, char** argv) {
    
    // Configure the argument parser
    args::ArgumentParser parser("Solve for generalized signed distance (3D domains).");
    args::HelpFlag help(parser, "help", "Display this help menu", {"help"});
    args::Positional<std::string> meshFilename(parser, "mesh", "A mesh or point cloud file.");
    args::ValueFlag<double> hCoef(parser, "h", "Controls the tet/grid spacing proportional to $2^{-h}$.",
                                  {"h", "hCoef"});
    
    args::Group group(parser);
    args::Flag grid(group, "grid", "Solve on a background grid (vs. tet mesh).", {"g", "grid"});
    args::Flag fast(group, "fast", "Solve using a less accurate, but significantly faster, method of integration.",
                    {"f", "fast"});
    args::Flag verbose(group, "verbose", "Verbose output", {"V", "verbose"});
    args::Flag headless(group, "headless", "Don't use the GUI.", {"l", "headless"});
    
    // Parse args
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help&) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    
    if (!meshFilename) {
        std::cerr << "Please specify a mesh file as argument." << std::endl;
        return EXIT_FAILURE;
    }
    if (hCoef) {
        HCOEF = args::get(hCoef);
    }
    
    // Load mesh
    std::string meshFilepath = args::get(meshFilename);
    MESH_MODE = grid ? MeshMode::Grid : MeshMode::Tet;
    OUTPUT_FILENAME = OUTPUT_DIR + "/GSD.obj";
    HEADLESS = headless;
    SHM_OPTIONS.exportData = headless; // always true if in headless mode
    SHM_OPTIONS.meshname = polyscope::guessNiceNameFromPath(meshFilepath);
    VERBOSE = verbose;
    
    // Get file extension.
    std::string ext = meshFilepath.substr(meshFilepath.find_last_of(".") + 1);
    pointcloud::PointData<Vector3> pointPositions;
    pointcloud::PointData<Vector3> pointNormals0;
    pointcloud::PointData<Vector3> pointNormals1;
    if (ext != "pc" && ext != "normal") {
        std::tie(mesh, geometry) = readSurfaceMesh(meshFilepath);
        INPUT_MODE = InputMode::Mesh;
    } else if (ext == "normal"){
        std::cout << "reading edge normal file " << std::endl;
        
        
        
    }else {
        // read Dual Normal
        std::vector<Vector3> positions, normals0, normals1;
        std::tie(positions, normals0, normals1) = readPointNormals(meshFilepath);
        size_t nPts = positions.size();

        cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
        pointPositions = pointcloud::PointData<Vector3>(*cloud);
        pointNormals0 = pointcloud::PointData<Vector3>(*cloud);
        pointNormals1 = pointcloud::PointData<Vector3>(*cloud); // Add this line for second normals

        // Check if we have two normals per point
        bool hasTwoNormals = !normals1.empty();

        for (size_t i = 0; i < nPts; i++) {
            pointPositions[i] = positions[i];
            pointNormals0[i] = normals0[i];

            // Only set secondary normals if they exist
            if (hasTwoNormals) {
                pointNormals1[i] = normals1[i];
            }
        }
        
        
        std::cout << "hasTwoNormals " << hasTwoNormals << std::endl;

        // Use the appropriate geometry class based on whether we have one or two normals
        if (hasTwoNormals) {
            std::cout << "trying to use 2 normals" << std::endl;
            pointGeom = std::unique_ptr<pointcloud::PointPositionDualNormalGeometry>(
                new pointcloud::PointPositionDualNormalGeometry(*cloud, pointPositions, pointNormals0, pointNormals1));
        } else {
            pointGeom = std::unique_ptr<pointcloud::PointPositionNormalGeometry>(
                new pointcloud::PointPositionNormalGeometry(*cloud, pointPositions, pointNormals0));
        }

        INPUT_MODE = InputMode::Points;
        
        // by default use Grid Mode
//        LAST_SOLVER_MODE = MeshMode::Grid;
        MESH_MODE = MeshMode::Grid;
        

    }
    tetSolver = std::unique_ptr<SignedHeatTetSolver>(new SignedHeatTetSolver());
    gridSolver = std::unique_ptr<SignedHeatGridSolver>(new SignedHeatGridSolver());
    tetSolver->VERBOSE = verbose;
    gridSolver->VERBOSE = verbose;

    if (!HEADLESS) {
        polyscope::init();
        polyscope::state::userCallback = callback;
        if (ext != "pc") {
            psMesh = polyscope::registerSurfaceMesh(MESHNAME, geometry->vertexPositions, mesh->getFaceVertexList());
            if (mesh->isTriangular()) psMesh->setAllPermutations(polyscopePermutations(*mesh));
        } else {
            psCloud = polyscope::registerPointCloud("point cloud", pointPositions);
            psCloud->addVectorQuantity("normals", pointNormals0)
                   ->setVectorLengthScale(0.05)
                   ->setEnabled(true);
            
            // Add code to display the second set of normals if they exist
            // We need to check if we're using the dual normal geometry class
            auto* dualNormalGeom = dynamic_cast<pointcloud::PointPositionDualNormalGeometry*>(pointGeom.get());
            
            if (dualNormalGeom != nullptr) {
                // We have dual normals, so display the second set
                psCloud->addVectorQuantity("normals2", dualNormalGeom->secondNormals)
                       ->setVectorLengthScale(0.05)  // Same length as first normals
                       ->setEnabled(true);
            }
        }
        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
        polyscope::show();
    } else {
        solve();
    }

    return EXIT_SUCCESS;
}

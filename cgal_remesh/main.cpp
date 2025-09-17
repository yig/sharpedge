#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/IO/OBJ.h>

#include <iostream>
#include <string>
#include <map>
#include <limits>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::edge_descriptor edge_descriptor;



std::string add_suffix_before_extension(const std::string& filename, const std::string& suffix) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return filename + suffix;
    }
    return filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
}


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input.obj> [-t targetEdgeLength]" << std::endl;
        return 0;
    }

    std::string input_file = argv[1];
    
    try {
        Mesh mesh;
        std::cout << "Loading OBJ mesh from: " << input_file << std::endl;

        if (!CGAL::IO::read_OBJ(input_file, mesh)) {
            std::cerr << "Error: cannot read OBJ file " << input_file << std::endl;
            return 1;
        }

        std::cout << "✓ Mesh loaded successfully!" << std::endl;
        std::cout << "  Vertices: " << mesh.number_of_vertices() << std::endl;
        std::cout << "  Edges: " << mesh.number_of_edges() << std::endl;
        std::cout << "  Faces: " << mesh.number_of_faces() << std::endl;

        float targetEdgeLength = 0.015f;

        // 解析可选参数
        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                targetEdgeLength = std::atof(argv[i + 1]);
                i++; // 跳过数值
            }
        }
        
        std::cout << "  TargetEdgeLength: " << targetEdgeLength << std::endl;


        // 统计每条边被多少个面使用
        std::map<std::pair<size_t, size_t>, int> edge_use_count;
        for (auto f : faces(mesh)) {
            std::vector<Mesh::Vertex_index> verts;
            for (auto v : vertices_around_face(mesh.halfedge(f), mesh)) {
                verts.push_back(v);
            }
            for (int i = 0; i < (int)verts.size(); i++) {
                auto v1 = verts[i];
                auto v2 = verts[(i + 1) % verts.size()];
                auto edge = std::minmax(v1.idx(), v2.idx());
                edge_use_count[edge]++;
            }
        }

        // 创建边界约束：只标记使用次数 = 1 的边
        auto edge_is_constrained = mesh.add_property_map<edge_descriptor, bool>("e:constrained", false).first;

        int boundary_edges = 0;
        for (auto e : edges(mesh)) {
            auto v1 = source(e, mesh).idx();
            auto v2 = target(e, mesh).idx();
            auto edge = std::minmax(v1, v2);
            if (edge_use_count[edge] == 1) {
                edge_is_constrained[e] = true;
                boundary_edges++;
            }
        }

        std::cout << "Boundary edges (counted manually): " << boundary_edges << std::endl;

        // 重网格化
        CGAL::Polygon_mesh_processing::isotropic_remeshing(
            faces(mesh),
            targetEdgeLength,
            mesh,
            CGAL::Polygon_mesh_processing::parameters::edge_is_constrained_map(edge_is_constrained)
                .protect_constraints(true)
                .number_of_iterations(5)
        );

        std::cout << "✓ Remeshing completed!" << std::endl;
        
        std::string output_file = add_suffix_before_extension(input_file, "_remesh");

        // 保存结果
        std::cout << "Saving to: " << output_file << std::endl;
        if (!CGAL::IO::write_OBJ(output_file, mesh)) {
            std::cerr << "Error: cannot write file " << output_file << std::endl;
            return 1;
        }

        std::cout << "✅ SUCCESS! Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

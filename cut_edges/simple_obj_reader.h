// simple_obj_reader.h
#pragma once
#include <Eigen/Core>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class SimpleOBJReader {
public:
    static bool readOBJ(const std::string& filename, 
                       Eigen::MatrixXd& vertices, 
                       Eigen::MatrixXi& faces) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }
        
        std::vector<Eigen::Vector3d> v_list;
        std::vector<Eigen::Vector3i> f_list;
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;
            
            if (prefix == "v") {
                // 顶点坐标
                double x, y, z;
                if (iss >> x >> y >> z) {
                    v_list.emplace_back(x, y, z);
                }
            }
            else if (prefix == "f") {
                // 面片（只处理三角形）
                std::string vertex1, vertex2, vertex3;
                if (iss >> vertex1 >> vertex2 >> vertex3) {
                    // 处理 "vertex/texture/normal" 格式
                    int v1 = parseVertexIndex(vertex1);
                    int v2 = parseVertexIndex(vertex2);
                    int v3 = parseVertexIndex(vertex3);
                    
                    if (v1 > 0 && v2 > 0 && v3 > 0) {
                        f_list.emplace_back(v1-1, v2-1, v3-1); // 转换为0-based索引
                    }
                }
            }
        }
        
        file.close();
        
        // 转换为Eigen矩阵
        if (v_list.empty() || f_list.empty()) {
            std::cerr << "No valid vertices or faces found in " << filename << std::endl;
            return false;
        }
        
        vertices.resize(v_list.size(), 3);
        for (size_t i = 0; i < v_list.size(); ++i) {
            vertices.row(i) = v_list[i];
        }
        
        faces.resize(f_list.size(), 3);
        for (size_t i = 0; i < f_list.size(); ++i) {
            faces.row(i) = f_list[i];
        }
        
        std::cout << "Read " << vertices.rows() << " vertices and " 
                  << faces.rows() << " faces from " << filename << std::endl;
        
        return true;
    }
    
    static bool writeOBJ(const std::string& filename,
                        const Eigen::MatrixXd& vertices,
                        const Eigen::MatrixXi& faces) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot create file: " << filename << std::endl;
            return false;
        }
        
        // 写入顶点
        for (int i = 0; i < vertices.rows(); ++i) {
            file << "v " << vertices(i, 0) << " " 
                 << vertices(i, 1) << " " 
                 << vertices(i, 2) << std::endl;
        }
        
        // 写入面片
        for (int i = 0; i < faces.rows(); ++i) {
            file << "f " << (faces(i, 0) + 1) << " " 
                 << (faces(i, 1) + 1) << " " 
                 << (faces(i, 2) + 1) << std::endl;
        }
        
        file.close();
        std::cout << "Wrote mesh to " << filename << std::endl;
        return true;
    }

private:
    static int parseVertexIndex(const std::string& vertexStr) {
        size_t slashPos = vertexStr.find('/');
        if (slashPos != std::string::npos) {
            return std::stoi(vertexStr.substr(0, slashPos));
        } else {
            return std::stoi(vertexStr);
        }
    }
};

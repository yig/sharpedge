#include "cut_mesh_along_edges.h"

// Example usage:
#include <iostream>

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
    cut_mesh_along_edges(V, F, cut_edges, V_out, F_out);
    std::cout << "V_out:\n" << V_out << std::endl;
    std::cout << "F_out:\n" << F_out << std::endl;
    
}



int main( int argc, char* argv[] ) {
    
    test_case();
    return 0;
}

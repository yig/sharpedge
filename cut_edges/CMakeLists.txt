cmake_minimum_required(VERSION 3.16)

# 项目信息
project(CutMeshProject VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译器特定的选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

# 查找Eigen3库
find_package(Eigen3 3.3 REQUIRED)

# 如果find_package找不到Eigen，可以手动设置路径：
# set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
# if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
#     message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_INCLUDE_DIR")
# endif()

# 收集所有源文件
set(SOURCES
    main.cpp
    edge_dual_normal.cpp          # 添加这个源文件
)

# 收集头文件（可选，但有助于IDE识别）
set(HEADERS
    cut_mesh_along_edges.h
    edge_dual_normal.h
    simple_obj_reader.h           # 如果是header-only，只需要包含头文件
)

# 创建可执行文件，包含所有必要的源文件
add_executable(cut_mesh_test ${SOURCES} ${HEADERS})

# 链接Eigen库
target_link_libraries(cut_mesh_test Eigen3::Eigen)

# 如果使用手动设置的Eigen路径：
# target_include_directories(cut_mesh_test PRIVATE ${EIGEN3_INCLUDE_DIR})

# 设置输出目录
set_target_properties(cut_mesh_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 添加编译定义（可选）
target_compile_definitions(cut_mesh_test PRIVATE 
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# 启用测试（可选）
enable_testing()
add_test(NAME cut_mesh_basic_test COMMAND cut_mesh_test test_normal.txt test_mesh.obj)

# 安装规则（可选）
install(TARGETS cut_mesh_test
    RUNTIME DESTINATION bin
)

# 显示一些有用的信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Sources: ${SOURCES}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
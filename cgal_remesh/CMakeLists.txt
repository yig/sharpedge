cmake_minimum_required(VERSION 3.16)

project(cgal_boundary_remesh)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 查找 CGAL
find_package(CGAL REQUIRED)
message(STATUS "Found CGAL version: ${CGAL_VERSION}")

# 查找 Boost (CGAL 的依赖)
find_package(Boost REQUIRED)

# 创建可执行文件
add_executable(cgal_remesh 
    main.cpp
)

# 只链接 CGAL
target_link_libraries(cgal_remesh 
    CGAL::CGAL
)

# 设置包含目录
target_include_directories(cgal_remesh PRIVATE 
    ${CGAL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# 编译器标志
if(MSVC)
    target_compile_options(cgal_remesh PRIVATE /W4)
else()
    target_compile_options(cgal_remesh PRIVATE -Wall -Wextra)
endif()

message(STATUS "Configuration complete.")
message(STATUS "Run 'make' to build cgal_remesh executable.")
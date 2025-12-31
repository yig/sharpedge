cmake_minimum_required(VERSION 3.16)

# Project information
project(CutMeshProject VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

# Find Eigen3 library
find_package(Eigen3 3.3 REQUIRED)

# If find_package cannot locate Eigen3 manually set the path:
# set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
# if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
#     message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set EIGEN3_INCLUDE_DIR")
# endif()

# Collect all source files
set(SOURCES
    main.cpp
    edge_dual_normal.cpp          # Add this source file
)

# Collect header files (optional, but useful for IDEs)
set(HEADERS
    cut_mesh_along_edges.h
    edge_dual_normal.h
    simple_obj_reader.h           # If header-only, just include it
)

# Create the executable and include all necessary source files
add_executable(cut_mesh ${SOURCES} ${HEADERS})

# Link Eigen library
target_link_libraries(cut_mesh Eigen3::Eigen)

# If using a manually set Eigen path:
# target_include_directories(cut_mesh PRIVATE ${EIGEN3_INCLUDE_DIR})

# Set output directory
set_target_properties(cut_mesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add compile definitions (optional)
target_compile_definitions(cut_mesh PRIVATE 
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Installation rule (optional)
install(TARGETS cut_mesh
    RUNTIME DESTINATION .
)

# Display useful build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Sources: ${SOURCES}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
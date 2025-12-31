cmake_minimum_required(VERSION 3.16)

project(cgal_boundary_remesh)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build: Debug, Release, RelWithDebInfo, or MinSizeRel."
        FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find CGAL
find_package(CGAL REQUIRED)
message(STATUS "Found CGAL version: ${CGAL_VERSION}")

# Find Boost (required by CGAL)
find_package(Boost REQUIRED)

# Create executable
add_executable(cgal_remesh 
    main.cpp
)

# Link CGAL only
target_link_libraries(cgal_remesh 
    CGAL::CGAL
)

# Include directories
target_include_directories(cgal_remesh PRIVATE 
    ${CGAL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

install(TARGETS cgal_remesh
    RUNTIME DESTINATION .
)

# Compiler warning options
if(MSVC)
    target_compile_options(cgal_remesh PRIVATE /W4)
else()
    target_compile_options(cgal_remesh PRIVATE -Wall -Wextra)
endif()

message(STATUS "Configuration complete.")
message(STATUS "Run 'make' to build the cgal_remesh executable.")

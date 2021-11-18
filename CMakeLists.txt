# author: Ania Brown [QuEST]
# author: Jacob Wilkins [QuEST]
# author: Richard Meister [pyQuEST]

cmake_minimum_required(VERSION 3.5)
if(SKBUILD)
  message(STATUS "Building project using scikit-build")
endif()
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

project(
    pyQuEST_Project
    LANGUAGES C CXX)

# Variables needed by QuEST.
set(QuEST_DIR "QuEST/QuEST" CACHE STRING
    "Name of the directory containing the QuEST library sources relative to the root CMakeLists.txt")

# Build the QuEST static library.
set(BUILD_SHARED_LIBS FALSE CACHE BOOL "Build shared library")
add_subdirectory(${QuEST_DIR})
# Static linking the modules needs position independent code on some compilers.
set_property(TARGET QuEST PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build and install the pyQuEST extension.
add_subdirectory(pyquest)

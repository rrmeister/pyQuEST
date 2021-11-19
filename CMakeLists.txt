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

# Build the QuEST library.
IF(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(BUILD_SHARED_LIBS TRUE CACHE BOOL "Build shared library")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(BUILD_SHARED_LIBS TRUE CACHE BOOL "Build shared library")
else()
    set(BUILD_SHARED_LIBS FALSE CACHE BOOL "Build shared library")
endif()

add_subdirectory(${QuEST_DIR})
install(TARGETS QuEST LIBRARY DESTINATION pyquest)
# Static linking the modules needs position independent code on some compilers.
set_property(TARGET QuEST PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build and install the pyQuEST extension.
add_subdirectory(pyquest)

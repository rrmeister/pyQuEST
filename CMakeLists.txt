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
set(QuEST_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/QuEST_static_lib.patch" CACHE INTERNAL
    "Location of the file to patch QuEST for static library compilation.")

# Patch QuEST to produce a static library.
find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} apply ${QuEST_PATCH}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${QuEST_DIR}
                RESULT_VARIABLE GIT_PATCH_RESULT)

# Build the QuEST static library.
add_subdirectory(${QuEST_DIR})
# Static linking the modules needs position independent code on some compilers.
set_property(TARGET QuEST PROPERTY POSITION_INDEPENDENT_CODE ON)

# Revert patch to keep QuEST directory in its original state.
execute_process(COMMAND ${GIT_EXECUTABLE} apply -R ${QuEST_PATCH}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${QuEST_DIR})

# Build and install the pyQuEST extension.
add_subdirectory(pyquest)

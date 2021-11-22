cmake_minimum_required(VERSION 3.5)
if(SKBUILD)
    message(STATUS "Building project using scikit-build")
endif()
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

project(
    pyQuEST_Project
    LANGUAGES C CXX)

# We want the QuEST library to be shared between all modules of pyQuEST.
# Because building with static libraries would introduce bugs, we force
# building a shared lib regardless of the cache variable.
set(BUILD_SHARED_LIBS TRUE)

# Build the QuEST library.
add_subdirectory(QuEST/QuEST)
set_property(TARGET QuEST PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
install(TARGETS QuEST DESTINATION pyquest)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Build and install the pyQuEST extension.
add_subdirectory(pyquest)

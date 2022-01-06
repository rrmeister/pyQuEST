cmake_minimum_required(VERSION 3.13)

# By default, embed function signatures in docstrings to make working in IDEs easier.
# This part of the CYTHON_FLAGS can be overridden by the user.
set(CYTHON_FLAGS "-X binding=True -X embedsignature=True" CACHE STRING
    "Arguments to pass straight to Cython")

# Need to hand off the QuEST precision to the Cython compiler as well. This part is
# always appended to the user-specified CYTHON_FLAGS if not present. The PRECISION
# cache variable is set in QuEST.
if(NOT CYTHON_FLAGS MATCHES "QuEST_PREC")
  string(APPEND CYTHON_FLAGS " -E QuEST_PREC=${PRECISION}")
endif()

set(CYTHON_ANNOTATE True CACHE BOOL
    "Create annotated HTML for Cython code to inspect Python interactions")

# Cython needs this defined on MinGW64
if(MINGW AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
    add_compile_definitions(MS_WIN64)
endif()

# scikit-build modules needed for compilation.
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

# Cython targets we need to compile. File extensions are automatically
# added by add_cython_target later.
set(CY_TARGETS
    core
    operators
    gates
    unitaries
    decoherence
    initialisations)

# Error handling is done in a the separate header-only library quest_error.h.
add_library(quest_error.h INTERFACE)
target_include_directories(quest_error.h INTERFACE .)

foreach(cy_target ${CY_TARGETS})
    add_cython_target(${cy_target} CXX PY3)
    add_library(${cy_target} MODULE ${${cy_target}})
    python_extension_module(${cy_target})
    target_include_directories(${cy_target} PUBLIC ${NumPy_INCLUDE_DIRS})
    target_link_libraries(${cy_target} QuEST quest_error.h)

    # Cython <3.0 uses an old numpy API which causes numpy to raise a warning.
    # We can't do anything about the API version, so we just silence that warning
    # by setting NPY_NO_DEPRECATED_API to 0.
    target_compile_definitions(${cy_target} PUBLIC NPY_NO_DEPRECATED_API=0)

    # MinGW64 uses its own libraries which are usually not in the dll search path on Windows.
    # Therefore statically link all relevant libraries directly into the modules.
    if(MINGW AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
        target_link_options(${cy_target} PUBLIC -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive)
    endif()

    # Disable "deprecated" warnings until Cython is updated to not use
    # tp_print anymore.
    if (NOT MSVC)  # MSVC does not know this flag.
        target_compile_options(${cy_target} PRIVATE -Wno-deprecated-declarations)
    endif()
endforeach(cy_target)

install(TARGETS ${CY_TARGETS} LIBRARY DESTINATION pyquest)

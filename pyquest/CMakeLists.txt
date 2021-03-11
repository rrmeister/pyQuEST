# Need to hand off the QuEST precision to the Cython compiler as well.
set(CYTHON_FLAGS "-E QuEST_PREC=${PRECISION}" CACHE INTERNAL
    "Quest precision passed to the pyQuEST compilation")
set(CYTHON_ANNOTATE True CACHE INTERNAL
    "Create annotated HTML for Cython code to inspect Python interactions")

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

# Error handling is done in a the separate header-only
# library error_handler.
add_library(error_handler INTERFACE)
target_include_directories(error_handler INTERFACE .)

foreach(cy_target ${CY_TARGETS})
    add_cython_target(${cy_target} CXX PY3)
    add_library(${cy_target} MODULE ${${cy_target}})
    python_extension_module(${cy_target})
    target_include_directories(${cy_target} PUBLIC ${NumPy_INCLUDE_DIRS})
    target_link_libraries(${cy_target} QuEST error_handler)

    # Disable "deprecated" warnings until Cython is updated to not use
    # tp_print anymore.
    target_compile_options(${cy_target} PRIVATE -Wno-deprecated-declarations)
endforeach(cy_target)

install(TARGETS ${CY_TARGETS} LIBRARY DESTINATION pyquest)

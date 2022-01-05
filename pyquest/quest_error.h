/*
 * quest_error.h - Header-only library to add Python exception handling for
 *     QuEST. Will raise a `pyquest.quest_error.QuESTError` exception in Python
 *     when the QuEST backend throws a `QuEST_exception`, which occurs every
 *     time the overriden function `invalidQuESTInputError()` is called. This
 *     header must be included in the relevant Python modules which call the
 *     QuEST backend.
 */

#ifndef QUEST_ERR
#define QUEST_ERR
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>
#include <string>
#include "Python.h"
#include "quest_exception.h"
#define __Pyx_CppExn2PyErr quest_exception_handler

// We initialise the reference to the Python exception to NULL and
// only load it when there's actually an excpetion.
PyObject* QuESTError_class = NULL;

// Modification of __Pyx_CppExn2PyErr from Cython/CppSupport.cpp extended to
// handling quest_exception separately.
static void quest_exception_handler() {
    // If the Exception is not loaded yet, we must import the
    // pyquest.quest_error module and get a reference to the
    // QuESTError class.
    if (QuESTError_class == NULL) {
        PyObject *error_module = PyImport_ImportModule("pyquest.quest_error");
        QuESTError_class = PyObject_GetAttrString(error_module, "QuESTError");
        Py_XDECREF(error_module);
    }
    // Catch a handful of different errors here and turn them into the
    // equivalent Python errors.
    try {
        if (PyErr_Occurred())
            ; // let the latest Python exn pass through and ignore the current one
        else
            throw;
    } catch (const quest_exception& exn) {
        PyErr_SetString(QuESTError_class, exn.what());
    // The rest of this function is identical to Cython/CppSupport.cpp
    } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
    } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    } catch (const std::bad_typeid& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
    } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
    } catch (const std::ios_base::failure& exn) {
        PyErr_SetString(PyExc_IOError, exn.what());
    } catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
    } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
    } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
    } catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
    }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
    }
}

#endif //QUEST_ERR

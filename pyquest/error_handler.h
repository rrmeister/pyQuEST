/*
 * error_handler.h - Header-only library to add Python exception handling to
 *     QuEST. Will raise a "pyquest.quest_error.QuESTError" exception on any
 *     error inside QuEST.
 */

#ifndef QUEST_ERR_HAND
#define QUEST_ERR_HAND
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>
#include <string>
#include "Python.h"
#define __Pyx_CppExn2PyErr quest_exception_handler

// Our custom exception for any QuEST errors.
class quest_exception : public std::runtime_error {
public:
    static PyObject* QuESTError_class;
    quest_exception(std::string message) : std::runtime_error(message) {
        // If the Exception is not loaded yet, we must import the
        // pyquest.quest_error module and get a reference to the
        // QuESTError class.
        if (QuESTError_class == NULL) {
            PyObject *error_module = PyImport_ImportModule("pyquest.quest_error");
            QuESTError_class = PyObject_GetAttrString(error_module, "QuESTError");
            Py_XDECREF(error_module);
        }
    }
};

// We initialize the reference to the Python exception to NULL and
// only load it when there's actually an excpetion.
PyObject* quest_exception::QuESTError_class = NULL;

// Overwrite weak symbol of the generic QuEST error handler (which just
// exits the executable) with one that throws an exception.
extern "C" void invalidQuESTInputError(const char* errMsg, const char* errFunc) {
    throw quest_exception("Error in QuEST function "
                          + std::string(errFunc) + ": " + std::string(errMsg));
}

// Modification of __Pyx_CppExn2PyErr from Cython/CppSupport.cpp extended to
// handling quest_exception separately.
static void quest_exception_handler() {
    // Catch a handful of different errors here and turn them into the
    // equivalent Python errors.
    try {
        if (PyErr_Occurred())
            ; // let the latest Python exn pass through and ignore the current one
        else
            throw;
    } catch (const quest_exception& exn) {
        PyErr_SetString(exn.QuESTError_class, exn.what());
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

#endif //QUEST_ERR_HAND

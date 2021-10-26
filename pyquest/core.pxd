from libcpp.deque cimport deque
from libcpp cimport bool as bool_t
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from cpython.ref cimport PyObject, Py_XDECREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport qreal, qcomp, Complex
from pyquest.quest_interface cimport OP_TYPES, Qureg, QuESTEnv
from pyquest.operators cimport BaseOperator, GlobalOperator
from pyquest.gates cimport M
cimport numpy as np


cdef class QuESTEnvironment:
    cdef QuESTEnv c_env
    cdef object _env_capsule
    cdef bool_t _cuda
    cdef bool_t _openmp
    cdef bool_t _mpi
    cdef int _num_threads
    cdef int _num_ranks
    cdef log_register(self, Register reg)
    cdef object _logged_registers
    cdef _close(self)


cdef class Register:
    cdef object __weakref__  # Makes Register weak-refable in Cython
    cdef Qureg c_register
    cdef object _borrowed_from
    cdef object _borrowers
    cdef Complex _scaling_factor
    cpdef init_blank_state(self)
    cpdef apply_circuit(self, Circuit circ)
    cpdef apply_operator(self, BaseOperator op)
    cpdef qcomp inner_product(self, Register other)
    cpdef Register copy(self)
    cpdef void copy_to(self, Register other)
    cpdef void copy_from(self, Register other)
    cdef _destroy(self)
    cdef qcomp _get_amp(self, size_t row, size_t col)
    cdef qcomp[:, :] _get_state_from_slices(self, slice row_slice, slice col_slice)
    cdef qcomp[:, :] _get_state_from_col_slice(self, row_index, slice col_slice)
    cdef qcomp[:, :] _get_state_from_row_slice(self, slice row_slice, col_index)
    cdef qcomp[:, :] _get_state_from_indexables(self, row_index, col_index)
    cdef _fix_index(self, index)
    cdef Register _create_with_borrowed_reference(Register original_reg)
    cdef void _register_borrower(self, new_borrower)
    cdef void _unregister_borrower(self, borrower)
    cdef void _set_borrowee(self, borrowee)
    cdef void _ensure_no_borrow(self)
    cdef void _apply_delayed_operations(self)
    cdef void _apply_scaling(self)


cdef class Circuit(GlobalOperator):
    cdef deque[PyObject*] c_operations
    cdef py_operations

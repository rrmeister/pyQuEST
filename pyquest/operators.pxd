from libcpp cimport bool
from libc.stdlib cimport malloc, calloc, free
from cpython.pycapsule cimport PyCapsule_GetPointer
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport qreal, qcomp, OP_TYPES, Qureg, pauliOpType
from pyquest.quest_interface cimport phaseFunc, bitEncoding
from pyquest.quest_interface cimport QuESTEnv, Complex
from pyquest.quest_interface cimport ComplexMatrix2, ComplexMatrix4, ComplexMatrixN
from pyquest.quest_interface cimport createComplexMatrixN, destroyComplexMatrixN
cimport numpy as np


cdef class BaseOperator:
    cdef int TYPE
    cpdef as_matrix(self, num_qubits=*)
    cdef int apply_to(self, Qureg c_register) except -1


cdef class GlobalOperator(BaseOperator):
    pass


cdef class ControlledOperator(BaseOperator):
    cdef int _num_controls
    cdef int *_controls


cdef class SingleQubitOperator(ControlledOperator):
    cdef int _target


cdef class MultiQubitOperator(ControlledOperator):
    cdef int _num_targets
    cdef int *_targets


cdef class MatrixOperator(MultiQubitOperator):
    cdef void *_matrix
    cdef qreal **_real
    cdef qreal **_imag
    cdef _create_array_property(self)
    cdef _numpy_array_to_matrix_attribute(self, np.ndarray arr)
    cdef _copy_generic_array(self, np.ndarray arr)
    cdef _copy_single_array(self, float[:, :] arr)
    cdef _copy_double_array(self, double[:, :] arr)
    cdef _copy_longdouble_array(self, long double[:, :] arr)
    cdef _copy_csingle_array(self, float complex[:, :] arr)
    cdef _copy_cdouble_array(self, double complex[:, :] arr)
    cdef _copy_clongdouble_array(self, long double complex[:, :] arr)


cdef class DiagonalOperator(GlobalOperator):
    cdef quest.DiagonalOp _diag_op


cdef class PauliProduct(GlobalOperator):
    cdef int _num_qubits
    cdef list _pauli_terms
    cdef qreal _coefficient
    cdef int *_pauli_codes


cdef class PauliSum(GlobalOperator):
    cdef int _min_qubits
    cdef int _num_qubits
    cdef int _num_terms
    cdef int *_all_pauli_codes
    cdef qreal *_coefficients
    cdef list _pauli_terms


cdef class TrotterCircuit(GlobalOperator):
    pass


cdef class PhaseFunc(GlobalOperator):
    cdef phaseFunc _phase_func_type
    cdef bool _is_poly
    cdef bitEncoding _bit_encoding
    cdef int _num_overrides
    cdef long long int *_override_inds
    cdef qreal *_override_phases
    cdef int _num_parameters
    cdef qreal *_parameters
    cdef int _num_regs
    cdef int *_num_qubits_per_reg
    cdef int *_qubits_in_regs
    cdef int *_num_terms_per_reg
    cdef qreal *_coeffs
    cdef qreal *_exponents

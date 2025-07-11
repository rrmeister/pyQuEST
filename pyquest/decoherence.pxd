from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
cimport numpy as np
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport qreal, qcomp, Qureg
from pyquest.quest_interface cimport ComplexMatrix2, ComplexMatrix4, ComplexMatrixN
from pyquest.quest_interface cimport createComplexMatrixN, destroyComplexMatrixN
from pyquest.operators cimport SingleQubitOperator, MultiQubitOperator, GlobalOperator
from pyquest.core cimport OP_TYPES, Register


cdef class Damping(SingleQubitOperator):
    cdef qreal _prob


cdef class Dephasing(MultiQubitOperator):
    cdef qreal _prob


cdef class Depolarising(MultiQubitOperator):
    cdef qreal _prob


cdef class KrausMap(MultiQubitOperator):
    cdef void* _ops
    cdef int _num_ops


cdef class PauliNoise(SingleQubitOperator):
    cdef qreal _prob_x
    cdef qreal _prob_y
    cdef qreal _prob_z


cdef class MixDensityMatrix(GlobalOperator):
    cdef qreal _prob
    cdef Register _other_register

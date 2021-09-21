from libc.stdlib cimport malloc, free
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport OP_TYPES, qreal, Qureg
from pyquest.operators cimport MultiQubitOperator


cdef class M(MultiQubitOperator):
    cdef int *_results
    cdef qreal *_probabilities
    cdef int *_force

from libc.stdlib cimport malloc, free
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport qcomp, Vector, qreal, OP_TYPES, Qureg
from pyquest.quest_interface cimport CompMatr1, CompMatr2, CompMatr
from pyquest.operators cimport SingleQubitOperator, MultiQubitOperator, MatrixOperator


cdef class U(MatrixOperator):
    cdef int *_control_pattern


cdef class CompactU(SingleQubitOperator):
    cdef qcomp _alpha
    cdef qcomp _beta


cdef class X(MultiQubitOperator):
    pass


cdef class Y(SingleQubitOperator):
    pass


cdef class Z(SingleQubitOperator):
    pass


cdef class Swap(MultiQubitOperator):
    pass


cdef class SqrtSwap(MultiQubitOperator):
    pass


cdef class H(SingleQubitOperator):
    pass


cdef class S(SingleQubitOperator):
    pass


cdef class T(SingleQubitOperator):
    pass


cdef class BaseRotate(SingleQubitOperator):
    cdef qreal _angle


cdef class Rx(BaseRotate):
    pass


cdef class Ry(BaseRotate):
    pass


cdef class Rz(BaseRotate):
    pass


cdef class Phase(BaseRotate):
    pass


cdef class RotateAroundAxis(BaseRotate):
    cdef Vector _axis


cdef class MultiRotatePauli(BaseRotate):
    pass

cdef class ParamSwap(BaseRotate):
    cdef CompMatr2 _u

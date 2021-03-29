from libc.stdlib cimport malloc, calloc, free
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport Complex, Vector, qreal, OP_TYPES, Qureg
from pyquest.quest_interface cimport pauliOpType
from pyquest.quest_interface cimport ComplexMatrix2, ComplexMatrix4, ComplexMatrixN
from pyquest.quest_interface cimport createComplexMatrixN, destroyComplexMatrixN
from pyquest.operators cimport BaseOperator, SingleQubitOperator, GlobalOperator
from pyquest.operators cimport MultiQubitOperator, MatrixOperator


cdef class U(MatrixOperator):
    pass


cdef class CompactU(SingleQubitOperator):
    cdef Complex _alpha
    cdef Complex _beta


cdef class PauliOperator(SingleQubitOperator):
    pass


cdef class X(PauliOperator):
    pass


cdef class Y(PauliOperator):
    pass


cdef class Z(PauliOperator):
    pass


cdef class PauliProduct(GlobalOperator):
    cdef int _num_qubits
    cdef pauliOpType* _pauli_types


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
    cdef ComplexMatrix4 _u

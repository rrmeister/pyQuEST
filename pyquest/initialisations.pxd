from pyquest.operators cimport GlobalOperator
from pyquest.core cimport Register
cimport pyquest.quest_interface as quest
from pyquest.quest_interface cimport OP_TYPES, Qureg


cdef class StateInit(GlobalOperator):
    pass


cdef class BlankState(StateInit):
    pass


cdef class ClassicalState(StateInit):
    cdef long long int _state


cdef class PlusState(StateInit):
    pass


cdef class PureState(StateInit):
    cdef Register _pure_state


# This one is a bit redundant (it's really a ClassicalState with
# state_ind 0, but we have it for convenience)
cdef class ZeroState(StateInit):
    pass

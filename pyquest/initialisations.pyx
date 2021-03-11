cdef class BlankState(StateInit):
    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_INIT_BLANK

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.initBlankState(c_register)


cdef class ClassicalState(StateInit):
    def __cinit__(self, long long int state_ind):
        self.TYPE = OP_TYPES.OP_INIT_CLASSICAL
        self._state = state_ind

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.initClassicalState(c_register, self._state)


cdef class PlusState(StateInit):
    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_INIT_PLUS

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.initPlusState(c_register)


cdef class PureState(StateInit):
    def __cinit__(self, Register pure_state, copy_state=True):
        self.TYPE = OP_TYPES.OP_INIT_PURE
        if copy_state:
            self._pure_state = pure_state.copy()
        else:
            self._pure_state = pure_state

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.initPureState(c_register, self._pure_state.c_register)


cdef class ZeroState(StateInit):
    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_INIT_ZERO

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.initZeroState(c_register)

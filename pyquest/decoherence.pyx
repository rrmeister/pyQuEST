cdef class Damping(SingleQubitOperator):
    def __cinit__(self, target, prob):
        self.TYPE = OP_TYPES.OP_DAMP
        self._prob = prob

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixDamping(c_register, self._target, self._prob)


cdef class Dephasing(MultiQubitOperator):
    def __cinit__(self, targets, prob):
        self.TYPE = OP_TYPES.OP_DEPHASE
        if not 0 < self._num_targets < 3:
            raise ValueError("Dephasing noise must act on 1 or 2 qubits.")
        self._prob = prob

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_targets == 1:
            quest.mixDephasing(c_register, self._targets[0], self._prob)
        # Check for number of targets as extra safeguard.
        elif self._num_targets == 2:
            quest.mixTwoQubitDephasing(
                c_register, self._targets[0], self._targets[1], self._prob)


cdef class Depolarising(MultiQubitOperator):
    def __cinit__(self, target, prob):
        self.TYPE = OP_TYPES.OP_DEPOL
        if not 0 < self._num_targets < 3:
            raise ValueError("Depolarising noise must act on 1 or 2 qubits.")
        self._prob = prob

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_targets == 1:
            quest.mixDepolarising(c_register, self._targets[0], self._prob)
        if self._num_targets == 2:
            quest.mixTwoQubitDepolarising(
                c_register, self._targets[0], self._targets[1], self._prob)


cdef class KrausMap(MultiQubitOperator):
    def __cinit__(self, targets=None, operators=None, target=None):
        self.TYPE = OP_TYPES.OP_KRAUS
        raise NotImplementedError("KrausMap not yet implemented.")


cdef class PauliNoise(SingleQubitOperator):
    def __cinit__(self, target, probs):
        self.TYPE = OP_TYPES.OP_PAULI_NOISE
        self._prob_x = probs[0]
        self._prob_y = probs[1]
        self._prob_z = probs[2]

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixPauli(c_register, self._target,
                       self._prob_x, self._prob_y, self._prob_z)


cdef class MixDensityMatrix(GlobalOperator):
    def __cinit__(self, prob, Register density_matrix, copy_register=True):
        self.TYPE = OP_TYPES.OP_MIX_DENSITY
        self._prob = prob
        if not density_matrix.is_density_matrix:
            raise ValueError("Register 'density_matrix' must be "
                             "a density matrix.")
        if copy_register:
            self._other_register = density_matrix.copy()
        else:
            self._other_register = density_matrix

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixDensityMatrix(c_register, self._prob,
                               self._other_register.c_register)

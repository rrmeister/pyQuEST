"""Classes to represent norm-conserving operations in pyquest.

This module contains classes representing non-unitary but
norm-conserving operations on a quantum state. They generally use short
names, but often have more verbose aliases, which are defined at the end
of the file and listed in the ``Classes`` section after each gate in
square brackets.

Classes:
    M [Measurement]:
        Measurement of one or more qubits.
"""

cdef class M(MultiQubitOperator):

    def __cinit__(self, targets=None, target=None, force=None):
        cdef int k
        self.TYPE = OP_TYPES.OP_MEASURE
        self._force = NULL
        if force is not None:
            self._force = <int*>malloc(self._num_targets * sizeof(self._force[0]))
            if self._num_targets == 1:
                try:
                    self._force[0] = force[0] if force is not None else -1
                except TypeError:
                    self._force[0] = force if force is not None else -1
            else:
                for k in range(self._num_targets):
                    self._force[k] = force[k] if force[k] is not None else -1
        self._results = <int*>malloc(self._num_targets * sizeof(self._results[0]))
        self._probabilities = <qreal*>malloc(self._num_targets * sizeof(self._probabilities[0]))
        for k in range(self._num_targets):
            # -1 in these properties means no measurement yet, or the
            # last measurement failed.
            self._results[k] = -1
            self._probabilities[k] = -1

    def __dealloc__(self):
        free(self._force)
        free(self._results)
        free(self._probabilities)

    def __repr__(self):
        res = type(self).__name__ + "(" + str(self.targets)
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        if self._force != NULL:
            res += ", force=" + str(self.force)
        return res + ")"

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef size_t k
        for k in range(self._num_targets):
            self._results[k] = -1
            self._probabilities[k] = -1
        for k in range(self._num_targets):
            if self._force == NULL or self._force[k] == -1:
                self._results[k] = quest.measureWithStats(
                    c_register, self._targets[k], &(self._probabilities[k]))
            else:
                self._probabilities[k] = quest.collapseToOutcome(
                    c_register, self._targets[k], self._force[k])
                self._results[k] = self._force[k]

    @property
    def results(self):
        cdef size_t k
        return [self._results[k] if self._results[k] != -1 else None
                for k in range(self._num_targets)]

    @property
    def probabilities(self):
        cdef size_t k
        return [self._probabilities[k]
                if self._probabilities[k] != -1
                else None
                for k in range(self._num_targets)]

    @property
    def force(self):
        cdef int k
        if self._force == NULL:
            return None
        return [self._force[k] if self._force[k] != -1 else None
                for k in range(self._num_targets)]

Measurement = M

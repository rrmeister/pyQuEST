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

    def __cinit__(self, targets=None, target=None):
        self.TYPE = OP_TYPES.OP_MEASURE
        self._results = <int*>malloc(self._num_targets * sizeof(self._results[0]))
        self._probabilities = <qreal*>malloc(self._num_targets * sizeof(self._probabilities[0]))

    def __dealloc__(self):
        free(self._results)
        free(self._probabilities)

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef size_t k
        for k in range(self._num_targets):
            self._results[k] = quest.measureWithStats(
                c_register, self._targets[k], &(self._probabilities[k]))

    @property
    def results(self):
        cdef size_t k
        cdef list py_results = [0] * self._num_targets
        for k in range(self._num_targets):
            py_results[k] = self._results[k]
        return py_results

    @property
    def probabilities(self):
        cdef size_t k
        cdef list py_probs = [0] * self._num_targets
        for k in range(self._num_targets):
            py_probs[k] = self._probabilities[k]
        return py_probs

Measurement = M

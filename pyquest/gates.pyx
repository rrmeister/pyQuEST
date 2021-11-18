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

# Because QuEST is statically linked into pyquest, each module executes
# its own copy of the code. Therefore, the pseudorandom seed set in core
# is not communicated to the other modules. Calling seedQuESTDefault()
# here is a temporary fix for this problem, but a dirty one. It will be
# inevitable to load QuEST as a dynamic library (preferred) or establish
# some kind of communication between the modules to have the same seed
# and advance the MT in all modules whenever a random number is
# generated.
cdef void seed_quest(QuESTEnv* env):
    quest.seedQuESTDefault(env)


cdef class M(MultiQubitOperator):
    """Class implementing measurements.

    When applied to a register, an object of this class performs a
    measurement of one or more qubits in the z-basis. Qubits are
    measured in the order they are given in the `targets` argument.

    The results of the last time the gate was applied to a register are
    available in the `results` property, the corresponding probabilities
    are in the `probabilities` property.

    For faster calculations (e.g. when post-selecting on a measurement
    outcome), the (unphysical) additional option of forcing the outcome
    of the measurement by providing it in the `force` arument.
    """

    def __cinit__(self, targets=None, target=None, force=None):
        """Create a new measurement object.

        Args:
            targets(int | iterable[int]): Indices of the qubits to
                measure, in the order they are supposed to be measured.
                Can be a scalar for a single target, or an iterable
                for multiple targets. Remember that measurements are
                destructive and the outcome of an earlier measurement
                can influence the probabilities of later measurements.
            target(int | iterable[int]): Alternative for `targets`;
                only one of `target` or `targets` may be not `None`.
            force(int | iterable[int]): Force the outcome of all or
                some measurements. Can be a scalar or an iterable of
                length 1, if a single qubit is measured. Must be `None`
                or an iterable of the same length as `targets`, if
                multiple qubits are measured. Each entry must be either
                `None` (meaning do not force the outcome of that
                specific measurement), `0`, or `1`. Defaults to `None`.
        """
        cdef int k
        self.TYPE = OP_TYPES.OP_MEASURE
        self._force = NULL
        if force is not None:
            self._force = <int*>malloc(self._num_targets * sizeof(self._force[0]))
            if self._num_targets == 1:
                try:
                    self._force[0] = force if force is not None else -1
                except TypeError:
                    if len(force) != self._num_targets:
                        raise ValueError(
                            "Number of forced outcomes must match the number "
                            "of targets.")
                    self._force[0] = force[0] if force is not None else -1
            else:
                if len(force) != self._num_targets:
                    raise ValueError(
                        "Number of forced outcomes must match the number "
                        "of targets.")
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

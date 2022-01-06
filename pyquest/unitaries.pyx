"""Classes to represent unitary operations in pyquest.

This module contains classes representing unitary operations on a
quantum state. They generally use short names, but often have more
verbose aliases, which are defined at the end of the file and listed
in the ``Classes`` section after each gate in square brackets.

Classes:
    U [Unitary]:
        Generic unitary gate for which a matrix form must be given.
    CompactU [CompactUnitary]:
        Single-qubit unitary in a compact form with only 2 parameters.
    X [NOT, PauliX]:
        Single-qubit Pauli-X gate.
    Y [PauliY]:
        Single-qubit Pauli-Y gate.
    Z [PauliZ]:
        Single-qubit Pauli-Z gate.
    Swap:
        Swap gate on exactly two qubits.
    SqrtSwap:
        Square root of swap gate on exactly two qubits.
    H [Hadamard]:
        Single-qubit Hadamard gate.
    S [SGate]:
        Single-qubit phase shift gate with a phase of pi/2 (S-Gate).
    T [TGate]:
        Single-qubit phase shift gate with a phase of pi/4 (T-Gate).
    BaseRotate:
        Abstract base class for all rotation gates.
    Rx [RotateX]:
        Single-qubit gate rotating about the x-axis of the Bloch sphere.
    Ry [RotateY]:
        Single-qubit gate rotating about the y-axis of the Bloch sphere.
    Rz [RotateZ]:
        Single-qubit gate rotating about the z-axis of the Bloch sphere.
    Phase [PhaseShift]:
        Single-qubit gate shifting the phase of the |1> state.
    RotateAroundAxis:
        Single-qubit gate rotating the qubit about an arbitrary axis.
"""


from pyquest.quest_error import QuESTError


cdef class U(MatrixOperator):

    def __cinit__(self, targets=None, matrix=None, controls=None, target=None,
                  control_pattern=None):
        cdef int k
        self.TYPE = OP_TYPES.OP_UNITARY
        self._control_pattern = NULL
        if control_pattern is not None:
            if self._num_targets > 1:
                raise ValueError(
                    "Control patterns are only available for "
                    "single-qubit targets.")
            self._control_pattern = <int*>malloc(
                self._num_controls * sizeof(self._control_pattern[0]))
            if self._num_controls == 1:
                try:
                    self._control_pattern[0] = control_pattern
                except TypeError:
                    if len(control_pattern) != self._num_controls:
                        raise ValueError(
                            "Length of control pattern must match the number "
                            "of controls.")
                    self._control_pattern[0] = control_pattern[0]
            else:
                if len(control_pattern) != self._num_controls:
                    raise ValueError(
                        "Length of control pattern must match the number "
                        "of controls.")
                for k in range(self._num_controls):
                    self._control_pattern[k] = control_pattern[k]

    def __init__(self, targets=None, matrix=None, controls=None, target=None,
                 control_pattern=None):
        super().__init__(targets, matrix, controls, target)

    def __dealloc__(self):
        # Because _create_array_property has been overridden,
        # deallocation must also be performed separately.
        if self._num_targets > 2:
            if self._matrix != NULL:
                destroyComplexMatrixN((<ComplexMatrixN*>self._matrix)[0])
        else:
            free(self._real)
            free(self._imag)
        # The parent destructor might also call free on
        # self._matrix, self._real, and self._imag. Setting them to
        # NULL avoids freeing the same pointer twice.
        self._real = NULL
        self._imag = NULL
        free(self._matrix)
        self._matrix = NULL
        free(self._control_pattern)

    def __repr__(self):
        if self._control_pattern == NULL:
            return super().__repr__()
        return (super().__repr__()[:-1]
                + ', control_pattern='
                + str(self.control_pattern)
                + ')')

    @property
    def control_pattern(self):
        cdef int k
        if self._control_pattern == NULL:
            return None
        return [self._control_pattern[k] for k in range(self._num_controls)]

    cdef _create_array_property(self):
        # We need to override this method, because core QuEST
        # supports ComplexMatrix* with controls for unitaries, but
        # not for generic matrices.
        cdef size_t matrix_dim = 2 ** self._num_targets
        cdef size_t k
        if self._num_targets == 1:
            self._matrix = malloc(sizeof(ComplexMatrix2))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = &(<ComplexMatrix2*>self._matrix).real[k][0]
                self._imag[k] = &(<ComplexMatrix2*>self._matrix).imag[k][0]
        elif self._num_targets == 2:
            self._matrix = malloc(sizeof(ComplexMatrix4))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = &(<ComplexMatrix4*>self._matrix).real[k][0]
                self._imag[k] = &(<ComplexMatrix4*>self._matrix).imag[k][0]
        else:
            self._matrix = malloc(sizeof(ComplexMatrixN))
            (<ComplexMatrixN*>self._matrix)[0] = createComplexMatrixN(self._num_targets)
            self._real = (<ComplexMatrixN*>self._matrix).real
            self._imag = (<ComplexMatrixN*>self._matrix).imag

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            if self._num_targets == 1:
                quest.unitary(
                    c_register, self._targets[0],
                    (<ComplexMatrix2*>self._matrix)[0])
            elif self._num_targets == 2:
                quest.twoQubitUnitary(
                    c_register, self._targets[0], self._targets[1],
                    (<ComplexMatrix4*>self._matrix)[0])
            else:
                quest.multiQubitUnitary(
                    c_register, self._targets, self._num_targets,
                    (<ComplexMatrixN*>self._matrix)[0])
        elif self._num_controls == 1:
            if self._num_targets == 1:
                if self._control_pattern == NULL:
                    quest.controlledUnitary(
                        c_register, self._controls[0], self._targets[0],
                        (<ComplexMatrix2*>self._matrix)[0])
                else:
                    quest.multiStateControlledUnitary(
                        c_register, self._controls, self._control_pattern,
                        self._num_controls, self._targets[0],
                        (<ComplexMatrix2*>self._matrix)[0])
            elif self._num_targets == 2:
                quest.controlledTwoQubitUnitary(
                    c_register, self._controls[0], self._targets[0],
                    self._targets[1], (<ComplexMatrix4*>self._matrix)[0])
            else:
                quest.controlledMultiQubitUnitary(
                    c_register, self._controls[0], self._targets,
                    self._num_targets, (<ComplexMatrixN*>self._matrix)[0])
        else:
            if self._num_targets == 1:
                if self._control_pattern == NULL:
                    quest.multiControlledUnitary(
                        c_register, self._controls, self._num_controls,
                        self._targets[0], (<ComplexMatrix2*>self._matrix)[0])
                else:
                    quest.multiStateControlledUnitary(
                        c_register, self._controls, self._control_pattern,
                        self._num_controls, self._targets[0],
                        (<ComplexMatrix2*>self._matrix)[0])
            elif self._num_targets == 2:
                quest.multiControlledTwoQubitUnitary(
                    c_register, self._controls, self._num_controls,
                    self._targets[0], self._targets[1],
                    (<ComplexMatrix4*>self._matrix)[0])
            else:
                quest.multiControlledMultiQubitUnitary(
                    c_register, self._controls, self._num_controls, self._targets,
                    self._num_targets, (<ComplexMatrixN*>self._matrix)[0])


cdef class CompactU(SingleQubitOperator):

    def __cinit__(self, target, alpha, beta, controls=None):
        self.TYPE = OP_TYPES.OP_COMPACT_UNITARY
        self._alpha.real = alpha.real
        self._alpha.imag = alpha.imag
        self._beta.real = beta.real
        self._beta.imag = beta.imag
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled compact unitary not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.compactUnitary(
                c_register, self._target, self._alpha, self._beta)
        if self._num_controls == 1:
            quest.controlledCompactUnitary(
                c_register, self._controls[0], self._target,
                self._alpha, self._beta)


cdef class X(MultiQubitOperator):

    def __cinit__(self, targets, controls=None):
        self.TYPE = OP_TYPES.OP_PAULI_X

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            if self._num_targets == 1:
                quest.pauliX(c_register, self._targets[0])
            else:
                quest.multiQubitNot(c_register, self._targets,
                                    self._num_targets)
        elif self._num_controls == 1 and self._num_targets == 1:
            quest.controlledNot(
                c_register, self._controls[0], self._targets[0])
        else:
            quest.multiControlledMultiQubitNot(
                c_register, self._controls, self._num_controls,
                self._targets, self._num_targets)

    @property
    def inverse(self):
        return self


cdef class Y(SingleQubitOperator):

    def __cinit__(self, target, controls=None):
        self.TYPE = OP_TYPES.OP_PAULI_Y
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled Pauli operator not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.pauliY(c_register, self._target)
        else:
            quest.controlledPauliY(
                c_register, self._controls[0], self._target)

    @property
    def inverse(self):
        return self


cdef class Z(SingleQubitOperator):
    # This class might make more sense as a MultiQubitOperator, since
    # there is not really a distinction between controls and targets.

    def __cinit__(self, target, controls=None):
        self.TYPE = OP_TYPES.OP_PAULI_Z

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef int *controls
        cdef int m
        if self._num_controls == 0:
            quest.pauliZ(c_register, self._target)
        elif self._num_controls == 1:
            quest.controlledPhaseFlip(
                c_register, self._controls[0], self._target)
        else:
            # We need a single array containing all qubits the
            # multi-controlled Pauli-Z is acting on, because of
            # the signature of multiControlledPhaseFlip(...).
            controls = <int*>malloc((self._num_controls + 1)
                                    * sizeof(controls[0]))
            controls[0] = self._target
            for m in range(self._num_controls):
                controls[m + 1] = self._controls[m]
            quest.multiControlledPhaseFlip(
                c_register, controls, self._num_controls + 1)
            free(controls)

    @property
    def inverse(self):
        return self


cdef class Swap(MultiQubitOperator):

    def __cinit__(self, targets=None, target=None):
        self.TYPE = OP_TYPES.OP_SWAP
        if self._num_targets != 2:
            raise ValueError("Swap gate must act on exactly two qubits")

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.swapGate(c_register, self._targets[0], self._targets[1])

    @property
    def inverse(self):
        return self


cdef class SqrtSwap(MultiQubitOperator):

    def __cinit__(self, targets=None, target=None):
        self.TYPE = OP_TYPES.OP_SQRT_SWAP
        if self._num_targets != 2:
            raise ValueError("Sqrt-swap gate must act on exactly two qubits.")

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.sqrtSwapGate(c_register, self._targets[0], self._targets[1])


cdef class H(SingleQubitOperator):

    def __cinit__(self, target):
        self.TYPE = OP_TYPES.OP_HADAMARD

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.hadamard(c_register, self._target)

    @property
    def inverse(self):
        return self


cdef class S(SingleQubitOperator):

    def __cinit__(self, target):
        self.TYPE = OP_TYPES.OP_S

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.sGate(c_register, self._target)


cdef class T(SingleQubitOperator):

    def __cinit__(self, target):
        self.TYPE = OP_TYPES.OP_T

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.tGate(c_register, self._target)


cdef class BaseRotate(SingleQubitOperator):

    def __cinit__(self, target, angle, *args, controls=None):
        self._angle = angle

    def __repr__(self):
        res = (type(self).__name__ + "(" + str(self.target)
               + ", " + str(self._angle))
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        return res + ")"

    def __copy__(self):
        return self.__class__(
            target=self.targets, angle=self._angle,
            controls=self.controls)

    def __getnewargs_ex__(self):
        args = (self.target, self.angle)
        kwargs = {'controls': self.controls}

    def __reduce__(self):
        return (self.__class__, (self.target, self.angle), (self.controls, ))

    def __setstate__(self, state):
        self.controls = state[0]

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

    @property
    def inverse(self):
        return type(self)(self._target, -self._angle, controls=self.controls)


cdef class Rx(BaseRotate):

    def __cinit__(self, target, angle, controls=None):
        self.TYPE = OP_TYPES.OP_ROTATE_X
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled X-rotation not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.rotateX(c_register, self._target, self._angle)
        else:
            quest.controlledRotateX(
                c_register, self._controls[0], self._target, self._angle)


cdef class Ry(BaseRotate):

    def __cinit__(self, target, angle, *, controls=None):
        self.TYPE = OP_TYPES.OP_ROTATE_Y
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled y-rotation not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.rotateY(c_register, self._target, self._angle)
        else:
            quest.controlledRotateY(
                c_register, self._controls[0], self._target, self._angle)


cdef class Rz(BaseRotate):

    def __cinit__(self, target, angle, controls=None):
        self.TYPE = OP_TYPES.OP_ROTATE_Z
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled z-rotation not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.rotateZ(c_register, self._target, self._angle)
        else:
            quest.controlledRotateZ(
                c_register, self._controls[0], self._target, self._angle)


cdef class Phase(BaseRotate):
    # Might also make more sense as a multi-qubit gate.
    def __cinit__(self, target, angle, controls=None):
        self.TYPE = OP_TYPES.OP_PHASE_SHIFT

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef size_t m
        cdef int *controls
        if self._num_controls == 0:
            quest.phaseShift(c_register, self._target, self._angle)
        elif self._num_controls == 1:
            quest.controlledPhaseShift(
                c_register, self._controls[0], self._target, self._angle)
        else:
            # The same as for multi-controlled PauliZ applies here.
            controls = <int*>malloc((self._num_controls + 1)
                                    * sizeof(controls[0]))
            controls[0] = self._target
            for m in range(self._num_controls):
                controls[m + 1] = self._controls[m]
            quest.multiControlledPhaseShift(
                c_register, controls, self._num_controls + 1, self._angle)
            free(controls)


cdef class RotateAroundAxis(BaseRotate):

    def __cinit__(self, target, angle, axis, controls=None):
        self.TYPE = OP_TYPES.OP_ROTATE_AXIS
        self._axis.x = axis[0]
        self._axis.y = axis[1]
        self._axis.z = axis[2]
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled rotation around axis not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.rotateAroundAxis(
                c_register, self._target, self._angle, self._axis)
        else:
            quest.controlledRotateAroundAxis(
                c_register, self._controls[0], self._target,
                self._angle, self._axis)


cdef class MultiRotatePauli(BaseRotate):

    def __cinit__(self, target, angle, paulis):
        self.TYPE = OP_TYPES.OP_MULTI_ROTATE
        raise NotImplementedError("Pauli multi-rotation not yet implemented.")


Unitary = U
CompactUnitary = CompactU
NOT = X
PauliX = X
PauliY = Y
PauliZ = Z
Hadamard = H
SGate = S
TGate = T
RotateX = Rx
RotateY = Ry
RotateZ = Rz
PhaseShift = Phase

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

import numpy as np
from pyquest.quest_error import QuESTError


cdef class U(MatrixOperator):

    def __cinit__(self, targets=None, matrix=None, controls=None, target=None):
        self.TYPE = OP_TYPES.OP_UNITARY

    cdef _create_array_property(self):
        # We need to overwrite this method, because core QuEST
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
                quest.controlledUnitary(
                    c_register, self._controls[0], self._targets[0],
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
                quest.multiControlledUnitary(
                    c_register, self._controls, self._num_controls,
                    self._targets[0], (<ComplexMatrix2*>self._matrix)[0])
            if self._num_targets == 2:
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


cdef class PauliOperator(SingleQubitOperator):

    def __mul__(a, b):
        """Manage multiplication with other PauliOperator instances.

        Returns a PauliProduct if the both operators are PauliOperators,
        but NotImplemented for any other type. Multiplications of
        PauliProduct with X, Y, and Z are handled by PauliProduct.
        """
        if isinstance(a, PauliOperator) and isinstance(b, PauliOperator):
            if (<PauliOperator>a)._target != (<PauliOperator>b)._target:
                return PauliProduct([a, b])
            else:
                return PauliProduct([a]) * b
        else:
            return NotImplemented

    @property
    def inverse(self):
        return self


cdef class X(PauliOperator):

    def __cinit__(self, target, controls=None):
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled Pauli-X operator not yet supported.")
        self.TYPE = OP_TYPES.OP_PAULI_X

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.pauliX(c_register, self._target)
        else:
            quest.controlledNot(
                c_register, self._controls[0], self._target)


cdef class Y(PauliOperator):

    def __cinit__(self, target, controls=None):
        self.TYPE = OP_TYPES.OP_PAULI_Y
        if self._num_controls > 1:
            raise NotImplementedError(
                "Multi-controlled Pauli-Y operator not yet supported.")

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            quest.pauliY(c_register, self._target)
        else:
            quest.controlledPauliY(
                c_register, self._controls[0], self._target)


cdef class Z(PauliOperator):
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


cdef class PauliProduct(GlobalOperator):

    PAULI_REPR = {0: '', 1: 'X', 2: 'Y', 3: 'Z'}

    MULTIPLICATION_TABLE = np.array([[0, 1, 2, 3],
                                     [1, 0, 3, 2],
                                     [2, 3, 0, 1],
                                     [3, 2, 1, 0]], dtype=np.intc)

    MULTIPLICATION_FACTORS = np.array([[0, 0, 0, 0],
                                       [0, 0, 1, 3],
                                       [0, 3, 0, 1],
                                       [0, 1, 3, 0]], dtype=np.intc)

    def __cinit__(self, pauli_factors=None):
        if pauli_factors is None:
            self._num_qubits = 0
            return
        cdef PauliOperator factor
        if isinstance(pauli_factors, BaseOperator):
            pauli_factors = [pauli_factors]
        if any([factor._num_controls for factor in pauli_factors]):
            raise ValueError("No controlled operators are allowed in PauliProduct.")
        self._num_qubits = max([factor._target for factor in pauli_factors] + [0]) + 1  # 0-based
        self._pauli_types = <pauliOpType*>calloc(self._num_qubits, sizeof(self._pauli_types[0]))
        for factor in pauli_factors:
            if self._pauli_types[factor._target] != 0:
                raise ValueError("Each qubit can only have one Pauli operator.")
            if isinstance(factor, X):
                self._pauli_types[factor._target] = pauliOpType.PAULI_X
            elif isinstance(factor, Y):
                self._pauli_types[factor._target] = pauliOpType.PAULI_Y
            elif isinstance(factor, Z):
                self._pauli_types[factor._target] = pauliOpType.PAULI_Z
            else:
                raise ValueError("Only X, Y, and Z operators "
                                 "are valid in pauli_factors")

    def __dealloc__(self):
        free(self._pauli_types)

    def __repr__(self):
        cdef size_t k
        cdef pauli_str = ""
        for k in range(self._num_qubits):
            if self._pauli_types[k] > 0:
                pauli_str += (self.PAULI_REPR[self._pauli_types[k]]
                              + "(" + str(k) + "), ")
        pauli_str = pauli_str[:-2]  # cut off last comma
        return type(self).__name__ + "([" + pauli_str + "])"

    def __mul__(a, b):
        cdef PauliProduct res
        if isinstance(a, PauliOperator):
            a = PauliProduct([a])
        if not isinstance(a, PauliProduct):
            return NotImplemented

        if isinstance(b, PauliOperator):
            b = PauliProduct([b])
        if not isinstance(b, PauliProduct):
            return NotImplemented

        res = PauliProduct()  # Does not allocate any memory.
        res._num_qubits = max((<PauliProduct>a)._num_qubits,
                              (<PauliProduct>b)._num_qubits)
        res._pauli_types = <pauliOpType*>calloc(
            res._num_qubits, sizeof(res._pauli_types[0]))
        cdef qcomp coefficient = 1
        PauliProduct._multiply_pauli_strings(
            (<PauliProduct>a)._pauli_types, (<PauliProduct>a)._num_qubits,
            (<PauliProduct>b)._pauli_types, (<PauliProduct>b)._num_qubits,
            res._pauli_types, &coefficient)

        if coefficient == 1:
            return res
        else:
            # Return a PauliSum once implemeted
            raise NotImplementedError(
                "PauliProducts cannot have repeated indices yet.")

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef size_t k
        if c_register.numQubitsRepresented < self._num_qubits:
            raise ValueError(
                f"Register does not have enough qubits for this operator. "
                f"Required {self._num_qubits}, only got {c_register.numQubitsRepresented}")
        for k in range(self._num_qubits):
            if self._pauli_types[k] == pauliOpType.PAULI_X:
                quest.pauliX(c_register, k)
            elif self._pauli_types[k] == pauliOpType.PAULI_Y:
                quest.pauliY(c_register, k)
            elif self._pauli_types[k] == pauliOpType.PAULI_Z:
                quest.pauliZ(c_register, k)

    @property
    def targets(self):
        return [k for k in range(self._num_qubits)
                if self._pauli_types[k] > 0]

    @property
    def controls(self):
        return []

    @property
    def inverse(self):
        return self

    @staticmethod
    cdef int _multiply_pauli_strings(
            pauliOpType* a_types, size_t a_num_qubits,
            pauliOpType* b_types, size_t b_num_qubits,
            pauliOpType* res_types, qcomp* coefficient) except -1:
        cdef size_t multiply_qubits = min(a_num_qubits, b_num_qubits)
        cdef size_t res_num_qubits = max(a_num_qubits, b_num_qubits)
        # We need to know which array is larger to copy the trivial
        # operators to the result.
        cdef pauliOpType* overhang_types
        if a_num_qubits > b_num_qubits:
            overhang_types = a_types
        else:
            overhang_types = b_types
        cdef size_t k
        cdef int[:, :] mult_table = PauliProduct.MULTIPLICATION_TABLE
        cdef int[:, :] mult_factors = PauliProduct.MULTIPLICATION_FACTORS
        cdef int overall_factor = 0
        for k in range(multiply_qubits):
            res_types[k] = <pauliOpType>mult_table[<int>a_types[k], <int>b_types[k]]
            overall_factor += mult_factors[<int>a_types[k], <int>b_types[k]]
        for k in range(multiply_qubits, res_num_qubits):
            res_types[k] = overhang_types[k]
        overall_factor %= 4
        coefficient[0] *= 1j ** overall_factor


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

    @property
    def inverse(self):
        return Rx(self._target, -self._angle, controls=self.controls)


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

    @property
    def inverse(self):
        return Ry(self._target, -self._angle, controls=self.controls)


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

    @property
    def inverse(self):
        return Rz(self._target, -self._angle, controls=self.controls)


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


cdef class R(GlobalOperator):

    PAULI_REPR = {0: '', 1: 'X', 2: 'Y', 3: 'Z'}

    def __cinit__(self, pauli_operators, angle):
        self.TYPE = OP_TYPES.OP_MULTI_ROTATE
        self._angle = angle
        if isinstance(pauli_operators, PauliOperator):
            pauli_operators = PauliProduct(pauli_operators)
        if not isinstance(pauli_operators, PauliProduct):
            raise TypeError("Only Pauli operators and PauliProducts are "
                            "allowd as pauli_operators.")
        self._num_qubits = (<PauliProduct>pauli_operators)._num_qubits
        self._pauli_types = <pauliOpType*>malloc(
            self._num_qubits * sizeof(self._pauli_types[0]))
        self._qubits = <int*>malloc(self._num_qubits * sizeof(self._qubits[0]))
        cdef size_t k
        for k in range(self._num_qubits):
            self._qubits[k] = k
            self._pauli_types[k] = (<PauliProduct>pauli_operators)._pauli_types[k]

    def __dealloc__(self):
        free(self._pauli_types)
        free(self._qubits)

    def __repr__(self):
        cdef size_t k
        cdef pauli_str = ""
        for k in range(self._num_qubits):
            if self._pauli_types[k] > 0:
                pauli_str += (self.PAULI_REPR[self._pauli_types[k]]
                              + "(" + str(k) + ") * ")
        pauli_str = pauli_str[:-3]  # cut off last asterisk
        return type(self).__name__ + "(" + pauli_str + ", " + str(self._angle) + ")"

    cdef int apply_to(self, Qureg c_register) except -1:
        if c_register.numQubitsRepresented < self._num_qubits:
            raise ValueError(
                f"Register does not have enough qubits for this operator. "
                f"Required {self._num_qubits}, only got {c_register.numQubitsRepresented}")
        quest.multiRotatePauli(
            c_register, self._qubits, self._pauli_types,
            self._num_qubits, self._angle)

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
MultiRotatePauli = R
Rotate = R

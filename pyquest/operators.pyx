"""Classes to represent generic operations in pyquest.

The most generic (possibly non-unitary, non-norm-preserving) operations
on quantum states are contained in this module.

Classes:
    BaseOperator:
        Abstract generic base class for all operators.
    ControlledOperator:
        Base class for all controlled operators.
    SingleQubitOperator:
        Base class for operators acting on a single qubit.
    MultiQubitOperator:
        Base class for operators potentially acting on multiple qubits.
    MatrixOperator:
        Most generic operator, specified by its matrix representation.
    PauliProduct:
        A weighted product of Pauli operators.
    PauliSum:
        A sum of ´´PauliProduct´´s.
"""

import numpy as np
import cython
import pyquest
from pyquest.quest_error import QuESTError
import logging

logger = logging.getLogger(__name__)


cdef class BaseOperator:
    """Generic base class for every class representing an operator.

    All operators on a quantum state must be derived from this class in
    order to work with pyquest. It only provides prototypes for the most
    basic tasks and returns ``NotImplementedError``s if not properly
    overridden.

    Cython attributes:
        TYPE (pyquest.quest_interface.OP_TYPES): Identifier of the
            operation for faster type checking than with
            ``IsInstance()``.
    """

    def __cinit__(self, *args, **kwargs):
        """Minimal set-up for a basic operator.

        Set TYPE to ``OP_ABSTRACT`` to identify instances of not
        properly implemented operator classes. All *args and *kwargs
        are ignored.
        """
        self.TYPE = OP_TYPES.OP_ABSTRACT

    def __repr__(self):
        return type(self).__name__ + "()"

    @property
    def inverse(self):
        """Calculate and return the inverse of an operator."""
        raise NotImplementedError(
            "inverse not implemented for this operator.")

    @property
    def targets(self):
        """Return a list of qubits this operator acts on."""
        raise NotImplementedError(
            "targets not implemented for this operator.")

    cpdef as_matrix(self, num_qubits=None):
        """Return the operator in matrix representation."""
        if num_qubits is None:
            try:
                num_qubits = max(self.targets) + 1  # 0-based index
            except NotImplementedError:
                raise ValueError(
                    "Number of qubits must be given for this operator.")
        cdef QuESTEnv *env_ptr = <QuESTEnv*>PyCapsule_GetPointer(
            pyquest.env.env_capsule, NULL)
        cdef Qureg tmp_reg = quest.createQureg(
            num_qubits, env_ptr[0])
        cdef long long k, m
        cdef long long mat_dim = 1LL << num_qubits
        cdef Complex amp
        cdef qcomp[:, :] res_mat = np.ndarray(
            (mat_dim, mat_dim), dtype=pyquest.core.np_qcomp)
        for k in range(mat_dim):
            quest.initClassicalState(tmp_reg, k)
            self.apply_to(tmp_reg)
            for m in range(mat_dim):
                amp = quest.getAmp(tmp_reg, m)
                res_mat[m, k].real = amp.real
                res_mat[m, k].imag = amp.imag
        quest.destroyQureg(tmp_reg, env_ptr[0])
        return res_mat.base

    cdef int apply_to(self, Qureg c_register) except -1:
        """Apply the operator to a ``quest_interface.Qureg``."""
        raise NotImplementedError(
            str(self) + " does not have an apply_to() method.")


cdef class ControlledOperator(BaseOperator):
    """Class implementing functionality for controlled operations.

    Abstract base class handling the keyword-only ``controls`` argument
    in the constructor and providing an interface for accessing it.
    """

    def __cinit__(self, *args, controls=None, **kwargs):
        self.controls = controls

    def __dealloc__(self):
        free(self._controls)

    def __repr__(self):
        if self._num_controls > 0:
            return type(self).__name__ + "(controls=" + str(self.controls) + ")"
        else:
            return type(self).__name__ + "()"

    @property
    def controls(self):
        cdef size_t k
        cdef list py_ctrls = [0] * self._num_controls
        for k in range(self._num_controls):
            py_ctrls[k] = self._controls[k]
        return py_ctrls

    @controls.setter
    def controls(self, value):
        cdef size_t k
        if value is None:
            self._num_controls = 0
            self._controls = NULL
            return
        try:
            self._num_controls = len(value)
            self._controls = <int*>malloc(self._num_controls
                                          * sizeof(self._controls[0]))
            for k in range(self._num_controls):
                self._controls[k] = value[k]
        except TypeError:
            try:
                # free() just in case the try branch failed after malloc
                free(self._controls)
                self._num_controls = 1
                self._controls = <int*>malloc(sizeof(self._controls[0]))
                self._controls[0] = value
            except TypeError:
                raise TypeError("Only integers and indexables of integers "
                                "are valid controls.")


cdef class SingleQubitOperator(ControlledOperator):
    """Class implementing an operator with only one target qubit.

    Abstract base class for all operator acting on a single qubit only.
    Provides properties for accessing the target qubit.
    """

    def __cinit__(self, target, *args, **kwargs):
        try:
            self._target = target
        except TypeError:
            self._target = target[0]
            if len(target) > 1:
                logger.warning(
                    "More than one target passed to single-qubit "
                    "operator. Using first target only.")

    def __repr__(self):
        res = type(self).__name__ + "(" + str(self.target)
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        return res + ")"

    def __reduce__(self):
        return (self.__class__, (self.target, ), self.controls)

    def __setstate__(self, state):
        self.controls = state

    @property
    def target(self):
        return self._target

    @property
    def targets(self):
        return [self._target]


cdef class MultiQubitOperator(ControlledOperator):
    """Class implementing an operator with multiple target qubits.

    Abstract base class for operators with (potentially) more than one
    target qubit. Provides functionality for accessing the target
    qubits.
    """

    def __cinit__(self, targets=None, *args, target=None, **kwargs):
        if targets is not None and target is not None:
            raise ValueError("Only one of the keyword arguments 'targets' "
                             "and 'target' may be supplied.")
        if targets is None:
            targets = target
        if targets is None:
            raise ValueError("One of the keyword arguments 'targets' "
                             "or 'target' must be supplied.")
        # Single- and Multi-qubit operators have their try-except
        # cases reversed such that the fastest operation is the
        # encouraged one.
        try:
            self._num_targets = len(targets)
            self._targets = <int*>malloc(self._num_targets
                                         * sizeof(self._targets[0]))
            for k in range(self._num_targets):
                self._targets[k] = targets[k]
        # If we got a scalar, there is only 1 qubit on which to act.
        except TypeError:
            try:
                self._num_targets = 1
                free(self._targets)  # In case we malloced in try.
                self._targets = <int*>malloc(sizeof(self._targets[0]))
                self._targets[0] = targets
            except TypeError:
                raise TypeError("Only integers and indexables of integers "
                                "are valid targets.")

    def __dealloc__(self):
        free(self._targets)

    def __reduce__(self):
        return (self.__class__, (self.targets, ), self.controls)

    def __setstate__(self, state):
        self.controls = state

    def __repr__(self):
        res = type(self).__name__ + "(" + str(self.targets)
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        return res + ")"

    @property
    def targets(self):
        cdef size_t k
        py_targs = [0] * self._num_targets
        for k in range(self._num_targets):
            py_targs[k] = self._targets[k]
        return py_targs

    @property
    def target(self):
        if self._num_targets == 1:
            return self._targets[0]
        return self.targets


cdef class MatrixOperator(MultiQubitOperator):

    def __cinit__(self, targets=None, matrix=None, controls=None, target=None):
        self.TYPE = OP_TYPES.OP_MATRIX
        if matrix is None:
            raise TypeError("Matrix representation must be given.")
        cdef size_t matrix_dim = 2 ** self._num_targets
        if isinstance(matrix, np.ndarray):
            if matrix.ndim != 2:
                raise ValueError("Array must have exactly 2 dimensions.")
            if matrix.shape[0] != matrix_dim or matrix.shape[1] != matrix_dim:
                raise ValueError("Matrix dimension must be "
                                 "2 ** num_qubits x 2 ** num_qubits.")
            self._create_array_property()
            self._numpy_array_to_matrix_attribute(matrix)
        else:
            raise NotImplementedError("Only numpy.ndarray matrices are "
                                      "currently supported.")

    def __dealloc__(self):
        # For 1 or 2 qubits the matrix elements are directly in the
        # _matrix attribute, so they are freed togehter with _matrix.
        if (self._matrix != NULL and (self._num_targets > 2
                                      or self._num_controls > 0)):
            destroyComplexMatrixN((<ComplexMatrixN*>self._matrix)[0])
        else:
            # The arrays of the row-pointers for 1 and 2 qubits
            # are allocated manually, because the arrays themselves
            # are in the _matrix variable.
            free(self._real)
            free(self._imag)
        free(self._matrix)

    def __repr__(self):
        res = type(self).__name__ + "(\n    " + str(self.targets) + ","

        cdef size_t matrix_dim = 2 ** self._num_targets
        cdef size_t i, j
        res += "\n    array(\n        ["
        for i in range(matrix_dim):
            res += "["
            for j in range(matrix_dim):
                res += f'{self._real[i][j]:f}{self._imag[i][j]:+f}j, '
            res = res[:-2] + "],\n         "
        res = res[:-11] + "])"
        if self._num_controls > 0:
            res += ",\n    controls=" + str(self.controls)
        return res + ")"

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            if self._num_targets == 1:
                quest.applyMatrix2(
                    c_register, self._targets[0],
                    (<ComplexMatrix2*>self._matrix)[0])
            elif self._num_targets == 2:
                quest.applyMatrix4(
                    c_register, self._targets[0], self._targets[1],
                    (<ComplexMatrix4*>self._matrix)[0])
            else:
                quest.applyMatrixN(
                    c_register, self._targets, self._num_targets,
                    (<ComplexMatrixN*>self._matrix)[0])
        else:
            quest.applyMultiControlledMatrixN(
                c_register, self._controls, self._num_controls, self._targets,
                self._num_targets, (<ComplexMatrixN*>self._matrix)[0])

    cdef _create_array_property(self):
        cdef size_t matrix_dim = 2 ** self._num_targets
        cdef size_t k
        # The only cases where core QuEST supports ComplexMatrix2
        # or ComplexMatrix4 for generic matrices are non-controlled
        # cases.
        if self._num_targets == 1 and self._num_controls == 0:
            self._matrix = malloc(sizeof(ComplexMatrix2))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = (<ComplexMatrix2*>self._matrix).real[k]
                self._imag[k] = (<ComplexMatrix2*>self._matrix).imag[k]
        elif self._num_targets == 2 and self._num_controls == 0:
            self._matrix = malloc(sizeof(ComplexMatrix4))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = (<ComplexMatrix4*>self._matrix).real[k]
                self._imag[k] = (<ComplexMatrix4*>self._matrix).imag[k]
        else:
            self._matrix = malloc(sizeof(ComplexMatrixN))
            (<ComplexMatrixN*>self._matrix)[0] = createComplexMatrixN(self._num_targets)
            self._real = (<ComplexMatrixN*>self._matrix).real
            self._imag = (<ComplexMatrixN*>self._matrix).imag

    cdef _numpy_array_to_matrix_attribute(self, np.ndarray arr):
        # For typed memoryviews we need to call different methods
        # depending on the dtype of the array.
        if arr.dtype is np.single:
            self._copy_single_array(arr)
        elif arr.dtype is np.double:
            self._copy_double_array(arr)
        elif arr.dtype is np.longdouble:
            self._copy_longdouble_array(arr)
        elif arr.dtype is np.csingle:
            self._copy_csingle_array(arr)
        elif arr.dtype is np.cdouble:
            self._copy_cdouble_array(arr)
        elif arr.dtype is np.clongdouble:
            self._copy_clongdouble_array(arr)
        else:
            # For unsupported types we use the (slow) generic copy
            # through the Python API as a fallback.
            self._copy_generic_array(arr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_generic_array(self, np.ndarray arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr.real[k, m]
                self._imag[k][m] = arr.imag[k, m]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_single_array(self, float[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr.real[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_double_array(self, double[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr.real[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_longdouble_array(self, long double[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr.real[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_csingle_array(self, float complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_cdouble_array(self, double complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_clongdouble_array(self, long double complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag


cdef class PauliSum(GlobalOperator):
    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_PAULI_SUM
        raise NotImplementedError("PauliSum not yet implemented.")


cdef class TrotterCircuit(GlobalOperator):

    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_TROTTER_CIRC
        raise NotImplementedError("TrotterCircuit not yet implemented.")


cdef class DiagonalOperator(GlobalOperator):

    def __cinit__(self, diag_elements, num_qubits):
        self.TYPE = OP_TYPES.OP_DIAGONAL
        raise NotImplementedError("DiagonalOperator not yet implemented.")

"""Classes and attributes for the core functionality of pyQuEST.

This module contains classes that are needed for the core tasks of
pyQuEST, like the QuEST environment and register, as well as some
useful properties.

Attributes:
    np_qreal (type): Numpy equivalent to the precision of real float
        in the QuEST backend (qreal).
    np_qcomp (type): Numpy equivalent to the precision of the complex
        float type in QuEST (qcomp).

Classes:
    QuESTEnvironment:
        Holding internals for the QuEST backend.
    Register:
        A quantum register holding a number of qubits.
    Circuit:
        A collection of operators to be applied to a Register at once.
"""

import logging
from weakref import WeakSet
import cython
import numpy as np

import pyquest  # The package holds our unique QuESTEnvironment
from pyquest.quest_error import QuESTError

logger = logging.getLogger(__name__)

IF QuEST_PREC == 1:
    np_qreal = np.single
    np_qcomp = np.csingle
ELIF QuEST_PREC == 2:
    np_qreal = np.double
    np_qcomp = np.cdouble
ELIF QuEST_PREC == 4:
    np_qreal = np.longdouble
    np_qcomp = np.clongdouble


cdef class QuESTEnvironment:
    """Class holding internals for the QuEST library.

    QuEST needs a ``QuESTEnv`` struct to create and destroy registers
    and other data structures, and this class wraps such a struct for
    pyQuEST to use for its operations. There should never be the need
    for the user to create an instance of this class, it should only be
    instantiated once by the ``__init__.py`` of the package.

    Registers created in a QuESTEnvironment are tracked with weak
    references and destroyed before closing the environment.

    A ``QuESTEnvironment`` object also holds information about the
    execution environment of QuEST, like number of threads, whether
    GPU acceleration is enabled, etc. These are object properties.
    """

    def __cinit__(self):
        """Create internals and extract environment properties."""
        self.c_env = quest.createQuESTEnv()
        logger.info("Created QuEST Environment at " + hex(<uintptr_t>&self.c_env))
        self._env_capsule = PyCapsule_New(<void*>&self.c_env, NULL, NULL)
        self._logged_registers = WeakSet()
        cdef char[200] env_str
        quest.getEnvironmentString(self.c_env, env_str)
        py_string = env_str.decode('UTF-8')
        prop_dict = dict([prop.split("=") for prop in py_string.split()])
        self._cuda = prop_dict['CUDA'] != '0'
        self._openmp = prop_dict['OpenMP'] != '0'
        self._mpi = prop_dict['MPI'] != '0'
        self._num_threads = int(prop_dict['threads'])
        self._num_ranks = int(prop_dict['ranks'])

    def __dealloc__(self):
        self._close()
        Py_XDECREF(<PyObject*>self._env_capsule)

    def __repr__(self):
        """Get a string containing all environment info at once."""
        return (f"{type(self).__name__}(cuda={self._cuda}, "
                f"openmp={self._openmp}, mpi={self._mpi}, "
                f"num_threads={self._num_threads}, "
                f"num_ranks={self._num_ranks}, "
                f"precision={sizeof(qreal) // 4})")

    @property
    def env_capsule(self):
        return self._env_capsule

    @property
    def logged_registers(self):
        """Get all alive registers created with this environment."""
        return str(self._logged_registers)

    @property
    def cuda(self):
        """Get whether the execution is GPU accelerated."""
        return self._cuda

    @property
    def openmp(self):
        """Get whether multithreading is enabled."""
        return self._openmp

    @property
    def mpi(self):
        """Get whether distributed execution is enabled."""
        return self._mpi

    @property
    def num_threads(self):
        """Get the number of threads for execution."""
        return self._num_threads

    @property
    def num_ranks(self):
        """Get the number of distributed processes in distributed mode.

        Each rank is processed on a single node, but may utilise
        multiple local threads.
        """
        return self._num_ranks

    @property
    def precision(self):
        """Get the precision of the QuEST floating point data type.

        Amplitudes and other variables in the QuEST library are stored
        in a float data type, whose size can be defined at compile time
        to be single (1), double (2), or quad (4) precision.
        """
        return sizeof(qreal) // 4

    def close_env(self):
        """Close the QuEST environment.

        Does not need to be called by the user during normal operation
        and has mainly debug purposes.
        """
        self._close()

    cdef log_register(self, Register reg):
        """Add a ``Register`` to the list of logged child registers."""
        self._logged_registers.add(reg)

    cdef _close(self):
        """Destroy all logged registers and close the ``QuESTEnv``."""
        cdef Register reg
        for reg in self._logged_registers:
            reg._destroy()
        logger.info("Closing QuEST Environment at " + hex(<uintptr_t>&self.c_env))
        quest.destroyQuESTEnv(self.c_env)


cdef class Register:
    """A quantum register holding a quantum state.

    Stores the amplitudes of a quantum states for a given number of
    qubits, either as a pure state vector, or a density matrix.
    """

    def __cinit__(self, int num_qubits=0, density_matrix=False,
                  Register copy_reg=None):
        """Create new register from scratch or copy an existing state.

        Arguments:
            num_qubits (int): Number of qubits in the register. For
                density matrices the number of qubits internally
                required to represent the state is 2 * num_qubits.
                Ignored if copy_reg is given.
            density_matrix (bool): True if the register should represent
                a density matrix, False otherwise. Ignored if copy_reg
                is not None. Defaults to False.
            copy_reg (pyquest.Register): If given and not None, makes
                the newly created register an exact copy of copy_reg
                while ignoring num_qubits and density_matrix parameters.
        """
        if copy_reg is None:
            if density_matrix:
                self.c_register = quest.createDensityQureg(
                    num_qubits, (<QuESTEnvironment>pyquest.env).c_env)
            else:
                self.c_register = quest.createQureg(
                    num_qubits, (<QuESTEnvironment>pyquest.env).c_env)
        else:
            self.c_register = quest.createCloneQureg(
                copy_reg.c_register, (<QuESTEnvironment>pyquest.env).c_env)
        logger.info("Created quantum register " + str(self))
        (<QuESTEnvironment>pyquest.env).log_register(self)

    def __dealloc__(self):
        self._destroy()

    def __reduce__(self):
        """Pickle the register.

        There is no way to pass the full state into the constructor, so
        the amplitudes are used as an argument for ``__setstate__``.
        """
        if self.is_density_matrix:
            state = self[:, :]
        else:
            state = self[:]
        return (self.__class__,
                (self.num_qubits, self.is_density_matrix),
                state)

    def __setstate__(self, state):
        """Set the amplitudes of the register when unpickling."""
        if self.is_density_matrix:
            self[:, :] = state
        else:
            self[:] = state

    def __mul__(self, other):
        """Get inner product with the state in ``other``.

        Currently, multiplication is only supported with other
        ``Register``s, and returns their inner product for pure states,
        or their Hilbert-Schmidt scalar product for density matrices.

        Raises:
            QuESTError: If self and other are not both state vecotrs
                or both density matrices.
        """
        # FIXME: There is a slight inconsistency here where multiplying
        #        two state vectors yields a different result from
        #        multiplying two density matrices in the same pure
        #        states.
        if isinstance(other, Register):
            return self.inner_product(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Register):
            return np.conj(self.inner_product(other))
        else:
            return NotImplemented

    def __getitem__(self, index):
        """Get amplitudes from the state.

        For state vectors, indexing supports slices and explicit
        listing of indices. Slices operate as usual, if an indexable
        is given, the elements of that indices are returned. For example

            >>> reg[[3, 6, 4]]

        returns the amplitudes of elements 3, 6, and 4, in that order.

        Density matrices act the same, but slicing and explicit indexing
        work on rows and columns independently. The amplitudes returned
        are then the intersection of all selected rows and columns. For
        example

            >>> reg[[4,6], :]

        returns all columns of rows 4 and 6,

            >>> reg[[2,3,4], [2,6]]

        fetches the elements at indices (2, 2), (2, 6), (3, 2), (3, 6),
        (4, 2), and (4, 6) in a 3 by 2 ``np.ndarray``.

        Arguments:
            index: If ``self`` is a state vector, ``index`` can be an
                integer scalar, an indexable of integer scalars, or a
                slice. For density matrices, two such elements must
                be given, where the first acts on the rows, and the
                second on the columns.

        Returns:
            complex float or np.ndarray: If queried for a single
                element, a complex float; np.ndarray for multiple
                elements. In that case, a state vector returns a 1D
                array of length ``len(index)``, a density matrix returns
                an arrays of size (len(index[0]) x len(index[1])).
        """
        if self.c_register.isDensityMatrix and len(index) != 2:
            raise TypeError("Density matrix must be accessed with 2 "
                            "indices.")
        row_index, col_index = self._fix_index(index)
        cdef qcomp[:, :] res
        if isinstance(row_index, slice):
            if isinstance(col_index, slice):
                res = self._get_state_from_slices(row_index, col_index)
            else:
                res = self._get_state_from_row_slice(row_index, col_index)
        else:
            if isinstance(col_index, slice):
                res = self._get_state_from_col_slice(row_index, col_index)
            else:
                res = self._get_state_from_indexables(row_index, col_index)
        if res.size == 1:
            return res[0, 0]
        if not self.c_register.isDensityMatrix:
            return res.base.ravel()  # Return 1D array for state vectors
        return res.base

    def __setitem__(self, index, value):
        """Manually set amplitudes in a state.

        Indexing work exactly as in ``__getitem__``. ``value`` can be
        either a scalar to be set on every selected element, or the
        dimensions of ``value`` must match the number of selected
        elements.

        No checking for valid states is performed. Currently, only state
        vectors can be set manually, density matrices cannot.

        Arguments:
            index: Has the same behaviour as in ``__getitem__``.
            value: Either a scalar to be stored in every selected
                element, or an ``np.ndarray`` with dimensions matching
                the elements selected with ``index``.
        """
        # FIXME Needs a refactor into several special-case functions and
        #       a generic handler.
        cdef size_t start, stop, k, m, num_index
        cdef int step
        cdef qreal val_imag, val_real
        cdef bool_t from_scalar
        cdef const qcomp[:] value_arr
        try:
            value[0]
            from_scalar = False
        except (TypeError, IndexError):
            from_scalar = True
        if self.c_register.isDensityMatrix:
            # This is a core QuEST limitation.
            raise NotImplementedError(
                "Manually setting elements is only supported "
                "for state vectors.")
        if isinstance(index, slice):
            start, stop, step = index.indices(self.c_register.numAmpsTotal)
            num_index = len(range(start, stop, step))
            k = start
            if not from_scalar:
                # Try using a memoryview first for performance, pull
                # values through the CPython API only if necessary.
                try:
                    value_arr = value
                    for m in range(num_index):
                        val_real = value_arr[m].real
                        val_imag = value_arr[m].imag
                        quest.setAmps(self.c_register, k, &val_real, &val_imag, 1)
                        k += step
                except (TypeError, ValueError):
                    for m in range(num_index):
                        val_real = value[m].real
                        val_imag = value[m].imag
                        # Because QuEST needs separate arrays for real
                        # and imaginary parts, we cannot hand off a
                        # pointer to a single numpy memoryview. Thus we
                        # call setAmps for each element separately.
                        quest.setAmps(self.c_register, k, &val_real, &val_imag, 1)
                        k += step
            else:
                val_real = value.real
                val_imag = value.imag
                for m in range(num_index):
                    quest.setAmps(self.c_register, k, &val_real, &val_imag, 1)
                    k += step
        else:
            try:
                num_index = len(index)
                if not from_scalar:
                    for m in range(num_index):
                        val_real = value[m].real
                        val_imag = value[m].imag
                        quest.setAmps(self.c_register, index[m], &val_real, &val_imag, 1)
                else:
                    val_real = value.real
                    val_imag = value.imag
                    for m in range(num_index):
                        quest.setAmps(self.c_register, index[m], &val_real, &val_imag, 1)
            except TypeError:  # Last guess is we got a scalar index.
                val_real = value.real
                val_imag = value.imag
                quest.setAmps(self.c_register, index, &val_real, &val_imag, 1)

    @property
    def is_alive(self):
        """Return whether the underlying QuEST structure is valid."""
        return self.c_register.numAmpsTotal > 0

    @property
    def num_qubits(self):
        """Return the number of qubits in the register."""
        return quest.getNumQubits(self.c_register)

    @property
    def num_amps(self):
        """Return the number of amplitudes stored in the register.

        This is given by 2 ** num_qubits for pure states, and
        2 ** (2 * num_qubits) for density matrices.
        """
        return quest.getNumAmps(self.c_register)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def prob_of_all_outcomes(self, qubits):
        """Calculate probabilities of all measurement outcomes.

        For all possible states of the sub-register given by ``qubits``,
        the probability of measuring each of the states is calculated
        and returned in an array.

        Arguments:
            qubits (int | array_like[int]): An object, that numpy can
                convert into an array. That array will be flattened, and
                its elements are the indices of qubits to consider for
                the measurement outcomes. For the purpose of indexing
                the output array, the entries of ``qubits`` are
                considered to be in _increasing_ order of significance.
                The qubits need not be contiguous or in any specific
                order.

        Returns:
            np.ndarray[qreal]: Probabilities of measuring each output.
                The elements in the returned array are ordered such that
                the qubits in ``qubits`` are ordered from low to high
                significance. The following table illustrates this by
                mapping the index in the returned array k to the
                measurement output corresponding to that index.

                k     |   qubits[2]    qubits[1]    qubits[0]
                0     |           0            0            0
                1     |           0            0            1
                2     |           0            1            0
                3     |           0            1            1
                4     |           1            0            0
                ...

                Index 2 of the returned array, for example, contains the
                probability of measuring only ``qubits[1]`` in state 1,
                and all other qubits in state 0. The element at index 3
                gives the probability of measuring both ``qubit[0]`` and
                ``qubit[1]`` in state 1, and all others in state 0.

                The returned array has a size of ``2**len(qubits)``.
        """
        cdef int[:] arr_qubits = np.array(qubits, dtype=np.intc,
                                          order='C', copy=False).ravel()
        cdef int num_qubits = arr_qubits.size
        cdef qreal[:] outcome_probs = np.ndarray(1 << num_qubits,
                                                 dtype=np_qreal)
        quest.calcProbOfAllOutcomes(&outcome_probs[0], self.c_register,
                                    &arr_qubits[0], num_qubits)
        return outcome_probs.base

    @property
    def purity(self):
        """Return the purity of the stored quantum state."""
        if not self.c_register.isDensityMatrix:
            return 1.
        return quest.calcPurity(self.c_register)

    @property
    def total_prob(self):
        """Return the probability of being in any state.

        For properly normalised pure states and density matrices, this
        will be 1. More accurate than self * self. For not-normalised
        density matrices, only the real part of the trace is returned.
        """
        return quest.calcTotalProb(self.c_register)

    @property
    def is_density_matrix(self):
        """Return whether the register represents a density matrix."""
        return self.c_register.isDensityMatrix != 0

    cpdef copy(self):
        """Return a new ``Register`` with a copy of the state."""
        return Register(copy_reg=self)

    cpdef copy_to(self, Register other):
        """Copy the state to another ``Register``.

        The other register must have the same number of qubits and be
        of the same type (state vector or density matrix).
        """
        quest.cloneQureg(other.c_register, self.c_register)

    cpdef copy_from(self, Register other):
        """Copy the state from another ``Register``.

        The other register must have the same number of qubits and be
        of the same type (state vector or density matrix).
        """
        quest.cloneQureg(self.c_register, other.c_register)

    def destroy_reg(self):
        """Free the resources of the underlying data structure.

        Destroys the QuEST register and frees its resources. Memory is
        automatically freed when the object is garbage collected, but
        users may sometimes want to free a register manually. If the
        register has already been freed, do nothing.
        """
        self._destroy()

    cpdef qcomp inner_product(self, Register other):
        """Calculate the inner product of self with another state.

        If both self and other are state vectors, returns the regular
        inner product of the vectors. If both are density matrices,
        returns the Hilbert-Schmidt inner product.

        Raises:
            QuESTError: If self and other are not both state vecotrs or
                both density matrices.
        """
        cdef Complex prod
        if self.c_register.isDensityMatrix:
            return quest.calcDensityInnerProduct(
                self.c_register, other.c_register)
        else:
            prod = quest.calcInnerProduct(self.c_register, other.c_register)
            return prod.real + 1j * prod.imag

    cpdef apply_circuit(self, Circuit circ):
        """Apply a ``Circuit`` to the stored state.

        If the circuit contains any measurements, their outcomes are
        returned as a (nested) list.
        """
        cdef size_t k
        cdef list meas_results = []
        for k in range(circ.c_operations.size()):
            (<BaseOperator>circ.c_operations[k]).apply_to(self.c_register)
            if (<BaseOperator>circ.c_operations[k]).TYPE == OP_TYPES.OP_MEASURE:
                meas_results.append((<M>circ.c_operations[k]).results)
        if len(meas_results) > 0:
            return meas_results

    cpdef apply_operator(self, BaseOperator op):
        """Apply a single quantum operator to the stored state.

        If the operator is a measurement, returns the measured outcomes
        as a list.
        """
        op.apply_to(self.c_register)
        if op.TYPE == OP_TYPES.OP_MEASURE:
            return (<M>op).results

    cpdef init_blank_state(self):
        """Set all amplitudes to zero."""
        quest.initBlankState(self.c_register)

    @staticmethod
    def zero_like(Register other):
        """Create a new register with the same properties as ``other``.

        The size and type (state vector or density matrix) of the newly
        created register are identical to ``other``, but the returned
        register is initialised to the zero product state.
        """
        cdef Register new_reg = Register(other.c_register.numQubitsRepresented,
                                         other.c_register.isDensityMatrix)
        return new_reg

    cdef _destroy(self):
        """Free resources of the underlying C data structure.

        Will not delete self, but free the QuEST struct containing the
        state, so the object afterwards is essentially useless.
        """
        # Only call destroyQureg if this is a valid Qureg;
        # otherwise destroyQureg will segfault.
        if self.c_register.numAmpsTotal > 0:
            logger.info("Destroying quantum register " + str(self))
            quest.destroyQureg(self.c_register, (<QuESTEnvironment>pyquest.env).c_env)
            self.c_register.numAmpsTotal = 0
        else:
            logger.debug("Trying to destroy quantum register "
                         + str(self) + ", but is already destroyed")

    # FIXME this should have an "except?" decorator, but this currently
    # crashes the cython compiler for complex return types.
    cdef qcomp _get_amp(self, size_t row, size_t col):
        """Return the amplitude at a given index.

        For state vectors, the ``col`` index is ignored, and the element
        at position ``row`` is returned. Density matrices use both
        indices as the index of the amplitude to fetch.
        """
        cdef Complex amp
        if self.c_register.isDensityMatrix:
            amp = quest.getDensityAmp(self.c_register, row, col)
        else:
            amp = quest.getAmp(self.c_register, row)
        return amp.real + 1j * amp.imag

    # These specialised functions speed up the state retrieval by having
    # C-loops over the sliced dimensions.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef qcomp[:, :] _get_state_from_slices(self, slice row_slice, slice col_slice):
        cdef size_t mat_dim = 1LL << self.c_register.numQubitsRepresented
        cdef size_t c_start, c_stop, num_cols, cur_col, r_start, r_stop, num_rows, cur_row, k, m
        cdef int c_step, r_step
        c_start, c_stop, c_step = col_slice.indices(mat_dim)
        r_start, r_stop, r_step = row_slice.indices(mat_dim)
        num_cols = len(range(c_start, c_stop, c_step))
        num_rows = len(range(r_start, r_stop, r_step))
        cdef qcomp[:, :] res_mat = np.ndarray((num_rows, num_cols), dtype=np_qcomp)
        cur_row = r_start
        for k in range(num_rows):
            cur_col = c_start
            for m in range(num_cols):
                res_mat[k, m] = self._get_amp(cur_row, cur_col)
                cur_col += c_step
            cur_row += r_step
        return res_mat

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef qcomp[:, :] _get_state_from_col_slice(self, row_index, slice col_slice):
        cdef size_t mat_dim = 1LL << self.c_register.numQubitsRepresented
        cdef size_t start, stop, num_cols, cur_row, cur_col, k, m
        cdef int step
        start, stop, step = col_slice.indices(mat_dim)
        num_cols = len(range(start, stop, step))
        cdef qcomp[:, :] res_mat = np.ndarray((len(row_index), num_cols), dtype=np_qcomp)
        for k in range(len(row_index)):
            cur_row = row_index[k]
            cur_col = start
            for m in range(num_cols):
                res_mat[k, m] = self._get_amp(cur_row, cur_col)
                cur_col += step
        return res_mat

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef qcomp[:, :] _get_state_from_row_slice(self, slice row_slice, col_index):
        cdef size_t mat_dim = 1LL << self.c_register.numQubitsRepresented
        cdef size_t start, stop, num_rows, cur_row, cur_col, k, m
        cdef int step
        start, stop, step = row_slice.indices(mat_dim)
        num_rows = len(range(start, stop, step))
        cdef qcomp[:, :] res_mat = np.ndarray((num_rows, len(col_index)), dtype=np_qcomp)
        for m in range(len(col_index)):
            cur_col = col_index[m]
            cur_row = start
            for k in range(num_rows):
                res_mat[k, m] = self._get_amp(cur_row, cur_col)
                cur_row += step
        return res_mat

    cdef qcomp[:, :] _get_state_from_indexables(self, row_index, col_index):
        cdef qcomp[:, :] res_mat = np.ndarray((len(row_index), len(col_index)), dtype=np_qcomp)
        cdef size_t k, m
        for k in range(len(row_index)):
            for m in range(len(col_index)):
                res_mat[k, m] = self._get_amp(row_index[k], col_index[m])
        return res_mat

    cdef _fix_index(self, index):
        if self.c_register.isDensityMatrix:
            row_index = index[0]
            col_index = index[1]
        else:
            row_index = index
            col_index = [0]
        if not isinstance(row_index, slice):
            try:
                row_index[0]
            except TypeError:
                row_index = [row_index]
        if not isinstance(col_index, slice):
            try:
                col_index[0]
            except TypeError:
                col_index = [col_index]
        return row_index, col_index


cdef class Circuit(GlobalOperator):
    """A collection of quantum operations in sequence.

    For now, this class is just a glorified list of objects derived from
    ``BaseOperator``, but will get more useful features in the future.
    """

    def __cinit__(self, operators=None):
        if operators is None:
            operators = []
        try:
            self.py_operations = list(operators)
        except TypeError:
            self.py_operations = [operators]
        if not all([isinstance(op, BaseOperator) for op in self.py_operations]):
            raise TypeError("All elements in 'operators' must be derived "
                            "from pyquest.BaseOperator.")
        for op in self.py_operations:
            self.c_operations.push_back(<PyObject*>op)

    def __str__(self):
        res = ""
        for py_op in self.py_operations:
            res += str(py_op) + "\n"
        return res

    def __reduce__(self):
        return (self.__class__, (self.py_operations, ))

    def __repr__(self):
        return str(self.py_operations)

    def __getitem__(self, index):
        return self.py_operations[index]

    def __len__(self):
        return len(self.py_operations)

    def __iter__(self):
        return self.py_operations.__iter__()

    def __next__(self):
        return self.py_operations.__next__()

    @property
    def inverse(self):
        return Circuit([op.inverse for op in self.py_operations[::-1]])

    def remove(self, op):
        self.py_operations.remove(op)
        self.c_operations.clear()
        for op in self.py_operations:
            self.c_operations.push_back(<PyObject*>op)

    # TODO This does not yet return measurement results.
    cdef int apply_to(self, Qureg c_register) except -1:
        cdef size_t k
        for k in range(self.c_operations.size()):
            (<BaseOperator>self.c_operations[k]).apply_to(c_register)

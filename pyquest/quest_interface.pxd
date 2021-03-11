"""C-level interface to QuEST functionality.

This pxd-only module provides the low-level interface to the QuEST API.
It also contains the type definitions for the qreal and qcomp data types
and structs used by QuEST so they can be passed as arguments to the API
functions.

The QuEST native error handling is extended to throw Python exceptions
(QuESTError) whenever an error occurs inside a QuEST function.

Attributes:
    qreal: Floating point data type used by QuEST internals.
    qcomp: Complex floating point data type equivalent to the QuEST
        data type of the same name. Rarely used, the QuEST API uses
        the ``Complex`` struct for complex numbers.
"""

IF QuEST_PREC == 1:
    ctypedef float qreal
    ctypedef float complex qcomp
ELIF QuEST_PREC == 2:
    ctypedef double qreal
    ctypedef double complex qcomp
ELIF QuEST_PREC == 4:
    ctypedef long double qreal
    ctypedef long double complex  qcomp


cdef extern from "error_handler.h":
    # Place ``#include "error_handler.h"`` in the .cxx files.
    pass


cdef extern from "QuEST.h":
    # Data structures
    ctypedef struct QuESTEnv:
        pass
    ctypedef struct Complex:
        qreal real
        qreal imag
    ctypedef struct ComplexArray:
        qreal *real
        qreal *imag
    ctypedef struct ComplexMatrix2:
        qreal real[2][2]
        qreal imag[2][2]
    ctypedef struct ComplexMatrix4:
        qreal real[4][4]
        qreal imag[4][4]
    ctypedef struct ComplexMatrixN:
        int numQubits
        qreal **real
        qreal **imag
    ctypedef struct Qureg:
        long long int numAmpsTotal
        int numQubitsRepresented
        int isDensityMatrix
    ctypedef struct DiagonalOp:
        pass
    ctypedef struct PauliHamil:
        pass
    ctypedef struct Vector:
        qreal x, y, z
    ctypedef enum pauliOpType:
        pass

    # QuESTEnv methods
    QuESTEnv createQuESTEnv() except +
    void destroyQuESTEnv(QuESTEnv env) except +
    void getEnvironmentString(QuESTEnv env, char str[200])

    # Quantum register methods
    Qureg createQureg(int numQubits, QuESTEnv env) except +
    Qureg createDensityQureg(int numQubits, QuESTEnv env) except +
    void destroyQureg(Qureg qureg, QuESTEnv env) except +

    # State initializations
    Qureg createCloneQureg(Qureg qureg, QuESTEnv env) except +
    void cloneQureg(Qureg targetQureg, Qureg copyQureg) except +
    void initBlankState(Qureg qureg) except +
    void initClassicalState(Qureg qureg, long long int stateInd) except +
    void initPlusState(Qureg qureg) except +
    void initPureState(Qureg qureg, Qureg pure) except +
    void initStateFromAmps(Qureg qureg, qreal* reals, qreal* imags) except +
    void initZeroState(Qureg qureg) except +
    void setAmps(Qureg qureg, long long int startInd, qreal* reals,
                 qreal* imags, long long int numAmps) except +
    void setWeightedQureg(Complex fac1, Qureg qureg1, Complex fac2,
                          Qureg qureg2, Complex facOut, Qureg out) except +

    # Generic operators
    ComplexMatrixN createComplexMatrixN(int numQubits) except +
    void destroyComplexMatrixN(ComplexMatrixN m) except +
    void applyDiagonalOp(Qureg qureg, DiagonalOp op) except +
    void applyMatrix2(Qureg qureg, int targetQubit, ComplexMatrix2 u) except +
    void applyMatrix4(Qureg qureg, int targetQubit1, int targetQubit2,
                      ComplexMatrix4 u) except +
    void applyMatrixN(Qureg qureg, int* targs, int numTargs,
                      ComplexMatrixN u) except +
    void applyMultiControlledMatrixN(Qureg qureg, int* ctrls, int numCtrls,
                                     int* targs, int numTargs, ComplexMatrixN u) except +
    qreal calcExpecPauliSum(
        Qureg qureg, pauliOpType* allPauliCodes, qreal* termCoeffs,
        int numSumTerms, Qureg workspace) except +
    void applyPauliSum(Qureg inQureg, pauliOpType* allPauliCodes,
                       qreal* termCoeffs, int numSumTerms, Qureg outQureg) except +
    void applyTrotterCircuit(Qureg qureg, PauliHamil hamil, qreal time,
                             int order, int reps) except +

    # Gates (measurements)
    qreal collapseToOutcome(Qureg qureg, int measureQubit,
                            int outcome) except +
    int measureWithStats(Qureg qureg, int measureQubit,
                         qreal *outcomeProb) except +

    # Unitaries
    void pauliX(Qureg qureg, int targetQubit) except +
    void controlledNot(Qureg qureg, int controlQubit, int targetQubit) except +
    void pauliY(Qureg qureg, int targetQubit) except +
    void controlledPauliY(
        Qureg qureg, int controlQubit, int targetQubit) except +
    void pauliZ(Qureg qureg, int targetQubit) except +
    void controlledPhaseFlip(Qureg qureg, int idQubit1, int idQubit2) except +
    void multiControlledPhaseFlip(
        Qureg qureg, int *controlQubits, int numControlQubits) except +
    void phaseShift(Qureg qureg, int targetQubit, qreal angle) except +
    void controlledPhaseShift(
        Qureg qureg, int idQubit1, int idQubit2, qreal angle) except +
    void multiControlledPhaseShift(
        Qureg qureg, int *controlQubits, int numControlQubits, qreal angle) except +
    void rotateX(Qureg qureg, int rotQubit, qreal angle) except +
    void controlledRotateX(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void rotateY(Qureg qureg, int rotQubit, qreal angle) except +
    void controlledRotateY(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void rotateZ(Qureg qureg, int rotQubit, qreal angle) except +
    void controlledRotateZ(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void rotateAroundAxis(
        Qureg qureg, int rotQubit, qreal angle, Vector axis) except +
    void controlledRotateAroundAxis(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle,
        Vector axis) except +
    void multiRotateZ(
        Qureg qureg, int* qubits, int numQubits, qreal angle) except +
    void multiRotatePauli(
        Qureg qureg, int* targetQubits, pauliOpType* targetPaulis,
        int numTargets, qreal angle) except +
    void hadamard(Qureg qureg, int targetQubit) except +
    void sGate(Qureg qureg, int targetQubit) except +
    void tGate(Qureg qureg, int targetQubit) except +
    void swapGate(Qureg qureg, int qubit1, int qubit2) except +
    void sqrtSwapGate(Qureg qureg, int qb1, int qb2) except +
    void compactUnitary(
        Qureg qureg, int targetQubit, Complex alpha, Complex beta) except +
    void controlledCompactUnitary(
        Qureg qureg, int controlQubit, int targetQubit,
        Complex alpha, Complex beta) except +
    void unitary(Qureg qureg, int targetQubit, ComplexMatrix2 u) except +
    void controlledUnitary(
        Qureg qureg, int controlQubit, int targetQubit, ComplexMatrix2 u) except +
    void multiControlledUnitary(
        Qureg qureg, int* controlQubits, int numControlQubits,
        int targetQubit, ComplexMatrix2 u) except +
    void multiStateControlledUnitary(
        Qureg qureg, int* controlQubits, int* controlState,
        int numControlQubits, int targetQubit, ComplexMatrix2 u) except +
    void twoQubitUnitary(
        Qureg qureg, int targetQubit1, int targetQubit2, ComplexMatrix4 u) except +
    void controlledTwoQubitUnitary(
        Qureg qureg, int controlQubit, int targetQubit1, int targetQubit2,
        ComplexMatrix4 u) except +
    void multiControlledTwoQubitUnitary(
        Qureg qureg, int* controlQubits, int numControlQubits,
        int targetQubit1, int targetQubit2, ComplexMatrix4 u) except +
    void multiQubitUnitary(
        Qureg qureg, int* targs, int numTargs, ComplexMatrixN u) except +
    void controlledMultiQubitUnitary(
        Qureg qureg, int ctrl, int* targs, int numTargs, ComplexMatrixN u) except +
    void multiControlledMultiQubitUnitary(
        Qureg qureg, int* ctrls, int numCtrls, int* targs,
        int numTargs, ComplexMatrixN u) except +
    void mixDamping(Qureg qureg, int targetQubit, qreal prob) except +
    void mixDensityMatrix(Qureg combineQureg, qreal prob, Qureg otherQureg) except +
    void mixDephasing(Qureg qureg, int targetQubit, qreal prob) except +
    void mixDepolarising(Qureg qureg, int targetQubit, qreal prob) except +
    void mixKrausMap(Qureg qureg, int target, ComplexMatrix2 *ops, int numOps) except +
    void mixMultiQubitKrausMap(
        Qureg qureg, int* targets, int numTargets,
        ComplexMatrixN* ops, int numOps) except +
    void mixPauli(Qureg qureg, int targetQubit, qreal probX, qreal probY, qreal probZ) except +
    void mixTwoQubitDephasing(Qureg qureg, int qubit1, int qubit2, qreal prob) except +
    void mixTwoQubitDepolarising(Qureg qureg, int qubit1, int qubit2, qreal prob) except +
    void mixTwoQubitKrausMap(
        Qureg qureg, int target1, int target2, ComplexMatrix4 *ops, int numOps) except +

    # Calculations
    int getNumQubits(Qureg qureg) except +
    long long int getNumAmps(Qureg qureg) except +
    qreal calcPurity(Qureg qureg) except +
    qreal calcTotalProb(Qureg qureg) except +
    Complex calcInnerProduct(Qureg bra, Qureg ket) except +
    qreal calcDensityInnerProduct(Qureg rho1, Qureg rho2) except +
    Complex getAmp(Qureg qureg, long long int index) except +
    Complex getDensityAmp(Qureg qureg, long long int row, long long int col) except +
    void setAmps(Qureg qureg, long long int startInd,
                 qreal* reals, qreal* imags, long long int numAmps) except +
    qreal calcProbOfOutcome(Qureg qureg, int measureQubit, int outcome) except +


cdef enum OP_TYPES:
    # The OP_TYPES are defined here because if they are kept in core,
    # they lead to cyclic imports when cimported to the modules
    # containing the operator classes.
    OP_ABSTRACT  # Safeguard for abstract operators.
    OP_DIAGONAL
    OP_MATRIX
    OP_PAULI_PRODUCT
    OP_PAULI_SUM
    OP_TROTTER_CIRC
    OP_MEASURE
    OP_PAULI_X
    OP_PAULI_Y
    OP_PAULI_Z
    OP_SWAP
    OP_SQRT_SWAP
    OP_HADAMARD
    OP_S
    OP_T
    OP_ROTATE_X
    OP_ROTATE_Y
    OP_ROTATE_Z
    OP_PHASE_SHIFT
    OP_ROTATE_AXIS
    OP_MULTI_ROTATE
    OP_UNITARY
    OP_COMPACT_UNITARY
    OP_DAMP
    OP_DEPHASE
    OP_DEPOL
    OP_KRAUS
    OP_PAULI_NOISE
    OP_MIX_DENSITY
    OP_INIT_BLANK
    OP_INIT_CLASSICAL
    OP_INIT_PLUS
    OP_INIT_PURE
    OP_INIT_ZERO

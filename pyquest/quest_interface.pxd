"""C-level interface to QuEST v4 functionality.

This pxd-only module provides the low-level interface to the QuEST v4 API.
It also contains the type definitions for the qreal and qcomp data types
and structs used by QuEST so they can be passed as arguments to the API
functions.

The QuEST native error handling is extended to throw Python exceptions
(QuESTError) whenever an error occurs inside a QuEST function.

Attributes:
    qreal: Floating point data type used by QuEST internals.
    qcomp: Complex floating point data type used by QuEST.
"""

# Precision types
IF QuEST_PREC == 1:
    ctypedef float qreal
    ctypedef float complex qcomp
ELIF QuEST_PREC == 2:
    ctypedef double qreal
    ctypedef double complex qcomp
ELIF QuEST_PREC == 4:
    ctypedef long double qreal
    ctypedef long double complex qcomp

ctypedef long long int qindex

cdef extern from "quest_error.h":
    pass

# Vector structure (needed for rotation functions)
# Defined as pure Cython struct, not from quest.h (doesn't exist in v4)
ctypedef struct Vector:
    qreal x
    qreal y
    qreal z

cdef extern from "quest.h":
    # Opaque environment structure (v4 API)
    ctypedef struct QuESTEnv:
        int isMultithreaded
        int isGpuAccelerated
        int isDistributed
        int isCuQuantumEnabled
        int isGpuSharingEnabled
        int rank
        int numNodes
    
    # Quantum register structure (v4 API)
    ctypedef struct Qureg:
        int isMultithreaded
        int isGpuAccelerated
        int isDistributed
        int rank
        int numNodes
        int logNumNodes
        int isDensityMatrix
        int numQubits
        qindex numAmps
        qindex logNumAmps
        qindex numAmpsPerNode
        qindex logNumAmpsPerNode

cdef extern from "quest.h":
    # Pauli operator enumeration
    ctypedef enum pauliOpType:
        PAULI_I=0
        PAULI_X=1
        PAULI_Y=2
        PAULI_Z=3

cdef extern from "quest.h":
    # Fixed-size 1-qubit matrix (2x2)
    ctypedef struct CompMatr1:
        int numQubits
        qindex numRows
        qcomp elems[2][2]
    
    # Fixed-size 2-qubit matrix (4x4)
    ctypedef struct CompMatr2:
        int numQubits
        qindex numRows
        qcomp elems[4][4]
    
    # Variable-size N-qubit matrix
    ctypedef struct CompMatr:
        int numQubits
        qindex numRows
        int* isApproxUnitary
        int* isApproxHermitian
        int* wasGpuSynced
        qcomp** cpuElems
        qcomp* cpuElemsFlat
        qcomp* gpuElemsFlat
    
    # Fixed-size 1-qubit diagonal matrix
    ctypedef struct DiagMatr1:
        int numQubits
        qindex numElems
        qcomp elems[2]
    
    # Fixed-size 2-qubit diagonal matrix
    ctypedef struct DiagMatr2:
        int numQubits
        qindex numElems
        qcomp elems[4]
    
    # Variable-size diagonal matrix
    ctypedef struct DiagMatr:
        int numQubits
        qindex numElems
        int* isApproxUnitary
        int* isApproxHermitian
        int* isApproxNonZero
        int* isStrictlyNonNegative
        int* wasGpuSynced
        qcomp* cpuElems
        qcomp* gpuElems
    
    # Full-state diagonal matrix
    ctypedef struct FullStateDiagMatr:
        int numQubits
        qindex numElems
        int isGpuAccelerated
        int isMultithreaded
        int isDistributed
        qindex numElemsPerNode
        int* isApproxUnitary
        int* isApproxHermitian
        int* isApproxNonZero
        int* isStrictlyNonNegative
        int* wasGpuSynced
        qcomp* cpuElems
        qcomp* gpuElems

    # Environment functions (v4 API)
    void initQuESTEnv() except +
    QuESTEnv getQuESTEnv() except +
    void finalizeQuESTEnv() except +
    
    # Environment queries (v4 API)
    void getEnvironmentString(char str[200]) except +
    
    # Seeding functions (v4 API)
    void setSeeds(unsigned* seeds, int numSeeds) except +
    void setSeedsToDefault() except +
    void getSeeds(unsigned* seeds) except +
    int getNumSeeds() except +

    # Qureg functions
    Qureg createQureg(int numQubits) except +
    Qureg createDensityQureg(int numQubits) except +
    Qureg createCloneQureg(Qureg qureg) except +
    void destroyQureg(Qureg qureg) except +
    void cloneQureg(Qureg targetQureg, Qureg copyQureg) except +
    void setQuregToClone(Qureg targetQureg, Qureg copyQureg) except +

    # State initializations
    void initBlankState(Qureg qureg) except +
    void initClassicalState(Qureg qureg, long long int stateInd) except +
    void initPlusState(Qureg qureg) except +
    void initPureState(Qureg qureg, Qureg pure) except +
    void initStateFromAmps(Qureg qureg, qreal* reals, qreal* imags) except +
    void initZeroState(Qureg qureg) except +
    void setAmps(Qureg qureg, long long int startInd, qreal* reals,
                 qreal* imags, long long int numAmps) except +
    void setQuregAmps(Qureg qureg, long long int startIdx, qcomp* amps, long long int numAmps) except +
    void setWeightedQureg(qcomp fac1, Qureg qureg1, qcomp fac2,
                          Qureg qureg2, qcomp facOut, Qureg out) except +
    void setQuregToWeightedSum(Qureg out, qcomp* coeffs, Qureg* quregs, int numQuregs) except +

    # Generic operators (v4 API)
    CompMatr1 getCompMatr1(qcomp** in_) except +
    CompMatr2 getCompMatr2(qcomp* in_) except +
    CompMatr createCompMatr(int numQubits) except +
    void destroyCompMatr(CompMatr m) except +
    void setCompMatr(CompMatr out, qcomp** in_) except +
    
    # Diagonal operators (v4 API)
    FullStateDiagMatr createFullStateDiagMatr(int numQubits) except +
    void destroyFullStateDiagMatr(FullStateDiagMatr matr) except +
    void setFullStateDiagMatr(FullStateDiagMatr out, qindex startIdx, qcomp* in_, qindex numElems) except +
    void syncFullStateDiagMatr(FullStateDiagMatr matr) except +
    void leftapplyFullStateDiagMatr(Qureg qureg, FullStateDiagMatr matrix) except +
    
    # Diagonal operators from functions (v4 API - new for PhaseFunc refactoring)
    void setFullStateDiagMatrFromMultiVarFunc(
        FullStateDiagMatr out,
        qcomp (*func)(qindex*),
        int* numQubitsPerVar,
        int numVars,
        int areSigned
    ) except +
    
    void setDiagMatrFromMultiVarFunc(
        DiagMatr out,
        qcomp (*func)(qindex*),
        int* numQubitsPerVar,
        int numVars,
        int areSigned
    ) except +
    # Matrix application (v4 API)
    void leftapplyCompMatr1(Qureg qureg, int target, CompMatr1 matrix) except +
    void leftapplyCompMatr2(Qureg qureg, int target1, int target2, CompMatr2 matrix) except +
    void leftapplyCompMatr(Qureg qureg, int* targets, int numTargets,
                           CompMatr matrix) except +
    void applyMultiControlledMatrixN(Qureg qureg, int* ctrls, int numCtrls,
                                     int* targs, int numTargs, CompMatr u) except +
    qreal calcExpecPauliSum(
        Qureg qureg, pauliOpType* allPauliCodes, qreal* termCoeffs,
        int numSumTerms, Qureg workspace) except +
    void applyPauliSum(Qureg inQureg, pauliOpType* allPauliCodes,
                       qreal* termCoeffs, int numSumTerms, Qureg outQureg) except +
    void applyTrotterCircuit(Qureg qureg, int time, qreal period) except +
    void applyQuantumFourierTransform(Qureg qureg, int* qubits, int numQubits) except +
    void applyFullQuantumFourierTransform(Qureg qureg) except +

    # Gates (measurements)
    qreal collapseToOutcome(Qureg qureg, int measureQubit,
                            int outcome) except +
    int measureWithStats(Qureg qureg, int measureQubit,
                         qreal *outcomeProb) except +

    # Unitaries
    void pauliX(Qureg qureg, int targetQubit) except +
    void controlledNot(Qureg qureg, int controlQubit, int targetQubit) except +
    void multiQubitNot(Qureg qureg, int* targs, int numTargs) except +
    void multiControlledMultiQubitNot(Qureg qureg, int* ctrls, int numCtrls,
                                      int* targs, int numTargs) except +
    void pauliY(Qureg qureg, int targetQubit) except +
    void controlledPauliY(
        Qureg qureg, int controlQubit, int targetQubit) except +
    void pauliZ(Qureg qureg, int targetQubit) except +
    void applyTwoQubitPhaseFlip(Qureg qureg, int idQubit1, int idQubit2) except +
    void applyMultiQubitPhaseFlip(
        Qureg qureg, int *controlQubits, int numControlQubits) except +
    void applyPhaseShift(Qureg qureg, int targetQubit, qreal angle) except +
    void applyTwoQubitPhaseShift(
        Qureg qureg, int idQubit1, int idQubit2, qreal angle) except +
    void applyMultiQubitPhaseShift(
        Qureg qureg, int *controlQubits, int numControlQubits, qreal angle) except +
    void applyRotateX(Qureg qureg, int rotQubit, qreal angle) except +
    void applyControlledRotateX(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void applyRotateY(Qureg qureg, int rotQubit, qreal angle) except +
    void applyControlledRotateY(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void applyRotateZ(Qureg qureg, int rotQubit, qreal angle) except +
    void applyControlledRotateZ(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle) except +
    void applyRotateAroundAxis(
        Qureg qureg, int rotQubit, qreal angle, qreal vx, qreal vy, qreal vz) except +
    void applyControlledRotateAroundAxis(
        Qureg qureg, int controlQubit, int targetQubit, qreal angle,
        qreal vx, qreal vy, qreal vz) except +
    void multiRotateZ(
        Qureg qureg, int* qubits, int numQubits, qreal angle) except +
    void multiRotatePauli(
        Qureg qureg, int* targetQubits, pauliOpType* targetPaulis,
        int numTargets, qreal angle) except +

cdef extern from "quest.h":
    void applyHadamard(Qureg qureg, int targetQubit) except +
    void applyS(Qureg qureg, int targetQubit) except +
    void applyT(Qureg qureg, int targetQubit) except +
    void applySwap(Qureg qureg, int qubit1, int qubit2) except +
    void applySqrtSwap(Qureg qureg, int qb1, int qb2) except +
    void compactUnitary(
        Qureg qureg, int targetQubit, qcomp alpha, qcomp beta) except +
    void controlledCompactUnitary(
        Qureg qureg, int controlQubit, int targetQubit,
        qcomp alpha, qcomp beta) except +
    # Unitaries (v4 API)
    void applyCompMatr1(Qureg qureg, int targetQubit, CompMatr1 u) except +
    void applyControlledCompMatr1(Qureg qureg, int control, int target, CompMatr1 matrix) except +
    void applyMultiControlledCompMatr1(Qureg qureg, int* controls, int numControls, int target, CompMatr1 matrix) except +
    void applyMultiStateControlledCompMatr1(Qureg qureg, int* controls, int* states, int numControls, int target, CompMatr1 matrix) except +
    void applyCompMatr2(Qureg qureg, int targetQubit1, int targetQubit2,
                        CompMatr2 u) except +
    void applyControlledCompMatr2(Qureg qureg, int control, int target1, int target2, CompMatr2 matrix) except +
    void applyMultiControlledCompMatr2(Qureg qureg, int* controls, int numControls, int target1, int target2, CompMatr2 matrix) except +
    void applyMultiStateControlledCompMatr2(Qureg qureg, int* controls, int* states, int numControls, int target1, int target2, CompMatr2 matrix) except +
    void applyCompMatr(Qureg qureg, int* targets, int numTargets,
                       CompMatr u) except +
    void applyControlledCompMatr(Qureg qureg, int control, int* targets, int numTargets, CompMatr matrix) except +
    void applyMultiControlledCompMatr(Qureg qureg, int* controls, int numControls, int* targets, int numTargets, CompMatr matrix) except +
    void applyMultiStateControlledCompMatr(Qureg qureg, int* controls, int* states, int numControls, int* targets, int numTargets, CompMatr matrix) except +
    
    # Pauli Gates (v4 API)
    void applyPauliX(Qureg qureg, int target) except +
    void applyPauliY(Qureg qureg, int target) except +
    void applyPauliZ(Qureg qureg, int target) except +
    void applyControlledPauliX(Qureg qureg, int control, int target) except +
    void applyControlledPauliY(Qureg qureg, int control, int target) except +
    void applyControlledPauliZ(Qureg qureg, int control, int target) except +
    void applyMultiControlledPauliX(Qureg qureg, int* controls, int numControls, int target) except +
    void applyMultiControlledPauliY(Qureg qureg, int* controls, int numControls, int target) except +
    void applyMultiControlledPauliZ(Qureg qureg, int* controls, int numControls, int target) except +
    
    # NOT Gates (v4 API)
    void applyMultiQubitNot(Qureg qureg, int* targets, int numTargets) except +
    void applyControlledMultiQubitNot(Qureg qureg, int control, int* targets, int numTargets) except +
    void applyMultiControlledMultiQubitNot(Qureg qureg, int* controls, int numControls, int* targets, int numTargets) except +
    
    # Phase Flip Gates (v4 API)
    void applyPhaseFlip(Qureg qureg, int target) except +
    void applyTwoQubitPhaseFlip(Qureg qureg, int control, int target) except +
    void applyMultiQubitPhaseFlip(Qureg qureg, int* qubits, int numQubits) except +
    
    # Measurement Functions (v4 API)
    int applyQubitMeasurementAndGetProb(Qureg qureg, int qubit, qreal *outcomeProb) except +
    int applyForcedQubitMeasurement(Qureg qureg, int qubit, int outcome) except +
    
    void mixDamping(Qureg qureg, int targetQubit, qreal prob) except +
    void mixDensityMatrix(Qureg combineQureg, qreal prob, Qureg otherQureg) except +
    void mixQureg(Qureg qureg, Qureg otherQureg, qreal prob) except +
    void mixDephasing(Qureg qureg, int targetQubit, qreal prob) except +
    void mixDepolarising(Qureg qureg, int targetQubit, qreal prob) except +
    void mixKrausMap(Qureg qureg, int target, CompMatr1 *ops, int numOps) except +
    void mixMultiQubitKrausMap(
        Qureg qureg, int* targets, int numTargets,
        CompMatr* ops, int numOps) except +
    void mixPauli(Qureg qureg, int targetQubit, qreal probX, qreal probY, qreal probZ) except +
    void mixPaulis(Qureg qureg, int targetQubit, qreal probX, qreal probY, qreal probZ) except +
    void mixTwoQubitDephasing(Qureg qureg, int qubit1, int qubit2, qreal prob) except +
    void mixTwoQubitDepolarising(Qureg qureg, int qubit1, int qubit2, qreal prob) except +
    void mixTwoQubitKrausMap(
        Qureg qureg, int target1, int target2, CompMatr2 *ops, int numOps) except +

    # Calculations
    int getNumQubits(Qureg qureg) except +
    long long int getNumAmps(Qureg qureg) except +
    qreal calcPurity(Qureg qureg) except +
    qreal calcTotalProb(Qureg qureg) except +
    qcomp calcInnerProduct(Qureg bra, Qureg ket) except +
    qreal calcDensityInnerProduct(Qureg rho1, Qureg rho2) except +
    qreal calcFidelity(Qureg qureg, Qureg pureState) except +
    qcomp getQuregAmp(Qureg qureg, long long int index) except +
    qcomp getDensityQuregAmp(Qureg qureg, long long int row, long long int col) except +
    void setQuregAmps(Qureg qureg, long long int startIdx, qcomp* amps, long long int numAmps) except +
    void setQuregToClone(Qureg targetQureg, Qureg copyQureg) except +
    qreal calcProbOfOutcome(Qureg qureg, int measureQubit, int outcome) except +
    void calcProbOfAllOutcomes(qreal* outcomeProbs, Qureg qureg,
                               int* qubits, int numQubits) except +
    void calcProbsOfAllMultiQubitOutcomes(qreal* outcomeProbs, Qureg qureg,
                                          int* qubits, int numQubits) except +


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
    OP_QFT
    OP_FULL_QFT
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

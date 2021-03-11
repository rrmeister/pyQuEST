# pyQuEST

A Python interface for the Quantum Exact Simulation Toolkit (QuEST) written mainly in Cython.

## Getting started
After cloning the repository

```console
$ git clone -b develop --recursive https://github.com/rrmeister/pyQuEST
$ cd pyQuEST
```

it is recommended to create a virtual environment, e.g. with `venv`.

```console
$ python3 -m venv .
$ source bin/activate
```

To configure the QuEST backend for multithreading, GPU usage, float precision, etc., have a look at the QuEST build documentation.

The package can then be installed using pip.

```console
$ pip3 install .
```

## Usage
After successful installation, we can test pyQuEST by importing it and having a look at the environment it is running in.

```python
In [1]: import pyquest

In [2]: pyquest.env
Out[2]: QuESTEnvironment(cuda=False, openmp=False, mpi=False, num_threads=1, num_ranks=1, precision=2)
```

The `QuESTEnvironment` class is automatically instantiated once upon module import and never needs to be called by the user. It contains internals and can return information about the execution environment, as above.

### Example
The most important classes are `Register` representing a quantum register, and the operators which can be applied to it. Let's create such a register with 3 qubits and look at its contents.

```python
In [3]: from pyquest import Register

In [4]: from pyquest.unitaries import *

In [5]: reg = Register(3)

In [6]: reg[:]
Out[6]: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
```

Like in QuEST, the state is automatically initialised to the all-zero product state. To apply some gates to it, first import some unitaries and the `Circuit` class.

```python
In [7]: from pyquest.unitaries import H, X, Ry

In [8]: from pyquest import Circuit
```

The operators are constructed from their classes with their target qubits, then any additional parameters (like a rotation angle), and then the control qubits as a keyword-only argument, e.g. `Ry(0, .2, controls=[1])` creates a rotation operator about the y-axis of the Bloch sphere by 0.2 radians on qubit 0, controlled by qubit 1. A single operator can be applied to a register `reg` with `reg.apply_operator(X(1))`. To apply multiple operators at once, first collect them into a `Circuit`.

```python
In [9]: circ = Circuit([H(0), X(2, controls=[0]), Ry(1, .23, controls=[2])])

In [10]: reg.apply_circuit(circ)

In [11]: reg[:]
Out[11]:
array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
       0.        +0.j, 0.70243619+0.j, 0.        +0.j, 0.08113816+0.j])
```

Multiplying two registers together will return their inner product. For example, the expectation value of `X(1)` is

```python
In [12]: temp = Register(copy_reg=reg)

In [13]: temp.apply_operator(X(1))

In [14]: reg * temp
Out[14]: (0.11398876176759418+0j)
```

A measurement can be performed with

```python
In [15]: from pyquest.gates import M

In [16]: reg.apply_operator(M(1))
Out[16]: [0]
```

Remember that measurements are destructive.

```python
In [17]: reg[:]
Out[17]:
array([0.70944592+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
       0.        +0.j, 0.70475988+0.j, 0.        +0.j, 0.        +0.j])
```

A register can also be directly modified, but may then end up in an invalid state if not properly normalised.
```python
In [18]: reg[:] = 0

In [19]: reg[:]
Out[19]: array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

In [20]: reg[4] = 1j

In [21]: reg[:]
Out[21]: array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j])
```

For further details on which operators are available and all methods of `Register` and `Circuit`, for now check out the documentation in the source files.

## Development
If you want to contribute to the project or just play around with the source, you can use an editable install

```console
$ pip3 install -e .
```

but this temporarily re-installs the build requirements every time the project is compiled and clutters the `pyquest` source directory with build artifacts. It is therefore a good idea to simply install the build requirements manually

```console
$ pip3 install -r build_requirements.txt
```

and then call

```console
$ python3 setup.py build
```

to build pyQuEST in the `_skbuild/<os_id>/cmake-install/pyquest` folder, or use

```console
$ python3 setup.py install
```
to install the package without telling `pip` about it. This is really only recommended in a virtual environment.

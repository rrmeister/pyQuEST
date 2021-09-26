"""Python wrapper for the Quantum Exact Simulation Toolkit (QuEST).

This package provides an easy to use Python interface for the QuEST
library, which is written in C.
"""

from pyquest.core import QuESTEnvironment, Register, Circuit  # noqa: F401

env = QuESTEnvironment()
"""Package level variable holding the only instance of QuESTEnvironment.

To properly initialize a QuESTEnv, create a single instance at
package load time and have every function needing it use this instance.
This also allows user checking of the environment via ``pyquest.env``.
"""

cuda = env.cuda
openmp = env.openmp
mpi = env.mpi
num_threads = env.num_threads
num_ranks = env.num_ranks
precision = env.precision
rank = env.rank
"""Promote environment properties to the package level.

These attributes are only for convenience, so individual properties of
the environment can be checked at the package level,
e.g. with ``pyquest.precision``.
"""

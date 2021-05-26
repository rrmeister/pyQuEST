import sys
from skbuild import setup

# scikit-build disables docstrings for "Release" and "MinSizeRel"
# builds. To prevent this in a hacky way, set the default build type to
# a new (made up) value "RelWithDocs".

# Check if there is a user-specified build type in the arguments;
# if not, set it to the new default "RelWithDocs".
for arg in sys.argv:
    # '--' marks the end of the skbuild arguments
    if arg == '--' or arg.startswith('--build-type'):
        break
else:
    sys.argv.insert(0, '--build-type=RelWithDocs')

setup(
    name="pyquest",
    version="0.0.1",
    author="Richard Meister",
    author_email="richardm.tug@gmail.com",
    url="https://github.com/rrmeister/pyQuEST",
    packages=["pyquest"],
    description="A Python interface for QuEST.",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmake_args=["-DPRECISION:STRING=2"],
    install_requires=["numpy>=1.20"],
)

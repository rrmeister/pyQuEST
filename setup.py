import sys
from skbuild import setup

# ----------------------------------------------------------------------
# QuEST compilation settings
# ----------------------------------------------------------------------
quest_config = {
    'precision': 2,            # Size of a float; 1 (single),
                               # 2 (double), or 4 (quad precision).
    'multithreaded': True,     # Enable multithreading via OpenMP.
    'distributed': False,      # Enable distributed code via MPI.
    'gpu_accelerated': False,  # Enable Nvidia GPU support via CUDA.
    # CUDA needs to know the compute capability of your GPU. Find your
    # GPU model on https://developer.nvidia.com/cuda-gpus and set
    # 'gpu_compute_capability' to the "Compute Capability" listed
    # there, without the period (e.g. 30 for 3.0).
    'gpu_compute_capability': None,
}
# ----------------------------------------------------------------------


# Validate QuEST compilation configuration; these are re-checked by
# the QuEST cmake configuration, but raising appropriate exceptions
# here makes them easier to spot and fix for the user.

if quest_config['gpu_accelerated']:
    if quest_config['precision'] == 4:
        raise ValueError("Quad precision ('precision': 4) not supported on"
                         "CUDA devices. Use lower precision or set 'cuda' "
                         "to False.")
    if quest_config['gpu_compute_capability'] is None:
        raise TypeError("When compiling with GPU support, "
                        "'cuda_compute_capability' must be set.")
    if quest_config['multithreaded']:
        raise ValueError("GPU acceleration and multithreading cannot be used "
                         "at the same time. Disable one of 'gpu_accelerated' "
                         "and 'multithreaded'.")

if quest_config['precision'] not in [1, 2, 4]:
    raise ValueError("Precision must be either 1 (single), 2 (double), "
                     "or 4 (quad precision)")


# scikit-build disables docstrings for "Release" and "MinSizeRel"
# builds. To prevent this in a hacky way, set the default build type to
# a new (made up) value "RelWithDocs".

# Check if there is a user-specified build type in the arguments;
# if not, set it to the new default "RelWithDocs".
insert_at = len(sys.argv)
for k, arg in enumerate(sys.argv):
    if arg.startswith('--build-type'):
        insert_at = None
        break
    # '--' marks the end of setuptools-arguments, so we must insert
    # before this separator.
    if arg == '--':
        insert_at = k
        break
if insert_at is not None:
    sys.argv.insert(insert_at, '--build-type=RelWithDocs')


quest_cmake_args = [
    "-DPRECISION:STRING=" + str(quest_config['precision']),
    "-DMULTITHREADED:BOOL=" + ("ON" if quest_config['multithreaded']
                               else "OFF"),
    "-DDISTRIBUTED:BOOL=" + ("ON" if quest_config['distributed']
                             else "OFF"),
    "-DGPUACCELERATED:BOOL=" + ("ON" if quest_config['gpu_accelerated']
                                else "OFF")]
if quest_config['gpu_accelerated']:
    quest_cmake_args.append(
        "-DGPU_COMPUTE_CAPABILITY:STRING="
        + str(quest_config['gpu_compute_capability']))


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
    cmake_args=quest_cmake_args,
    install_requires=["numpy>=1.20"],
)

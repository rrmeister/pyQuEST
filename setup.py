from skbuild import setup

setup(
    name="pyquest",
    version="0.0.1",
    packages=['pyquest'],
    cmake_args=['-DPRECISION:STRING=2'],
    install_requires=["numpy"],
)

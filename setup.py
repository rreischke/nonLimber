from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension

__version__ = "0.0.2"

ext_modules = [
    Pybind11Extension(
        "levinpower",
        ["src/Levin_power.cpp", "python/pybind11_interface.cpp"],
        cxx_std=11,
        include_dirs=["src"],
        libraries=["m", "gsl", "gslcblas"],
        extra_compile_args=["-Xpreprocessor", "-fopenmp"],
        ),
]

setup(
    name="levinpower",
    version=__version__,
    # author="Robert Reischke",
    # author_email="s",
    # url="",
    # description="",
    # long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
)

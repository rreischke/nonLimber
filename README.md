# Calculating the non-Limber expression for the angular power spectrum using the Levin integration. 

The integral is described in https://github.com/LSSTDESC/N5K/blob/master/README.md and the Levin method was originally proposed here: https://www.sciencedirect.com/science/article/pii/0377042794001189 .

## Compiling
The code is compiled by typing make install. In the main directory there is another makefile for an example main. The documentation can be created via doxygen -g doxyfile followed by doxygen doxyfile .
Only standard libraries and the GSL are needed.

For the python module, `pip install .` (or `pip install -e .` for a `develop` install) should do the trick. An up-to-date version of `pip` (10+) is required.
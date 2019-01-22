#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Run with python aode-setup.py build_ext --inplace

from distutils.core import setup, Extension
import numpy as np
import os

os.environ["CC"] = "/opt/local/bin/gcc-mp-5"
#os.environ["XCC"] = "gcc-mp-5"
#'-I/opt/local/include/libomp', '-L/opt/local/lib/libomp',
ext_modules = [Extension('runclassifier', sources=['runclassifier.c'],
	extra_compile_args=['-fopenmp'],
	extra_link_args=['-lgomp'],
	)]

setup(
	name = 'RunClassifier',
	version = '1.0',
	include_dirs = [np.get_include()], #Add Include path of numpy
	ext_modules = ext_modules,
)
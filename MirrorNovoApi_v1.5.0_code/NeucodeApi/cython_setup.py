

import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy

# python cython_setup.py build_ext --inplace

setup(ext_modules=cythonize([
      "config.pyx",
        "data_loader.pyx",
        "denovo.pyx",
        "init_args.pyx",
        "model.pyx",
        "train_func.pyx",
        "writer.pyx"
]))

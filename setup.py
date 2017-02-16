from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import scipy

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
                Extension("py_signet",
                sources=["py_signet.pyx"],
                include_dirs=[".", numpy.get_include(), scipy.get_include(), "/usr/local/include/"], language='c++',
                extra_link_args=["-L/usr/local/lib/", "-lgsl", "-lgslcblas", "-lm"],),
                ]

)
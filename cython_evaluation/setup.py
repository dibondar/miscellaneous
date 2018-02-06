from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('fast_evaluate', ['fast_evaluate.pyx'])
]

setup(
    name='Sample app',
    cmdclass={'build_ext':build_ext},
    ext_modules=ext_modules,
)
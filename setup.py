from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

cython_modules = ['research_utils/fast_pga.pyx']

setup(name='research_utils',
      version='0.1',
      description='Various utilities for my research.',
      author='Patrick Snape',
      author_email='p.snape@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(cython_modules, quiet=True),
      packages=find_packages()
)

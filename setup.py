#!/usr/bin/env python

from __future__ import print_function
from os.path import join
from numpy import get_include
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


def get_version():
    """Return the version of pyuvwsim"""
    context = {}
    version_file = join('pyuvwsim', 'version.py')
    try:
        execfile
    except NameError:
        exec(open(version_file).read(), context)
    else:
        execfile(version_file, context)
    return context['__version__']


numpy_inc = get_include()
pyuvwsim_extension = Extension('pyuvwsim._pyuvwsim',
                               sources=[join('pyuvwsim', 'src', 'pyuvwsim.c'),
                                        join('src', 'uvwsim.c')],
                               include_dirs=[numpy_inc, 'src',
                                             join('pyuvwsim', 'src')],
                               language='c')

setup(
    name='pyuvwsim',
    version=get_version(),
    description="A simple python API for generating interferometric baseline "
                "(uvw) coordinates",
    packages=['pyuvwsim'],
    ext_modules=[pyuvwsim_extension],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: Python :: 2.7',
    ],
    author='Oxford University e-Research Centre',
    author_email='benjamin.mort@oerc.ox.ac.uk',
    url='https://github.com/SKA-ScienceDataProcessor/uvwsim',
    license='BSD',
    install_requires=['numpy']
)

#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from os import path
from os.path import join

from numpy import get_include
numpy_inc = get_include()

def get_version():
    context = {}
    version_file = join('pyuvwsim','version.py')
    try:
        execfile
    except NameError:
        exec(open(version_file).read(), context)
    else:
        execfile(version_file, context)
    return context['__version__']

_pyuvwsim = Extension('pyuvwsim._pyuvwsim',
        sources=[join('pyuvwsim','src','pyuvwsim.c'), join('src','uvwsim.c')],
        include_dirs=[numpy_inc, 'src', join('pyuvwsim','src')],
        language='c')

extension_list = [_pyuvwsim]

setup(
    name='pyuvwsim',
    version=get_version(),
    description="A simple python API for generating interferometric baseline "
        "(uvw) coordinates",
    packages=['pyuvwsim'],
    ext_modules=extension_list,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    author='',
    author_email='',
    url='https://github.com/SKA-ScienceDataProcessor/uvwsim',
    license='BSD',
    install_requires=['numpy']
)

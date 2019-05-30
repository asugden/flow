#!/usr/bin/env python

from setuptools import setup, Extension
from glob import glob
import os
import platform

# Specify specific compiler on Mac
# if platform.system() == 'Darwin':
#     os.environ["CC"] = "gcc-mp-5"

setup_requires = []

scripts = []
# scripts.extend(glob('scripts/*py'))

# --- Encapsulate NumPy imports in a specialized Extension type ---------------

# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(Extension, object):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        super(NumpyExtension, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs

aaode = NumpyExtension(
    'aaode', sources=['flow/dep/analogaode.c'],
    extra_compile_args=['-fopenmp'], extra_link_args=['-lgomp'])
anb = NumpyExtension(
    'anb', sources=['flow/dep/analognaivebayes.c'],
    extra_compile_args=['-fopenmp'], extra_link_args=['-lgomp'])
runclassifier = NumpyExtension(
    'runclassifier', sources=['flow/dep/runclassifier.c'],
    extra_compile_args=['-fopenmp'], extra_link_args=['-lgomp'])

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="flow",
    version="0.0.1",
    packages=['flow',
              'flow.dep',
              ],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.8',
        'matplotlib>=3.0.2',
        'scipy>=0.19.0',
        'pandas>=0.21.1',
        'seaborn',
        'pyyaml',
        # 'scikit-image>=0.9.3',
        # 'shapely>=1.2.14',
        # 'scikit-learn>=0.11',
        # 'pillow>=2.6.1',
        'jsonschema',
        'future>=0.14',
        'six',
    ],
    scripts=scripts,
    # package_data={
    #     'replay': [
    #         'tests/*.py',
    #         'tests/data/example.tif',
    #         'tests/data/example.h5',
    #         'tests/data/example-volume.h5',
    #         'tests/data/example-tiffs/*.tif',
    #     ]
    # },
    #
    # metadata for upload to PyPI
    author="Arthur Sugden",
    author_email="arthur.sugden@gmail.com",
    description="Andermann Lab cortical reactivation analysis",
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience behavior",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    setup_requires=setup_requires,
    # setup_requires=['setuptools_cython'],
    url="https://www.andermannlab.com/",
    platforms=["Linux", "Mac OS-X", "Windows"],
    ext_modules=[anb, aaode, runclassifier],
    #
    # could also include long_description, download_url, etc.
)


# GLMs require the following R packages:
# lme4, afex
# To install, run R, and type:
# install.package("lme4", dependencies=TRUE)
# install.package("afex", dependencies=TRUE)

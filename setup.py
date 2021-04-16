# Installation script for python
from setuptools import setup, find_packages
from sys import version_info
import os
import re


requirements = ['numpy', 'pyyaml']
if version_info.major >=3 and version_info.minor >= 9:
    # For python above 3.9 the only existing TF is 2.5 which works well (even pre releases)
    tf_pack = "tensorflow"
else:
    tf_pack = "tensorflow>2.1"
requirements.append(tf_pack)

PACKAGE = 'pdfflow'

def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join('src', PACKAGE, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name='pdfflow',
      version=get_version(),
      description='PDF interpolation with Tensorflow',
      author = 'S.Carrazza, J.Cruz-Martinez, M.Rossi',
      author_email='stefano.carrazza@cern.ch, juan.cruz@mi.infn.it, marco.rossi5@unimi.it',
      url='https://github.com/N3PDF/pdfflow',
      package_dir={'':'src'},
      packages=find_packages('src'),
      zip_safe=False,
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      install_requires=requirements,
      extras_require={
          'capi' : [
            'cffi',
            ],
          'docs' : [
            'sphinx_rtd_theme',
            'recommonmark',
            'sphinxcontrib-bibtex',
            ],
          'examples' : [
            'matplotlib',
            'vegasflow',
            ],
          },
      python_requires='>=3.6',
      long_description=long_description,
      long_description_content_type="text/markdown",
)

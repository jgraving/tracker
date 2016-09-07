#! /usr/bin/env python
#
# Copyright (C) 2015-2016 Jacob Graving <jgraving@gmail.com>

import os
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

DESCRIPTION = "POSE: behavioral tracking using Python and OpenCV"
LONG_DESCRIPTION = """\
POSE (Python Open Source Ethology) is a Python library for tracking animal behavior of individual animals. 
The library uses uses OpenCV to automatically track the location of an animal in a video. 
It provides a high-level API for the automated analysis of animal behavior and locomotion.
"""

DISTNAME = 'pose'
MAINTAINER = 'Jacob Graving'
MAINTAINER_EMAIL = 'jgraving@gmail.com'
URL = 'http://jakegraving.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/jgraving/pose.git'
VERSION = '0.1.0'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    # Make sure dependencies exist

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import cv2
    except ImportError:
        install_requires.append('cv2')

    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=install_requires,
        packages=['pose'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Scientific/Engineering :: Image Recognition',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                     'Topic :: Multimedia :: Video'
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
          )

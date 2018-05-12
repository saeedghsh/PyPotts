#!/usr/bin/env python

from distutils.core import setup
import codecs
import os

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='PyPotts',
      version='0.1', # major.minor[.patch[.sub]].
      description='Neural Network based on Potts Spin, for task assignment',
      long_description=long_description,
      author='Saeed Gholami Shahbandi',
      author_email='http://www.google.com/recaptcha/mailhide/d?k=01tE3fdtc5PWagBP5AN3hInQ==&c=1YjiecfUTeq36sfpBz22wA==',
      maintainer='Saeed Gholami Shahbandi',
      maintainer_email='http://www.google.com/recaptcha/mailhide/d?k=01tE3fdtc5PWagBP5AN3hInQ==&c=1YjiecfUTeq36sfpBz22wA==',
      url='https://github.com/saeedghsh/PyPotts',
      packages=['PyPotts',],
      keywords='task assignment ',
      license='GPL'
     )

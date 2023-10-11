#!/usr/bin/env python3


from setuptools import setup, find_packages

setup(name='testdata',
      version='0.2',
      description='Test Data Management Library',
      author='Zhen NI',
      author_email='z.ni@hotmail.com',
      packages=find_packages(exclude=['test'])
      )

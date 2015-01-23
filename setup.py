#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='langchangetrack',
    version='0.1.0',
    description='Package for statistically significant language change.',
    long_description=readme + '\n\n' + history,
    author='Vivek Kulkarni',
    author_email='viveksck@gmail.com',
    url='https://github.com/viveksck/langchangetrack',
    packages=[
        'langchangetrack',
        'langchangetrack.utils',
        'langchangetrack.corpusreaders',
        'langchangetrack.tsconstruction',
        'langchangetrack.tsconstruction.distributional'
    ],
    package_dir={'langchangetrack':
                 'langchangetrack'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='langchangetrack',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    scripts=[
        'langchangetrack/tsconstruction/freq/scripts/create_freq_timeseries.py',
        'langchangetrack/tsconstruction/syntactic/scripts/pos_displacements.py',
        'langchangetrack/tsconstruction/distributional/scripts/train_embeddings_ngrams.py',
        'langchangetrack/tsconstruction/distributional/scripts/learn_map.py',
        'langchangetrack/tsconstruction/distributional/scripts/embedding_displacements.py',
        'langchangetrack/tsconstruction/dump_timeseries.py',
    ],
    test_suite='tests',
    tests_require=test_requirements
)

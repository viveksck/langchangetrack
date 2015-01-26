#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    'wheel==0.23.0',
    'argparse>=1.2.1',
    'numpy>=0.9.1',
    'scipy>=0.15.1',
    'more_itertools>=2.2',
    'joblib>=0.8.3-r1',
    'gensim==0.10.3',
    'statsmodels>=0.5.0',
    'changepoint>=0.1.0',
    'nltk>=3.0.0',
    'textblob>=0.9.0',
    'textblob-aptagger>=0.2.0',
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
        'langchangetrack/cpdetection/detect_changepoints_word_ts.py',
        'langchangetrack/scripts/detect_cp_freq.sh',
        'langchangetrack/scripts/detect_cp_pos.sh',
        'langchangetrack/scripts/detect_cp_distributional.sh',
        'langchangetrack/utils/scripts/freq_count.py',
        'langchangetrack/utils/scripts/common_vocab.py',
    ],
    test_suite='tests',
    tests_require=test_requirements
)

===============================
langchangetrack
===============================

.. image:: https://badge.fury.io/py/langchangetrack.png
    :target: http://badge.fury.io/py/langchangetrack

.. image:: https://travis-ci.org/viveksck/langchangetrack.png?branch=master
        :target: https://travis-ci.org/viveksck/langchangetrack

.. image:: https://pypip.in/d/langchangetrack/badge.png
        :target: https://pypi.python.org/pypi/langchangetrack


Package for Statistically Significant Language Change.

* Free software: BSD license
* Documentation: https://langchangetrack.readthedocs.org.

Features
--------

* This package provides tools to detect linguistic change in temporal corpora. 

* The meta algorithm works in 2 main steps

    #. **Time series construction**:Given a word, we construct a time series that tracks the displacement of a word through time. We track the displacement of a word using either Frequency, Part of Speech Distribution or Co-occurrences.

    #. **Change point detection**: We then use change point detection methods to detect if the time series contains a change point and if so what the change point is.

The details of the above steps are outlined in : http://arxiv.org/abs/1411.3315

Usage
------

    cd examples/distributional/
    ./ngrams_pipeline.sh ../data/temporal_corpus/ ./working/ ./output/ 1900 2000 5 locallinear 100 1000 "ngrams" ../data/temporal_corpus/common_vocab.txt 10

    cd examples/pos
    ./pos_pipeline.sh ../data/temporal_corpus/ ./working/ ./output/ 1900 1930 5 ../data/temporal_corpus/common_vocab.txt 10

    cd examples/freq
    ./freq_pipeline.sh ../data/temporal_corpus/ ./working/ ./output/ 1900 2000 5 ../data/temporal_corpus/common_vocab.txt 10

Requirements
------------
* wheel==0.23.0
* argparse>=1.2.1
* numpy>=0.9.1
* scipy>=0.15.1
* more_itertools>=2.2
* joblib>=0.8.3-r1
* gensim==0.10.3
* statsmodels>=0.5.0
* changepoint>=0.1.0
* nltk>=3.0.0
* textblob>=0.9.0
* textblob-aptagger>=0.2.0
* psutil>=2.2.0
* GNU Parallel
* R (good to have)
* rpy2 (good to have)



Installation
------------
#. Install GNU Parallel from here:  www.gnu.org/software/software.html
#. cd langchangetrack
#. pip install -r requirements.txt 
#. python setup.py install


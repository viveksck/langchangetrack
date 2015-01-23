#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import os
from os import path
import cPickle as pickle
import numpy as np
import scipy
import itertools
from scipy.spatial.distance import cosine, euclidean, norm
import pandas as pd
import more_itertools
from joblib import Parallel, delayed

from langchangetrack.utils.dummy_regressor import DummyRegressor 
from langchangetrack.utils.LocalLinearRegression import *
import gensim

import logging
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
logger = logging.getLogger("langchangetrack")

os.system("taskset -p 0xffff %d" % os.getpid())

def normalize_vector(vec):
    """ Normalize a vector by its L2 norm. """
    norm = (vec ** 2).sum() ** 0.5 
    return (vec / norm)

def pairwise(iterable):
    """ [a,b,c,d]=>[(a,b), (b,c), (c, d)] """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def process_word_source(w, eobj):
    """ Calculate displacements of word for source timepoint tuples. """
    return eobj.process_word(w, 0)

def process_word_dest(w, eobj):
    """ Calculate displacements of word for destination timepoint tuples."""
    return eobj.process_word(w, 1)

def process_chunk(chunk, func, *args):
    """ Apply a function on each element of a iterable. """
    L = []
    for i, e in enumerate(chunk):
        L.append(func(e, *args))
        if i % 10 == 0:
            print "Processing chunk", i
    return L

class Displacements(object):
    def __init__(self):
        """ Constructor """
        self.get_vectors = None
        self.load_model  = None
        self.models = {}
        self.has_predictors =  False
        self.load_predictor = None
        self.predictors = {}
        self.norm_embedding = False
        self.words_file = None
        self.timepoints = None
        self.data_dir = None
        self.pred_dir = None
        self.num_words = -1
        self.method = None
        self.win_size = -1
        self.fixed_point = -1
        self.embedding_suffix = None
        self.predictor_suffix = None

    def get_word_list(self):
        """ Returns a list of words for which time series needs to be generated. 
        """

        words_list = open(self.words_file, 'r').read().split('\n')
        if words_list[-1] == '': 
            words_list = words_list[:-1]
        if self.num_words != -1:
            return words_list[:num_words]
        else:
            return words_list

    def get_tuples(self, word, timepoint1, timepoint2):
        """ Return what time point pairs we must consider fot the word. """
        return [(word, timepoint1, word, timepoint2)]

    def generate_displacement_word(self, word, timepoints):
        L = []

        for ot, nt in timepoints:
            modelo = self.get_predictor(ot)
            modeln = self.get_predictor(nt)
            tuples = self.get_tuples(word, ot, nt)

            for tup in tuples:
                word1 = tup[0]
                timepoint1 = tup[1]
                word2 = tup[2]
                timepoint2 = tup[3]
            
                if self.is_present(timepoint1, word1) and self.is_present(timepoint2, word2):
                    vec1 = self.get_vector(timepoint1, word1)
                    vec2 = self.get_vector(timepoint2, word2)

                    if self.norm_embedding:
                        assert(np.isclose(norm(vec1),1.0))
                        assert(np.isclose(norm(vec2),1.0))

                    vec1_pred = modelo.predict(vec1)
                    vec2_pred = modeln.predict(vec2)

                    if self.norm_embedding:
                        vec1_pred = normalize_vector(vec1_pred)
                        vec2_pred = normalize_vector(vec2_pred)
                        assert(np.isclose(norm(vec1),1.0))
                        assert(np.isclose(norm(vec2),1.0))
 
                    d = self.calculate_distance(vec1_pred, vec2_pred)
                    assert(len(d) == self.number_distance_metrics())
                    L.append([word1, timepoint1, word2, timepoint2] + d)
                else:
                    #Word is not present in both time periods
                    L.append([word1, timepoint1, word2, timepoint2] + itertools.repeat(np.nan, self.number_distance_metrics()))
        return L

    def get_timepoints_word(self, w, timepoints):
        """ Get the list of timepoints to be considered for a word. """
        for i, t in enumerate(timepoints):
            if self.is_present(t, w):
                break
        # We have foind the first instance of the word at this time point,
        timepoints_considered = timepoints[i:]

        # Create the tuples for calculating displacements based on strategy
        # used.
        if self.method == "polar":
            timepoints1 = zip(timepoints_considered, list(itertools.repeat(timepoints_considered[0], len(timepoints_considered))))
            timepoints2 = zip(timepoints_considered, list(itertools.repeat(timepoints_considered[-1],len(timepoints_considered))))
        elif self.method == 'win':
            timepoints1 = zip(timepoints_considered[win_size:], timepoints_considered[:-win_size])
            timepoints2 = zip(timepoints_considered[:-win_size],timepoints_considered[win_size:])
        elif self.method == 'fixed':
            timepoints1 = zip(timepoints_considered, list(itertools.repeat(fixed_point, len(timepoints_considered))))
            timepoints2 = zip(timepoints_considered, list(itertools.repeat(timepoints_considered[-1],len(timepoints_considered))))

        # Return the list if tuples
        return timepoints1, timepoints2

    def process_word(self, w, index):
        """ Calculate displacements of the word at each timepoint tuple.
            index: Are we using timepoints1 or timepoints2.
        """
        t = self.get_timepoints_word(w, self.timepoints)
        return self.generate_displacement_word(w, t[index])

    def calculate_words_displacement(self, column_names):
        """ Calculate word displacements for each word in the Pandas data frame. """

        words = self.get_word_list()
        # Create chunks of the words to be processed.
        chunks = list(more_itertools.chunked(words, 100))

        # Calculate the displacements
        chunksL = Parallel(n_jobs=2, verbose=20)(delayed(process_chunk)(chunk, process_word_source, self) for chunk in chunks)
        chunksH = Parallel(n_jobs=2, verbose=20)(delayed(process_chunk)(chunk, process_word_dest, self) for chunk in chunks)
        L = more_itertools.flatten(chunksL)
        H = more_itertools.flatten(chunksH)
        flattendL = [x for sublist in L for x in sublist]
        flattendH = [x for sublist in H for x in sublist]

        # Store the results in a nice pandas data frame
        dfo, dfn = self.create_data_frames(flattendL, flattendH, column_names)
        return flattendL, flattendH, dfo, dfn

    def create_data_frames(self, L, H, column_names):
        """ Store the displacement of each word for the pair of timepoints in a
            nice Pandas data frame. """
        dfo = pd.DataFrame()
        dfo = dfo.from_records(L, columns=column_names)
        dfo_clean =dfo.fillna(method='ffill')
        dfn = pd.DataFrame()
        dfn = dfn.from_records(H, columns=column_names)
        dfn_clean = dfn.fillna(method='bfill')
        return dfo_clean, dfn_clean

    def get_model(self, timepoint):
        """ Return the model corresponding to this timepoint. """
        return self.models[timepoint]

    def get_predictor(self, timepoint):
        """ Return the predictor corresponding to this timepoint. """
        return self.predictors[timepoint]

    def number_distance_metrics(self):
        """ The number of distance metrics evaluated by calculate_distance.  """
        raise NotImplementedError, "Pure virtual function"

    def calculate_distance(self, vec1, vec2):
        """ Calculate distances between vector1 and vector2. """
        raise NotImplementedError, "Pure virtual function"

    def load_models_and_predictors(self):
        raise NotImplementedError, "Pure virtual function"

    def is_present(self, timepoint, word):
        """ Check if the word is present in the vocabulary at this timepoint. """ 
        raise NotImplementedError, "Pure virtual function"

    def get_vector(self, timepoint, word):
        """ Get the embedding for this word at the specified timepoint."""
        raise NotImplementedError, "Pure virtual function"


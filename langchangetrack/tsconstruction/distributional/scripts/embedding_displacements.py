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

import langchangetrack
from langchangetrack.utils.dummy_regressor import DummyRegressor
import gensim

from langchangetrack.tsconstruction.displacements import Displacements

import logging
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
logger = logging.getLogger("langchangetrack")

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
p.set_cpu_affinity(list(range(cpu_count())))


def uniform(distances):
    return np.ones(len(distances))


def get_vectors_sg(model, norm_embedding=True):
    """ Return the embeddings of  a skipgram model. """
    if norm_embedding:
        return model.syn0norm
    else:
        return model.syn0


def load_model_skipgram(model_path):
    """ Load the skipgram model from a file in word2vec format. """
    return gensim.models.Word2Vec.load_word2vec_format(model_path)


def load_predictor_skipgram(predictor_path):
    """ Load the predictor model. """
    return pickle.load(open(predictor_path))


class EmbeddingsDisplacements(Displacements):

    def __init__(self,
                 data_dir,
                 pred_dir,
                 words_file,
                 timepoints,
                 num_words,
                 get_vectors,
                 load_model,
                 load_predictor,
                 method,
                 win_size,
                 fixed_point,
                 embedding_suffix,
                 predictor_suffix,
                 workers):
        """ Constructor """
        # Initialize the super class.
        super(EmbeddingsDisplacements, self).__init__()
        self.get_vectors = get_vectors
        self.load_model = load_model
        self.has_predictors = True
        self.load_predictor = load_predictor
        self.norm_embedding = True
        self.words_file = words_file
        self.timepoints = timepoints
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.num_words = num_words
        self.method = method
        self.win_size = win_size
        self.fixed_point = fixed_point
        self.embedding_suffix = embedding_suffix
        self.predictor_suffix = predictor_suffix
        self.workers = workers

    def number_distance_metrics(self):
        return 2

    def calculate_distance(self, vec1, vec2):
        """ Calculate distances between vector1 and vector2. """
        return [cosine(vec1, vec2), euclidean(vec1, vec2)]

    def load_models_and_predictors(self):
        """ Load all the models and predictors. """
        self.models = {}
        self.predictors = {}
        model_paths = [path.join(self.data_dir, timepoint + '_embeddings' + self.embedding_suffix) for timepoint in self.timepoints]
        predictor_handles = [path.join(self.pred_dir, timepoint + '_embeddings' + self.predictor_suffix) for timepoint in self.timepoints]
        loaded_models = Parallel(n_jobs=self.workers)(delayed(self.load_model)(model_path) for model_path in model_paths)
        for i, timepoint in enumerate(self.timepoints):
            self.models[timepoint] = loaded_models[i]
            self.predictors[timepoint] = self.load_predictor(predictor_handles[i])
            if hasattr(self.predictors[timepoint], 'weight_func'):
                self.predictors[timepoint].weight_func = uniform
                print "Loaded predictor for", timepoint
        print "Done loading predictors"

    def is_present(self, timepoint, word):
        """ Check if the word is present in the vocabulary at this timepoint. """
        model = self.get_model(timepoint)
        return word in model.vocab

    def get_vector(self, timepoint, word):
        """ Get the embedding for this word at the specified timepoint."""
        model = self.get_model(timepoint)
        return self.get_vectors(model, self.norm_embedding)[model.vocab[word].index]


def main(args):
    syear = int(args.syear)
    eyear = int(args.eyear)
    stepsize = int(args.stepsize)
    timepoints = np.arange(syear, eyear, stepsize)
    timepoints = [str(t) for t in timepoints]
    workers = int(args.workers)
    # Create the main work horse.
    e = EmbeddingsDisplacements(args.datadir,
                                args.preddir,
                                args.filename,
                                timepoints,
                                int(args.num_words),
                                get_vectors_sg,
                                load_model_skipgram,
                                load_predictor_skipgram,
                                args.method,
                                args.win_size,
                                str(args.fixed_point),
                                args.embedding_suffix,
                                args.predictor_suffix,
                                workers)

    # Load the models and predictors
    e.load_models_and_predictors()

    # Calculate the word displacements and dump.
    L, H, dfo, dfn = e.calculate_words_displacement(column_names=['word', 's', 'otherword', 't', 'cosine', 'euclidean'], n_jobs=workers)
    fname = 'timeseries_s_t' + '_' + args.outputsuffix + '.pkl'
    pickle.dump((L, H, dfo, dfn), open(path.join(args.outputdir, fname), 'wb'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="Input file for words")
    parser.add_argument("-d", "--data_dir", dest="datadir", help="data directory")
    parser.add_argument("-p", "--pred_dir", dest="preddir", help="data directory")
    parser.add_argument("-o", "--output_dir", dest="outputdir", help="Output directory")
    parser.add_argument("-os", "--output_suffix", dest="outputsuffix", help="Output suffix")
    parser.add_argument("-es", "--emb_suffix", dest="embedding_suffix", help="embedding suffix")
    parser.add_argument("-ps", "--pred_suffix", dest="predictor_suffix", help="predictor suffix")
    parser.add_argument("-sy", "--start", dest="syear", default='1800', help="start year")
    parser.add_argument("-ey", "--end", dest="eyear", default='2010', help="end year(not included)")
    parser.add_argument("-s", "--window_size", dest="stepsize", default=5, help="Window size for time series")
    parser.add_argument("-e", "--embedding_type", dest="embedding_type", default='skipgram',  help="Embedding type")
    parser.add_argument("-m", "--method", dest="method", default="polar", help="Method to use")
    parser.add_argument("-w", "--win_size", dest="win_size", default="-1", help="Window size to use if not polar", type=int)
    parser.add_argument("-y", "--fixed_point", dest="fixed_point", default="-1", help="fixed point to use if method is fixed", type=int)
    parser.add_argument("-n", "--num_words", dest="num_words", default=-1, help="Number of words", type=int)
    parser.add_argument("-workers", "--workers", dest="workers", default=1, help="Maximum number of workers", type=int)
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    args = parser.parse_args()
    main(args)

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

from dummy_regressor import *
from LocalLinearRegression import *
import gensim

import logging
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
logger = logging.getLogger("langchangetrack")

os.system("taskset -p 0xffff %d" % os.getpid())

def normalize_embedding(vec):
    """ Normalize a vector by its L2 norm. """
    norm = (vec ** 2).sum() ** 0.5 
    return (vec / norm)

def get_vectors_sg(model, norm_embedding=True):
    """ Return the embeddings of  a skipgram model. """
    if norm_embedding:
        return model.syn0norm
    else:
        return model.syn0

def load_model_skipgram(model_path):
    """ Load the skipgram model from a file in word2vec format. """
    return gensim.models.Word2Vec.load_word2vec_format(model_path)

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

class EmbeddingsDisplacements(object):
    def __init__(self):
        """ Constructor """
        self.get_vectors = None
        self.load_model  = None
        self.models = {}
        self.predictors = {}
        self.norm_embedding = True
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

    def calculate_distance(self, vec1, vec2):
        """ Calculate distances between vector1 and vector2. """
        return [cosine(vec1, vec2), euclidean(vec1,vec2)]

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
                        vec1_pred = normalize_embedding(vec1_pred)
                        vec2_pred = normalize_embedding(vec2_pred)
                        assert(np.isclose(norm(vec1),1.0))
                        assert(np.isclose(norm(vec2),1.0))
 
                    d = self.calculate_distance(vec1_pred, vec2_pred)
                    L.append([word1, timepoint1, word2, timepoint2] + d)
                else:
                    #Word is not present in both time periods
                    L.append([word1, timepoint1, word2, timepoint2] + [np.nan, np.nan])
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

    def calculate_words_displacement(self):
        """ Calculate word displacements for each word in the Pandas data frame. """

        columns=['word', 's', 'nword', 't', 'cosine', 'euclidean']
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
        dfo, dfn = self.create_data_frames(flattendL, flattendH)
        return flattendL, flattendH, dfo, dfn

    def create_data_frames(self, L, H):
        """ Store the displacement of each word for the pair of timepoints in a
            nice Pandas data frame. """
        dfo = pd.DataFrame()
        dfo = dfo.from_records(L, columns=['w', 's', 'ow', 't', 'cosine', 'euclidean'])
        dfo_clean =dfo.fillna(method='ffill')
        dfn = pd.DataFrame()
        dfn = dfn.from_records(H, columns=['w', 's', 'ow', 't', 'cosine', 'euclidean'])
        dfn_clean = dfn.fillna(method='bfill')
        return dfo_clean, dfn_clean


    def load_models_and_predictors(self):
        """ Load all the models and predictors. """
        self.models = {}
        self.predictors = {}
        model_paths = [path.join(self.data_dir, timepoint+'_embeddings' + self.embedding_suffix) for timepoint in self.timepoints]
        predictor_handles = [path.join(self.pred_dir, timepoint + '_embeddings' + self.predictor_suffix) for timepoint in self.timepoints]
        loaded_models = Parallel(n_jobs=16)(delayed(self.load_model)(model_path) for model_path in model_paths)
        for i, timepoint in enumerate(self.timepoints):
            self.models[timepoint] = loaded_models[i]
            self.predictors[timepoint] = pickle.load(open(predictor_handles[i]))
            if hasattr(self.predictors[timepoint], 'weight_func'):
                self.predictors[timepoint].weight_func = KernelFunctions.uniform 
                print "Loaded predictor for", timepoint
        print "Done loading predictors"

    def get_model(self, timepoint):
        """ Return the model corresponding to this timepoint. """
        return self.models[timepoint]

    def get_predictor(self, timepoint):
        """ Return the predictor corresponding to this timepoint. """
        return self.predictors[timepoint]

    def is_present(self, timepoint, word):
        """ Check if the word is present in the vocabulary at this timepoint. """ 
        model = self.get_model(timepoint)
        return word in model.vocab

    def get_vector(self, timepoint, word):
        """ Get the embedding for this word at the specified timepoint."""
        model = self.get_model(timepoint)
        return self.get_vectors(model)[model.vocab[word].index]

def main(args):
    # Create the main work horse.
    e = EmbeddingsDisplacements() 

    # Set it up
    e.data_dir = args.datadir
    e.pred_dir = args.preddir
    e.words_file = args.filename
    e.output_dir = args.outputdir

    syear = int(args.syear)
    eyear = int(args.eyear)
    stepsize = int(args.stepsize)
    timepoints = np.arange(syear, eyear, stepsize)
    timepoints = [str(t) for t in timepoints]
    e.timepoints = timepoints

    e.num_words = int(args.num_words)
    e.get_vectors = get_vectors_sg
    e.load_model = load_model_skipgram

    e.method = args.method
    e.win_size = args.win_size
    e.fixed_point = str(args.fixed_point)

    e.embedding_suffix = args.embedding_suffix
    e.predictor_suffix = args.predictor_suffix

    # Load the models and predictors
    e.load_models_and_predictors()

    # Calculate the word displacements and dump.
    L, H, dfo, dfn = e.calculate_words_displacement()
    fname = 'timeseries_s_t' + '_' + args.outputsuffix + '.pkl'
    pickle.dump((L,H, dfo, dfn), open(path.join(args.outputdir, fname),'wb'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="Input file for words")
    parser.add_argument("-d", "--data_dir", dest="datadir", help="data directory")
    parser.add_argument("-p", "--pred_dir", dest="preddir", help="data directory")
    parser.add_argument("-o", "--output_dir", dest="outputdir", help="Output directory")
    parser.add_argument("-os", "--output_suffix", dest="outputsuffix", help="Output suffix")
    parser.add_argument("-es", "--emb_suffix", dest="embedding_suffix", help="embedding suffix")
    parser.add_argument("-ps", "--pred_suffix", dest="predictor_suffix",help="predictor suffix")
    parser.add_argument("-sy", "--start", dest="syear", default = '1800', help="start year")
    parser.add_argument("-ey", "--end", dest="eyear", default = '2010', help="end year(not included)")
    parser.add_argument("-s", "--window_size", dest="stepsize", default = 5, help="Window size for time series")
    parser.add_argument("-e", "--embedding_type", dest="embedding_type", default = 'skipgram',  help="Embedding type")
    parser.add_argument("-m", "--method", dest="method", default="polar", help="Method to use")
    parser.add_argument("-w", "--win_size", dest="win_size", default="-1", help="Window size to use if not polar", type=int)
    parser.add_argument("-y", "--fixed_point", dest="fixed_point", default="-1", help="fixed point to use if method is fixed", type=int)
    parser.add_argument("-n", "--num_words", dest="num_words", default = -1, help="Number of words", type=int)
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    args = parser.parse_args()
    main(args)

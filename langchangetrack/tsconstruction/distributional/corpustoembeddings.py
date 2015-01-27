#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import gensim

import logging
logger = logging.getLogger("langchangetrack")


class CorpusToEmbeddings(object):

    """ Class that encapsulates functionality for obtaining embeddings from a corpus."""

    def __init__(self, corpus_iter, embeddings_type, lang='en',
                 model_config={}, save_model_file=None):
        """ Initialize the object with the corpus iterator and 
            the type of embeddings.
        
            The corpus iterator should just support iterating over 
            sentences. It can be a list or a generator which yields
            sentences. The embeddings type can be one of the supported 
            embedding types: 'skipgram'
            
            The model_config is an optional named tuple containing specific 
            configurations parameters to be passed when training the model.
        """

        assert(corpus_iter)
        assert(embeddings_type in CorpusToEmbeddings.supported_embedding_types())

        self.corpus_iter = corpus_iter
        self.lang = lang
        self.embeddings_type = embeddings_type
        self.model_config = model_config

        self.embeddings_builder_map = {
            'skipgram': self.buildword2vec
        }
        self.model = None
        self.save_model_file = save_model_file
        return

    @staticmethod
    def supported_embedding_types():
        """ Embedding types we support. """
        return ['skipgram']

    def buildword2vec(self):
        """ Trains a word2vec model on the corpus. """

        cfg_size = self.model_config.get('size', 200)
        cfg_window = self.model_config.get('window', 5)
        cfg_min_count = self.model_config.get('min_count', 10)
        cfg_workers = self.model_config.get('workers', 16)
        cfg_alpha = self.model_config.get('alpha', 0.01)
        logger.info('window size:{}, alpha:{}, embedding size:{}, min_count:{}, workers:{}'.format(cfg_window, cfg_alpha, cfg_size, cfg_min_count, cfg_workers))
        self.model = gensim.models.Word2Vec(self.corpus_iter,
                                            size=cfg_size,
                                            window=cfg_window,
                                            min_count=cfg_min_count,
                                            alpha=cfg_alpha,
                                            workers=cfg_workers,
                                            sample=1e-5,
                                            negative=0)

        if self.save_model_file:
            self.model.save_word2vec_format(self.save_model_file)

    def build(self):
        """ Trains a model on the corpus to obtain embeddings."""
        sys.stdout.write("Building a model from the corpus.\n")
        sys.stdout.flush()
        self.embeddings_builder_map[self.embeddings_type]()
        sys.stdout.write("Model built.\n")
        sys.stdout.flush()

    def save_model(self, model_file):
        """ Saves the model file. """
        self.model.save_word2vec_format(model_file)
        return

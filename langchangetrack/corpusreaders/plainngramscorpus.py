#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import gensim


class PlainNGRAMSCorpus(object):

    """Iterate over sentences(ngram) of plain ngram file"""

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
            text = open(self.filename)
            for sentence in text:
                yield gensim.utils.simple_preprocess(sentence, deacc=True)

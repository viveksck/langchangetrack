#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Benchmark for the quality of the joint space"""

from argparse import ArgumentParser
import logging
import sys
from io import open
import os
from os import path
from time import time
from glob import glob
from collections import defaultdict
from copy import deepcopy
from random import shuffle
import json
import cPickle as pickle

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import numpy
from numpy import asarray
from LocalLinearRegression import LocalLinearRegression

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

reg_model = None
K_NN = 1000

class Mapping(object):
  """ Mapping between terms/phrases."""
  def __init__(self, source=None, target=None):
    self.s_lang = source
    self.t_lang = target
    self.map = None

class IdentityTranslations(Mapping):
  def __init__(self, source, target, se, te):
    super(IdentityTranslations, self).__init__(source, target)
    words = set(se.word_id.keys()) & set(te.word_id.keys())
    D = {}
    for word in words:
      D[word] = word
    self.map = D

class Embeddings(object):
  """ A list of words and their vector representatoins.

      We assume that the given words are sorted by their frequency.
  """

  def __init__(self, lang, filename=None, vectors=None, words=None):

    self.lang = lang
    if filename:
      self.filename = filename
      self.read_file()

    if vectors != None:
      self.vectors = asarray(vectors)
    if words:
      if len(set(words)) == len(words):
        self.word_id = {w:i for i,w in enumerate(words)}
      else:
        logging.debug("We have duplicate words.")
        self.word_id = {u'{}_{}'.format(w, i):i for i,w in enumerate(words)}
    self.id_word = {i:w for w,i in self.word_id.iteritems()}
    self.words = [w for w,i in Embeddings.sorted_words(self.word_id)]

  def read_file(self):
    raise NotImplementedError("Implement an embeddings reader.")

  def get_vectors(self, words=None):
    if words:
      return asarray([self.vectors[self.word_id[w]] for w in words])
    return self.vectors

  def __most_frequent(self, n, start=0):
    return [x for x,y in sorted(self.word_id.iteritems(), key=lambda(x,y): y)[start:n]]

  def most_frequent(self, n, start=0):
    return Embeddings(lang=self.lang, words=self.words[start:n],
                      vectors=self.vectors[start:n])

  def least_frequent_n(self, n):
    return [x for x,y in sorted(self.word_id.iteritems(),
                                key=lambda(x,y): y, reverse=True)[:n]]

  def words_translations(self, other, mapping, segment):
    start, end = segment
    s_words = self.__most_frequent(n=end, start=start)

    map_ = mapping.map
    t_words = [map_[w] for w in s_words]
    exact = [(w1,w2) for (w1,w2) in zip(s_words, t_words) if w1.lower()==w2.lower()]
    logging.info("{} exact words translations in between {}-{} for "
                 "{}-{} languages.".format(len(exact), start, end, mapping.s_lang, mapping.t_lang))

    s_new_vectors = self.vectors[start:end]
    t_new_vectors = asarray([other.vectors[other.word_id[w]] for w in t_words])

    source = Embeddings(vectors=s_new_vectors, words=s_words, lang=self.lang)
    target = Embeddings(vectors=t_new_vectors, words=t_words, lang=other.lang)
    return (source, target)

  @staticmethod
  def sorted_words(word_id):
    return sorted(word_id.iteritems(), key=lambda(x,y): y)

  def get_common(self, other, mapping):
    """ Limit the two embeddings to the terms that are covered by the mapping."""

    self_oov = defaultdict(lambda: 0)
    other_oov = defaultdict(lambda: 0)
    self_word_id = deepcopy(self.word_id)
    other_word_id = deepcopy(other.word_id)
    new_words = []
    map_ = mapping.map
    for i, w in enumerate(self.word_id):
      if w not in map_:
        self_oov[w] += 1
        del self_word_id[w]
        continue

      if map_[w] not in other.word_id:
        other_oov[map_[w]] += 1
        del self_word_id[w]

    for i, w in enumerate(other.word_id):
      if w not in map_:
        del other_word_id[w]

    logging.info("We could not find {} {} words in our dictionary.".format(
                   len(self_oov), self.lang))
    logging.info("We could not find {} {} words in our target words.".format(
                   len(other_oov), other.lang))
    logging.info("Our {} vocabulary has {} valid words.".format(
                   self.lang, len(self_word_id)))
    
    sorted_self_word_id = Embeddings.sorted_words(self_word_id)
    self_vectors = asarray([self.vectors[i] for w,i in sorted_self_word_id])
    self_words = [w for w,i in sorted_self_word_id]
    new_self = Embeddings(lang=self.lang, vectors=self_vectors, words=self_words)


    sorted_other_word_id = Embeddings.sorted_words(other_word_id)
    other_vectors = asarray([other.vectors[i] for w,i in sorted_other_word_id])
    other_words = [w for w,i in sorted_other_word_id]
    new_other = Embeddings(lang=self.lang, vectors=other_vectors, words=other_words)


    return (new_self, new_other)

  def split(self, mapping, ignore_exact=True):
    """ Generates two embeddings that cover the mapping terms.

        If we have a1: b1, a2: b2 mappings in an embeddings space where {a1, b1, 
        a2, b2} exists, we would like to generates two embeddings spaces one for
        {a1, a2} and another for {b1, b2}.

        Sometimes it is not desirable to include exact terms a3:a3 in the new 
        embeddings. Hence, you need to ignore the exact terms.
    """

    source_oov = defaultdict(lambda: 0)
    target_oov = defaultdict(lambda: 0)
    w_exact = defaultdict(lambda: 0)

    source_words = []
    target_words = []
    map_ = mapping.map
    for w, id_ in self.word_id.iteritems():
      if w not in map_:
        source_oov[w] += 1
        continue

      if map_[w] not in self.word_id:
        target_oov[map_[w]] += 1
        continue

      if w.lower() == map_[w].lower():
        w_exact[w] += 1
        if ignore_exact:
          continue

      source_words.append(w)
      target_words.append(map_[w])

    logging.debug("We could not find {} source words in our dictionary.".format(
                  len(source_oov)))
    logging.debug("We could not find {} target words in our target words.".format(
                  len(target_oov)))
    logging.debug("{} words are exact between languages".format(len(w_exact)))
    logging.debug("We found {} pairs of words valid for testing.".format(len(source_words)))

    new_s_vectors = asarray([self.vectors[self.word_id[w]] for w in source_words])
    source = Embeddings(vectors=new_s_vectors, words=source_words,
                        lang=mapping.s_lang)

    new_t_vectors = asarray([self.vectors[self.word_id[w]] for w in target_words])
    target = Embeddings(vectors=new_t_vectors, words=target_words,
                        lang=mapping.t_lang)
    new_mapping = Mapping(source=mapping.s_lang, target=mapping.t_lang)
    new_mapping.map = dict(zip(source.words, target.words))
    return (source, target, new_mapping)

  def common(self, other):
    """ Find common terms between languages.

        The post condition is that both embeddings vocabulary are in the same
        order.
    """

    common_words = []
    for word in self.word_id:
      if word in other.word_id:
        common_words.append(word)

    new_self_vectors = []
    new_other_vectors = []
    for word in common_words:
      new_self_vectors.append(self.vectors[self.word_id[word]])
      new_other_vectors.append(other.vectors[other.word_id[word]])

    new_self = Embeddings(vectors=asarray(new_self_vectors), words=common_words,
                          lang=self.lang)

    new_other = Embeddings(vectors=asarray(new_other_vectors), words=common_words,
                          lang=self.lang)

    return (new_self, new_other)


class Word2VecEmbeddings(Embeddings):
  """ Word2Vec embeddings reader."""

  def read_file(self, limit=-1):
    words = []
    embeddings = []
    with open(self.filename, 'rb') as f:
      words_number, size = [int(x) for x in f.readline().strip().split()][:2]
      for i, line in enumerate(f):
        try:
          ws = line.decode('utf-8').strip().split()
          words.append(' '.join(ws[:-size]))
          embeddings.append([float(x) for x in ws[-size:]])
          if i == limit:
            break
        except Exception, e:
          print "Exception", i
          print "Exception", line
    self.word_id = {w:i for i,w in enumerate(words)}
    self.vectors = asarray(embeddings)
    assert len(self.word_id) == self.vectors.shape[0]

class Evaluator(object):
  """ Evaluator of the alignment between two languages."""

  def __init__(self, source_embeddings, target_embeddings, metric='l2', k=5):
    self.metric = metric
    self.source_embeddings = source_embeddings
    self.target_embeddings = target_embeddings
    self.k = k
    self.row_normalize = True
    self.col_normalize = False

  @staticmethod
  def cosine_knn(vectors, point, k):
    distances = numpy.dot(vectors, point)
    indices = list(reversed(distances.argsort()))[:k]
    return distances[indices], [indices]

  def norm(self, vectors):
    out = vectors
    if self.row_normalize:
      norms = (vectors ** 2).sum(axis=1) ** 0.5
      out = (vectors.T / norms).T

    if self.col_normalize:
      norms  = (vectors ** 2).sum(axis=0) ** 0.5
      norms[norms==0] = 1
      out = vectors/norms
    return out

  def precision_at_k(self, test_pairs):
    if self.metric == 'cosine':
      return self.precision_at_k_cosine(test_pairs)
    return self.precision_at_k_l2(test_pairs)

  def precision_at_k_l2(self, test_pairs):
    t_knn = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', p=2)
    t_knn.fit(self.target_embeddings.vectors)

    right = 0
    index = 0
    for s, t in test_pairs:
      assert(s == t)
      point = self.source_embeddings.vectors[self.source_embeddings.word_id[s]]
      distances, indices = t_knn.kneighbors(point)

      t_words = [self.target_embeddings.id_word[i] for i in indices[0]]
      t = t.rsplit('_', 1)[0]
      t_words = [x.rsplit('_', 1)[0] for x in t_words]

      line = u"{: <20}{:<20}{:<50}".format(s, t , u' '.join(t_words))
      logging.debug(line.encode('utf-8'))
      if t in t_words:
        right += 1
      index = index + 1
    return right/float(len(test_pairs))

  def precision_at_k_cosine(self, test_pairs):
    s_vectors = self.norm(self.source_embeddings.vectors)
    t_vectors = self.norm(self.target_embeddings.vectors)

    right = 0
    for s, t in test_pairs:
      point = self.source_embeddings.vectors[self.source_embeddings.word_id[s]]
      distances, indices = Evaluator.cosine_knn(t_vectors, point, self.k)

      t_words = [self.target_embeddings.id_word[i] for i in indices[0]]

      t = t.rsplit('_', 1)[0]
      t_words = [x.rsplit('_', 1)[0] for x in t_words]

      line = u"{: <20}{:<20}{:<50}".format(s, t , u' '.join(t_words))
      logging.debug(line.encode('utf-8'))
      if t in t_words:
        right += 1
    return right/float(len(test_pairs))

  def evaluate(self, mapping, operation, training_segment, test_segment):

    (s_train, t_train) = self.source_embeddings.words_translations(self.target_embeddings, mapping, training_segment)
    (s_test, t_test) = self.source_embeddings.words_translations(self.target_embeddings, mapping, test_segment)
   
    s_train.vectors = self.norm(s_train.vectors)
    t_train.vectors = self.norm(t_train.vectors)
    s_test.vectors = self.norm(s_test.vectors)
    t_test.vectors = self.norm(t_test.vectors)

    if set(s_train.words).intersection(set(s_test.words)):
      print (u"Train and test words are overlapping")

    s_new, t_new = operation((s_train, t_train), (s_test, t_test))

    return None

def linear_regression(train_embeddings, test_embeddings):
    global reg_model
    s_embeddings, t_embeddings = train_embeddings
    s_test , t_test = test_embeddings

    reg = LinearRegression()
    reg.fit(s_embeddings.vectors, t_embeddings.vectors)
    pickle.dump(reg, open(reg_model, 'wb'))
    s = Embeddings(vectors=reg.predict(s_test.vectors),
                   words=s_test.words, lang=s_embeddings.lang)
    return s, t_test

def local_linear_regression(train_embeddings, test_embeddings):
    global reg_model
    print "Using local linear regression with k = ", K_NN
    s_embeddings, t_embeddings = train_embeddings
    s_test , t_test = test_embeddings
    reg = LocalLinearRegression(k_nn=K_NN)
    reg.fit(s_embeddings.vectors, t_embeddings.vectors)
    pickle.dump(reg, open(reg_model, 'wb'))
    return None,None

def identity(train_vectors, all_vectors):
  return all_vectors 

def evaluate_word2vec(sl, tl, source_file, target_file, method):
  print "Proceeding to load embeddings"
  s_ = Word2VecEmbeddings(lang=sl, filename=source_file)
  t_ = Word2VecEmbeddings(lang=tl, filename=target_file)
  print "Loaded word embeddings"
  mapping = IdentityTranslations(source=sl, target=tl, se=s_, te = t_)
  print "Mapping done"
  s, t = s_.get_common(t_, mapping)
  print "Common vocab done"
  evaluator = Evaluator(source_embeddings=s, target_embeddings=t, metric='l2')
  print "Evaluator constructed"
  assert(s.vectors.shape == t.vectors.shape)
  print "Evaluating"
  if method == 'linear':
    p1 = evaluator.evaluate(mapping, linear_regression, (0, s.vectors.shape[0]), (0, s.vectors.shape[0]))
  elif method == 'locallinear':
    p1 = evaluator.evaluate(mapping, local_linear_regression, (0, s.vectors.shape[0]), (0, s.vectors.shape[0]))

def main(args):
  global reg_model
  global K_NN
  reg_model = args.filename
  if args.method == 'linear': 
    evaluate_word2vec('old', 'new', args.old_model, args.new_model, 'linear')
  elif args.method == 'locallinear':
    K_NN = int(args.knn_val)
    evaluate_word2vec('old', 'new', args.old_model, args.new_model, 'locallinear')

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="filename", help="Input file")
  parser.add_argument("-o", "--old_model", dest="old_model", help="old model")
  parser.add_argument("-n", "--new_model", dest="new_model", help="new model")
  parser.add_argument("-k", "--knn", dest="knn_val", default=1000, type=int, help="K in KNN for local linear regression")
  parser.add_argument("-m", "--method", dest="method", help="method")
  parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                      default="INFO")
  args = parser.parse_args()
  if args.log == 'DEBUG':
    sys.excepthook = debug
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOGFORMAT)
  main(args)


import numpy
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import pickle

# Code to pickle a VW model
import copy_reg
from types import FunctionType, FileType, MethodType
def stub_pickler(obj):
    return stub_unpickler, ()
def stub_unpickler():
    return "STUB"

copy_reg.pickle(MethodType, stub_pickler, stub_unpickler)
copy_reg.pickle(FileType,   stub_pickler, stub_unpickler)
copy_reg.pickle(FunctionType, stub_pickler, stub_unpickler)


'''
    Given a list of numbers, produce a list of weights using the specified kernel
'''
class KernelFunctions:
    @staticmethod
    def uniform(distances):
        return numpy.ones(len(distances))
    
    @staticmethod
    def gauss(distances):
        dist_norm = distances / distances[len(distances) - 1]
        weights = [ math.exp(-dist * dist) for dist in dist_norm ]
        return weights
    
    @staticmethod
    def linear(distances):
        dist_norm = distances / distances[len(distances) - 1]
        weights = [ 1.0001 - dist for dist in dist_norm ]
        return weights
    
    @staticmethod
    def epanechnikov(distances):
        dist_norm = distances / distances[len(distances) - 1]
        weights = [ (3./4.)*(1.0001 - dist * dist) for dist in dist_norm ]
        return weights
    
    @staticmethod
    def tricube(distances):
        dist_norm = distances / distances[len(distances) - 1]
        weights = [ pow( (1.0001 - pow(dist, 3)), 3 ) for dist in dist_norm ]
        return weights

class LocalLinearRegression(BaseEstimator):
    
    def __init__(self, k_nn, weight_func=KernelFunctions.uniform):
        self.k_nn = k_nn
        self.weight_func = weight_func
        print self.k_nn, self.weight_func
    
    '''
        X: A list of points to transform
        Y: The corresponding target points
    '''
    def fit(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("len(X) != len(Y)")
        if len(X) < self.k_nn:
            raise ValueError("Not enough points for local linear regression for the specified number of neighbors (" +
                             str(len(X)) + " < " + str(self.k_nn) + ")")
        self.X = numpy.array(X)
        self.Y = numpy.array(Y)
        self.nn = NearestNeighbors(n_neighbors=self.k_nn, algorithm='ball_tree', p=2)
        self.nn.fit(self.X)
        print "Fit the model"
    
    '''
        X: The point to transform based on its neighbors 
    '''
    def predict(self, X):
        neighbors = self.nn.kneighbors(X)
        distances = neighbors[0][0] 
        neighbor_indices = neighbors[1][0]
        local_X = self.X.take(neighbor_indices, axis=0)
        local_Y = self.Y.take(neighbor_indices, axis=0)
        wls = sm.WLS(local_Y, local_X, weights=self.weight_func(distances)).fit()
        return wls.predict(X)
        

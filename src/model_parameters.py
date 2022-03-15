#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Parameter classes for the hyperbolic random graph model

Author: Béatrice Désy

Date : 03/01/2022
"""

from abc import ABC, abstractmethod
import pickle

class Parameters(ABC):
    @abstractmethod
    def specify(self, dictionary):
        pass

    def load_from_file(self, path_to_dict):
        with open(path_to_dict, 'rb') as file:
            dictionary = pickle.load(file)
        self.specify(dictionary)

    def save_to_file(self, path_for_dict):
        with open(path_for_dict, 'wb') as file:
            pickle.dump(vars(self), file)

class GlobalParameters(Parameters):
    def __init__(self):
        self.dimension = None
        self.N = None
        self.mu = None
        self.beta = None
        self.euclidean = None
        self.radius = None

    def specify(self, dictionary):
        self.N = dictionary['N']
        self.dimension = dictionary['dimension']
        self.beta = dictionary['beta']
        self.mu = dictionary['mu']
        self.radius = dictionary['radius']
        self.euclidean = dictionary['euclidean']
    
    def get_njitable(self):
        return (self.dimension, self.N, 
                self.mu, self.beta, self.radius, 
                self.euclidean)


class LocalParameters(Parameters):
    def __init__(self):
        self.nodes = None
        self.coordinates = None
        self.kappas = None
        self.target_degrees = None

    def specify(self, dictionary):
        self.nodes = dictionary['nodes']
        self.coordinates = dictionary['coordinates']
        self.kappas = dictionary['kappas']
        self.target_degrees = dictionary['target_degrees']

class OptimizationParameters(Parameters):
    def __init__(self):
        self.tol = None
        self.max_iterations = None
        self.perturbation = None
        self.verbose = None

    def specify(self, dictionary):
        self.tol = dictionary['tol']
        self.max_iterations = dictionary['max_iterations']
        self.perturbation = dictionary['perturbation']
        self.verbose = dictionary['verbose']


#tests
if __name__ == '__main__':
    from numba import njit
    test_dico = {'dimension': 1, 'N': 1000, 'mu': 0.01,
                 'beta': 3.5, 'euclidean': False, 'radius': 159.14}
    gp = GlobalParameters()
    gp.specify(test_dico)
    gp.save_to_file('test_file.pkl')
    gpp = GlobalParameters()
    gpp.load_from_file('test_file.pkl')
    print(vars(gpp))

    @njit
    def foo(global_params):
        #just a random function of my params
        print(global_params)
        print('all good')

    foo(gp.get_njitable())


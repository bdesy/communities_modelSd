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
        dictionary = pickle.load(path_to_dict)
        self.specify(dictionary)

    def save_to_file(self, path_for_dict):
        pickle.dump(vars(self), path_for_dict)

class GlobalParameters(Parameters):
    def specify(self, dictionary):
        self.dimension = dictionary['dimension']
        self.N = dictionary['N']
        self.mu = dictionary['mu']
        self.beta = dictionary['beta']
        self.euclidean = dictionary['euclidean']
        self.radius = dictionary['radius']
        self.njitable = (self.dimension, self.N, 
                         self.mu, self.beta, self.radius, 
                         self.euclidean)


class LocalParameters(Parameters):
    def specify(self, dictionary):
        self.nodes = dictionary['nodes']
        self.coordinates = dictionary['coordinates']
        self.kappas = dictionary['kappas']
        self.target_degrees = dictionary['target_degrees']

class OptimizatonParameters(Parameters):
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

    @njit
    def foo(global_params):
        #just a random function of my params
        print(global_params)
        print('all good')

    foo(gp.njitable)


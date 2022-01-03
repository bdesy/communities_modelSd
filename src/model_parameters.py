#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Parameter classes for the hyperbolic random graph model

Author: Béatrice Désy

Date : 03/01/2022
"""

from abc import ABC
import pickle

class Parameters(ABC):
	@abstractmethod
	def set_parameters(self, dictionary):
		pass

	@abstractmethod
	def create_parameters_dict(self):
		pass

	def load_parameters(self, path_to_dict):
        dictionary = pickle.load(path_to_dict)
        self.set_parameters(dictionary)

    def save_parameters(self, path_for_dict):
    	dictionary = self.create_parameters_dict()
    	pickle.dump(dictionary, path_for_dict)

class GlobalParameters(Parameters):
	def set_parameters(self, dictionary):
		self.dimension = dictionary.dimension
		self.N = dictionary.N
		self.mu = dictionary.mu
		self.beta = dictionary.beta
		self.euclidean = dictionary.euclidean
		self.radius = dictionary.radius

	def create_parameters_dict(self):
		parameters_dict = {'dimension': self.dimension,
						   'N': self.N,
						   'mu': self.mu,
						   'beta': self.beta,
						   'euclidean': self.euclidean,
						   'radius': self.radius}
		return parameters_dict

class LocalParameters(Parameters):
	def set_parameters(self, dictionary):
		self.nodes = dictionary.nodes
		self.coordinates = dictionary.coordinates
		self.kappas = dictionary.kappas
		self.target_degrees = dictionary.target_degrees

	def create_parameters_dict(self):
		parameters_dict = {'nodes': self.nodes,
						   'coordinates': self.coordinates,
						   'kappas': self.kappas,
						   'target_degrees': self.target_degrees}
		return parameters_dict

class OptimizatonParameters(Parameters):
	def set_parameters(self, dictionary):
		self.tol = dictionary.tol
		self.max_iterations = dictionary.max_iterations
		self.perturbation = dictionary.perturbation
		self.verbose = dictionary.verbose

	def create_parameters_dict(self):
		parameters_dict = {'tol': self.tol,
						   'max_iterations': self.max_iterations,
						   'perturbation': self.perturbation,
						   'verbose': self.verbose}
		return parameters_dict
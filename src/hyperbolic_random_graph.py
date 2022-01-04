#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Hyperbolic random graph model with heterogeneous angular positions

Author: Béatrice Désy

Date : 03/01/2022
"""

import pickle
import numpy as np
from numba import njit
from scipy.special import gamma
from tqdm import tqdm
from optimization_kappas import optimize_kappas
from geometric_functions import *
from model_parameters import *


def compute_default_mu(D, beta, average_kappa, *args_integral):
    if beta < D:
        mu = 0.01 #gamma(D/2.) / (average_kappa * 2 * np.pi**(D/2) ) POULET
        #mu /= integral_chi_normalization(D, beta, args_integral)
    else: 
        mu = gamma(D/2.) * np.sin(D*np.pi/beta) * beta
        mu /= np.pi**((D+1)/2)
        mu /= (2*average_kappa*D)
    return mu

@njit
def compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, global_parameters):
    D, N, mu, beta, R, euclidean = global_parameters
    chi = R * compute_angular_distance(coord_i, coord_j, D, euclidean)
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1. + chi**beta)

@njit
def compute_expected_degree(N, i, coordinates, kappas, global_parameters):
    coord_i = coordinates[i]
    kappa_i = kappas[i]
    expected_k_i = 0
    for j in range(N):
        if j!=(i):
            expected_k_i += compute_connection_probability(coord_i, coordinates[j], 
                                                           kappa_i, kappas[j], 
                                                           global_parameters)
    return expected_k_i


@njit
def compute_all_expected_degrees(N, coordinates, kappas, global_parameters):
    expected_degrees = np.zeros(N)
    for i in range(N):
        expected_degrees[i] = compute_expected_degree(N, i, coordinates, kappas, 
                                                      global_parameters)
    return expected_degrees


def get_target_degree_sequence(average_degree, N, rng, dist, sorted=True, y=2.5):
    if dist=='pwl':
        k_0 = (y-2) * average_degree / (y-1)
        a = y - 1.
        target_degrees = k_0 / rng.random(N)**(1./a)
    elif dist=='poisson':
        target_degrees = rng.poisson(average_degree-1., N)+1.
    elif dist=='exp':
        target_degrees = rng.exponential(scale=average_degree-1., size=N)+1.
        
    if sorted:
        target_degrees[::-1].sort()  
    
    return (target_degrees).astype(float)


@njit
def build_probability_matrix(N, kappas, coordinates, global_parameters, order=None):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i, kappa_i = coordinates[order[i]], kappas[order[i]]
        for j in range(i):
            coord_j, kappa_j = coordinates[order[j]], kappas[order[j]]
            mat[i,j] = compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, global_parameters)
    return mat+mat.T

@njit
def build_angular_distance_matrix(N, coordinates, D, euclidean, order=None):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i = coordinates[order[i]]
        for j in range(i):
            coord_j = coordinates[order[j]]
            mat[i,j] = compute_angular_distance(coord_i, coord_j, D, euclidean)
    return mat+mat.T


class ModelSD():
    def load_parameters(self, path_to_dict):
        file = open(path_to_dict, 'r')
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        self.set_parameters(dictionary)
        if self.D > 2:
            self.euclidean = True
        else:
            self.euclidean = False

    def set_parameters(self, params_dict):
        self.beta = params_dict['beta']
        if 'mu' in params_dict:
            self.mu = params_dict['mu']
        self.D = params_dict['dimension']
        if self.D > 2:
            self.euclidean = True
        else:
            self.euclidean = False
        if 'radius' in params_dict:
            self.R = params_dict['radius']
        if 'N' in params_dict:
            self.N = params_dict['N']
            self.R = compute_radius(self.N, self.D) 

    def set_mu_to_default_value(self, average_k):
        self.mu = compute_default_mu(self.D, self.beta, average_k)

    def load_hidden_variables(self, path_to_hidvar):
        self.nodes = np.loadtxt(path_to_hidvar, dtype=str).T[0]
        self.kappas = (np.loadtxt(path_to_hidvar, dtype=str).T[1]).astype('float')
        self.target_degrees = (np.loadtxt(path_to_hidvar, dtype=str).T[-1]).astype('float')
        if self.D<2.5:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:2+self.D]).astype('float').T
        else:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:3+self.D]).astype('float').T
        self.N = len(self.kappas)

    def set_hidden_variables(self, coordinates, kappas, target_degrees, nodes=None):
        self.coordinates = coordinates
        self.kappas = kappas
        self.target_degrees = target_degrees
        self.N = len(self.kappas)
        if nodes is None:
            self.nodes = np.arange(self.N)
        else:
            self.nodes = nodes

    def optimize_kappas(self, tol, max_iterations, rng, verbose=True, perturbation=0.1):
        kappas, success = optimize_kappas(self.N, tol, max_iterations, 
                        self.coordinates, self.kappas, 
                        self.R, self.beta, self.mu, self.target_degrees, 
                        rng, self.D, self.euclidean, verbose=verbose, perturbation=perturbation)
        self.kappas = kappas
        print('Optimization has succeeded : {}'.format(success))

    def load_from_graphml(self, path_to_xml):
        self.G = nx.read_graphml(path_to_xml)
        self.D = self.G.graph['dimension']
        self.mu = self.G.graph['mu']
        self.R = self.G.graph['radius']
        self.beta = self.G.graph['beta']
        path_to_hidvar = self.G.graph['hidden_variables_file']
        self.load_hidden_variables(path_to_hidvar)


    def compute_all_expected_degrees(self):
        self.expected_degrees = compute_all_expected_degrees(self.N, self.coordinates, self.kappas, 
                                    self.R, self.beta, self.mu, self.D, euclidean=self.euclidean)

    def build_probability_matrix(self, order=None):
        if (type(order) is str) and (order=='theta'):
            order = np.argsort(self.coordinates.T[0])
        self.probs = build_probability_matrix(self.N, self.kappas, self.coordinates, 
                                              self.R, self.beta, self.mu, self.D, 
                                              order=order, euclidean=self.euclidean)

    def build_angular_distance_matrix(self, order=None):
        if type(order) is str:
            if order=='theta':
                order = np.argsort(self.coordinates.T[0])
        self.angular_distance_matrix = build_angular_distance_matrix(self.N, self.coordinates, self.D, 
                                                                    order=order, euclidean=self.euclidean)

    def sample_random_matrix(self):
        rand = np.triu(np.random.random(size=(self.N, self.N)), k=1)
        rand += rand.T
        return np.where(self.probs>rand, 1, 0)

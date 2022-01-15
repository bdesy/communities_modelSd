#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Clustering figure from Garcia-Perez PhD thesis

Author: Béatrice Désy

Date : 14/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from geometric_functions import *
from numba import njit
import pickle

@njit
def mean_local_clustering(A, n):
    k = np.sum(A, axis=1)
    denum = k*(k-1)
    triangles = np.diag(np.dot(np.dot(A, A), A))
    local = np.where(denum>0.5, triangles/denum, 0)
    return np.sum(local)/n

def get_correct_mu(SD, target_average_degree, tol):
    SD.build_probability_matrix()
    av_degree = np.mean(np.sum(SD.probs, axis=1))
    while abs(target_average_degree - av_degree)>tol:
        perturbation = (target_average_degree-av_degree)/target_average_degree*SD.mu
        SD.gp.mu += perturbation
        SD.mu += perturbation
        SD.build_probability_matrix()
        av_degree = np.mean(np.sum(SD.probs, axis=1))

dimensions = np.arange(1,11)
gammas = [2.1, 2.325, 2.55, 2.775, 3.0]

N = 1000
beta = 1000.
nb_adj = 1
average_k = 10.

rng = np.random.default_rng()
opt_params = {'tol':1e-1, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}

res = {}

for D in dimensions:
    if D<2.5:
        euclidean=False
    else:
        euclidean=True
    global_params = {'N':N, 
                    'dimension': D,
                    'mu': compute_default_mu(D, beta, 10.),
                    'radius':compute_radius(N, D),
                    'beta':beta, 
                    'euclidean':euclidean}
    for y in gammas:
        key = 'S{}_gamma{}'.format(D, y)
        print(key)

        SD = ModelSD()
        tg = get_target_degree_sequence(average_k, 
                                        N, 
                                        rng, 
                                        'pwl', y=y,
                                        sorted=False)
        local_params = {'coordinates':sample_uniformly_on_hypersphere(N, D),
                        'kappas':tg+1e-3,
                        'nodes':np.arange(N),
                        'target_degrees':tg}
        SD.specify_parameters(global_params, local_params, opt_params)

        
        get_correct_mu(SD, 10., tol=0.1)
        SD.build_probability_matrix()
        print('final mu', SD.mu, 'average degree', np.mean(np.sum(SD.probs, axis=1)))
        print('')
        dist=[]
        for i in tqdm(range(nb_adj)):
            A = SD.sample_random_matrix().astype(float)
            dist.append(mean_local_clustering(A, N))
        dist = np.array(dist)
        res[key] = (np.mean(dist), np.std(dist))

with open('figure2', 'wb') as file:
    pickle.dump(res, file)




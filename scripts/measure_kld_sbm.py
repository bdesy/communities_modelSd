#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Script to measure KL divergence between DC-SBM and hyperbolic random graphs

Author: Béatrice Désy

Date : 31/08/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *
from graph_tool.all import *
import json
from numba import njit

@njit
def quick_build_sbm_matrix(n, comms_array, degree_seq, kappas, block_mat):
    probs = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            r, s = comms_array[i], comms_array[j]
            p_ij = degree_seq[i]*degree_seq[j]
            p_ij /= (kappas[r, 0]*kappas[s, 0])
            p_ij *= block_mat[r, s]
            probs[i,j] = p_ij
    probs += probs.T
    return np.where(probs>1., 1., probs)

def build_sbm_matrix(ajd, state):
    degree_seq = np.sum(adj, axis=0).astype(float)
    n = len(degree_seq)
    block_mat = (state.get_matrix()).todense()
    kappas = np.sum(block_mat, axis=1)
    comms = state.get_blocks()
    comms_array = np.array([comms[i] for i in range(n)])
    probs = quick_build_sbm_matrix(n, comms_array, degree_seq, kappas, block_mat)
    return probs

@njit
def KLD(p, q):
    mat = p * np.where(p*q>1e-14, np.log(p/q), 0)
    mat += (1.-p) * np.where((1.-p)*(1.-q)>1e-14, np.log((1.-p)/(1.-q)), 0)
    return np.sum(np.triu(mat))

def to_graph_tool(adj):
    g = Graph(directed=False)
    nnz = np.nonzero(np.triu(adj,1))
    nedges = len(nnz[0])
    g.add_edge_list(np.hstack([np.transpose(nnz),np.reshape(adj[nnz],(nedges,1))]))
    return g

#coordinate properties
centers = [[0, np.pi/2], [np.pi/2, 0.15], [3*np.pi/2, 0.15], [np.pi, np.pi/2]]
sizes = [150, 350, 350, 150]
N = int(np.sum(np.array(sizes)))
sig = 0.2
sigmas = [sig, sig, sig, sig]

#optimization stuff
tol = 1e-1
max_iterations = 1000
rng = np.random.default_rng()

#graph properties
average_k = 10.

#container for the results
beta_list = [2.1, 2.5, 3.0, 3.5, 4.0, 5.0, 10.]
res = {}
nb_adj = 100

for beta in beta_list:
    D=2
    SD = ModelSD()
    SD.set_parameters({'dimension':D, 'N':N, 'beta':beta})
    SD.set_mu_to_default_value(average_k)
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes=sizes)
    coordinates = np.column_stack((thetas, phis))
    target_degrees = get_target_degree_sequence(average_k, SD.N, rng, 'poisson', sorted=False, y=2.5) 
    SD.set_hidden_variables(coordinates, target_degrees+1e-3, target_degrees, nodes=None)
    SD.optimize_kappas(tol, max_iterations, rng, verbose=True, perturbation=0.1)
    SD.build_probability_matrix()

    #mdl_dist = measure_mdl(SD, nb_adj)
    key = 'S{}'.format(D)+'beta{}'.format(beta)
    print(key)
    res[key] = mdl_dist





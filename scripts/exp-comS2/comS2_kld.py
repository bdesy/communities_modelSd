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
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from geometric_functions import *
import argparse
from comS2_util import *
from numba import njit
import os

parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--nb_communities', type=int,
                        help='number of communities to put on equator')
parser.add_argument('-dd', '--degree_distribution', type=str,
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-o', '--order', type=bool, default=False,
                        help='whether or not to order the matrix with theta')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
args = parser.parse_args() 

@njit
def quick_build_sbm_matrix(n, comms_array, degree_seq, kappas, block_mat, order):
    probs = np.zeros((n,n))
    for ii in range(n):
        i = order[ii]
        for jj in range(i):
            j = order[jj] 
            r, s = comms_array[i], comms_array[j]
            p_ij = degree_seq[i]*degree_seq[j]
            p_ij /= (kappas[r]*kappas[s])
            p_ij *= block_mat[r, s]
            probs[i,j] = p_ij
    probs_sym = probs + probs.T
    return np.where(probs_sym>1., 1., probs_sym)

def get_ordered_homemade_sbm(n, sizes, adj, order):
    comms_array = np.zeros(n, dtype=int)
    i, c = 0, 0
    nc = len(sizes)
    ig = []
    for g in range(nc):
        ig.append(i)
        comms_array[i:i+sizes[g]] = c
        i+=sizes[g]
        c+=1
    block_mat = np.zeros((nc, nc))
    for j in range(nc):
        indices_j = np.argwhere(comms_array==j)
        j_i, j_f = ig[j], ig[j]+sizes[j]
        for k in range(nc):
            indices_k = np.argwhere(comms_array==k)
            k_i, k_f = ig[k], ig[k]+sizes[k]
            block_mat[j,k] = np.sum(adj[j_i:j_f, k_i:k_f])

    kappas = np.sum(block_mat, axis=1)
    return comms_array[order], kappas, block_mat


def get_cluster_coordinates_on_circle(N, nb_communities):
    theta_centers = np.linspace(0, 2*np.pi, nb_communities, endpoint=False)
    return list(theta_centers)

def get_communities_clusters(n, nc, sizes):
    i, c = 0, 0
    communities = np.zeros(n, dtype=int)
    for g in range(nc):
        communities[i:i+int(sizes[g])] = c
        i+=sizes[g]
        c+=1
    return communities

def get_sigma_d(sigma, D):
    sigma_d = 2*np.pi*np.exp(1-D)*sigma**2
    power = 1./(2*D)
    return (1./np.sqrt(2*np.pi))*sigma_d**power

def plot_matrices(a, b, D, beta, filepath):
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(a+1e-5), vmin=-5, vmax=0)
    plt.title(r'$S^{}$, $\beta={}$'.format(D, beta))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(b+1e-5), vmin=-5, vmax=0)
    plt.title('dcSBM')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath)
    #plt.show()
    plt.clf() 

#parse input 
dd = args.degree_distribution
nc = args.nb_communities
filename = 'data/kld/'+ str(nc) + 'comms_' + dd
order = None
sigma = args.sigma

#optimization stuff
opt_params = {'tol':1e-1, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':True}
rng = np.random.default_rng()

#graph properties
average_k = 10.
N=1000
mu=0.01

#container for the results
beta_list = [1.1, 1.3, 1.7, 2.1, 2.3, 2.7, 3.1, 3.3, 3.7, 4.0, 5.0, 7.0, 10., 14.]
res = {}
nb_adj = 20
explicit = True

print(filename)

#perform experiment

for beta_r in beta_list:
    target_degrees = get_target_degree_sequence(average_k, N, rng, dd, sorted=False) 
    coordinatesS2 = get_communities_coordinates_uniform(nc, N, sigma)
    coordinatesS1, R = project_coordinates_on_circle(coordinatesS2, N, rng, verbose=False)
    coordinates = [coordinatesS1, coordinatesS2]
    sizes = get_equal_communities_sizes(nc, N)
    for D in [2,1]:
        beta = beta_r*D
        key = 'S{}'.format(D)+'beta{}'.format(beta)
        global_params = get_global_params_dict(N, D, beta, mu)
        local_params = {'coordinates':coordinates[D-1], 
                    'kappas': target_degrees+1e-3, 
                    'target_degrees':target_degrees, 
                    'nodes':np.arange(N)}
        kld_dist = []
        SD = ModelSD()
        SD.specify_parameters(global_params, local_params, opt_params)
        SD.set_mu_to_default_value(average_k)
        SD.reassign_parameters()
        print(SD.mu, 'mu')
        SD.optimize_kappas(rng)
        SD.reassign_parameters()
        print(np.min(SD.kappas), 'min kappa')
        SD.build_probability_matrix(order=order)

        average_sbm_probs = np.zeros(SD.probs.shape)  
        for i in range(nb_adj):
            adj = SD.sample_random_matrix()
            degree_seq = np.sum(adj, axis=1).astype(float)
            comms_array, kappas, block_mat = get_ordered_homemade_sbm(N, sizes, adj, order=np.arange(N))
            print(comms_array)
            sbm_probs = quick_build_sbm_matrix(N, comms_array, degree_seq, kappas, block_mat, order=np.arange(N))
            average_sbm_probs += sbm_probs
            kld_dist.append(KLD_per_edge(SD.probs, sbm_probs))
        average_sbm_probs /= nb_adj
        if explicit:
            #fp = str('../data/kld/figures_ordered/av_S{}_beta{}.png'.format(D, beta))
            fp = 'data/kld/'+key
            fp = fp.replace('.', '')
            plot_matrices(SD.probs, average_sbm_probs, D, beta, fp) 
            #np.save('../data/kld/figures_ordered/av_S{}_beta{}_dcsbm'.format(D, beta), average_sbm_probs/nb_adj)
            #np.save('../data/kld/figures_ordered/av_S{}_beta{}_sd'.format(D, beta), SD.probs)
        res[key] = kld_dist
        print(kld_dist)


filepath = 'data/kld/test_'+filename

with open(filepath+'.json', 'w') as write_file:
    json.dump(res, write_file, indent=4)

with open(filepath+'_params.json', 'w') as write_file:
    json.dump(exp_params, write_file, indent=4)



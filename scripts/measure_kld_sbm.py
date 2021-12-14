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
import argparse
from time import time

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

def get_ordered_homemade_sbm(n, sizes, adj, order): ##Gérer le fait que la matrice est ptêt pas ordonnée
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
    mat = p * np.where(p*q>1e-14, np.log2(p/q), 0)
    mat += (1.-p) * np.where((1.-p)*(1.-q)>1e-14, np.log2((1.-p)/(1.-q)), 0)
    return np.sum(np.triu(mat))

@njit
def KLD_per_edge(p, q):
    n = p.shape[0]
    out = KLD(p, q)
    return out / (n*(n-1)/2)

def to_graph_tool(adj):
    g = Graph(directed=False)
    nnz = np.nonzero(np.triu(adj,1))
    nedges = len(nnz[0])
    g.add_edge_list(np.hstack([np.transpose(nnz),np.reshape(adj[nnz],(nedges,1))]))
    return g

def get_cluster_coordinates(N, nb_communities):
    sizes = [int(N/nb_communities) for i in range(nb_communities)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    theta_centers = np.linspace(0, 2*np.pi, nb_communities, endpoint=False)
    return list(theta_centers), sizes

def get_communities_clusters(n, nc, sizes):
    i, c = 0, 0
    communities = np.zeros(n, dtype=int)
    for g in range(nc):
        communities[i:i+int(sizes[g])] = c
        i+=sizes[g]
        c+=1
    return communities

def sample_gaussian_clusters_at_equator(D, theta_centers, sigmas, sizes):
    N = np.sum(sizes)
    communities = np.zeros(N, dtype=int)
    if D == 2:
        centers = [[theta, np.pi/2] for theta in theta_centers]
        thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes)
        coordinates = np.column_stack((np.array(thetas), np.array(phis)))
    elif D == 1:
        thetas = []
        for i in range(len(theta_centers)):
            n_i = sizes[i]
            for j in range(n_i):
                theta = np.random.normal(loc=theta_centers[i], scale=sigmas[i])%(2*np.pi)
                thetas.append(theta)
        coordinates = np.array(thetas).reshape((N, 1))
    elif D > 2:
        coord_list = []
        for i in range(len(theta_centers)):
            x_o = np.cos(theta_centers[i])
            y_o = np.sin(theta_centers[i])
            n_i = sizes[i]
            x = np.random.normal(loc=x_o, scale=sigmas[i], size=(n_i,1))
            y = np.random.normal(loc=y_o, scale=sigmas[i], size=(n_i,1))
            coordinates_i = np.concatenate((x,y), axis=1)
            for d in range(D-1):
                x_d = np.random.normal(loc=0, scale=sigmas[i], size=(n_i,1))
                coordinates_i = np.concatenate((coordinates_i, x_d), axis=1)
            denum = np.linalg.norm(coordinates_i, axis=1).reshape((n_i,1))
            coordinates_i = np.divide(coordinates_i, denum)
            coord_list.append(coordinates_i)
        coordinates = np.vstack(tuple(coord_list))
    return coordinates

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
    plt.show()
    plt.clf() 

#optimization stuff
tol = 1e-1
max_iterations = 1000
rng = np.random.default_rng()

#graph properties
average_k = 10.
N=1000

#container for the results
beta_list = [1.1, 1.3, 1.7, 2.1, 2.3, 2.7, 3.1, 3.3, 3.7, 4.0, 5.0, 7.0, 10., 14., 20.]
res = {}
nb_adj = 20
explicit = True

dd = args.degree_distribution
nc = args.nb_communities
filename = str(nc) + 'comms_' + dd
if args.order==True:
    order = 'theta'
    filename += '_ordered'
else:
    order = None
    filename += '_unordered'
print(filename)
theta_centers, sizes = get_cluster_coordinates(N, nc)

#perform experiment
ti=time()
for beta_r in beta_list:
    target_degrees = get_target_degree_sequence(average_k, N, rng, dd, sorted=False, y=2.5) 
    for D in [2,1]:
        beta = D*beta_r
        if beta > D:
            key = 'S{}'.format(D)+'beta{}'.format(beta)
            print(key)
            sigma_D = get_sigma_d(args.sigma, D)
            sigmas = [sigma_D for i in range (nc)]
            kld_dist = []
            for ell in range(1):
                coordinates = sample_gaussian_clusters_at_equator(D, theta_centers, sigmas, sizes)
                if args.order==True:
                    order_array = np.argsort(coordinates.T[0])
                else: 
                    order_array = np.arange(N)
                SD = ModelSD()
                SD.set_parameters({'dimension':D, 'N':N, 'beta':beta})
                SD.set_mu_to_default_value(average_k)
                SD.set_hidden_variables(coordinates, target_degrees+1e-3, target_degrees, nodes=None)
                print(np.min(SD.kappas), SD.mu, SD.N, SD.D, 'kappa0, mu, N, D')
                SD.optimize_kappas(tol, max_iterations, rng, verbose=True, perturbation=0.1)
                SD.build_probability_matrix(order=order) 
                average_sbm_probs = np.zeros(SD.probs.shape)  
                for i in range(nb_adj):
                    adj = SD.sample_random_matrix()
                    degree_seq = np.sum(adj, axis=1).astype(float)
                    comms_array, kappas, block_mat = get_ordered_homemade_sbm(N, sizes, adj, order=order_array)
                    print(comms_array)
                    sbm_probs = quick_build_sbm_matrix(N, comms_array, degree_seq, kappas, block_mat, order=order_array)
                    average_sbm_probs += sbm_probs
                    kld_dist.append(KLD_per_edge(SD.probs, sbm_probs))
                average_sbm_probs /= nb_adj
                if (ell==0) and explicit:
                    fp = str('../data/kld/figures_ordered/av_S{}_beta{}.png'.format(D, beta))
                    plot_matrices(SD.probs, average_sbm_probs, D, beta, fp) 
                    np.save('../data/kld/figures_ordered/av_S{}_beta{}_dcsbm'.format(D, beta), average_sbm_probs/nb_adj)
                    np.save('../data/kld/figures_ordered/av_S{}_beta{}_sd'.format(D, beta), SD.probs)
            res[key] = kld_dist

exp_params = {'beta_list':beta_list, 'average_k':average_k, 'N':N,
                'max_iterations':max_iterations, 'tol':tol,
                'nb_communities':nc, 'theta_centers':theta_centers,
                'sigmas':sigmas, 'sizes':sizes, 'order':args.order, 'dd':dd}

filepath = '../data/kld/test_'+filename

with open(filepath+'.json', 'w') as write_file:
    json.dump(res, write_file, indent=4)

with open(filepath+'_params.json', 'w') as write_file:
    json.dump(exp_params, write_file, indent=4)



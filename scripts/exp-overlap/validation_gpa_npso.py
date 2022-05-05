#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Check if modular matrices are diagonal with GPA (and nPSO) in D=1

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import json
from numba import njit
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
from overlap_util import *
import argparse

import networkx as nx
from infomap import Infomap
import matplotlib.pyplot as plt
import community as community_louvain

parser = argparse.ArgumentParser()
parser.add_argument('--community', '-c', type=str, choices=['louvain', 'infomap', 'SBM'],
                    help='community detection algorithm to use')
parser.add_argument('--lambda', '-V', type=float, default=1., 
                    help='Lambda parameter from Garcia-Perez 2018 soft communities...')
args = parser.parse_args()

#@njit
def compute_attractiveness(i, theta, placed_nodes, placed_thetas, y):
    rhs = 2./(placed_nodes**(1./(y-1.)))
    rhs /= i**((y-2.)/(y-1.))
    lhs = np.zeros(rhs.shape)
    for j in range(len(lhs)):
        coord_i = np.array([theta,])
        coord_j = placed_thetas[j]
        lhs[j] = compute_angular_distance(coord_i, coord_j, dimension=1, euclidean=False)
    return np.sum(np.where(lhs < rhs, 1, 0))

def get_angular_coordinates_GPA(N, y, V):
    nodes = []
    thetas = []
    for i in tqdm(range(1, N+1)):
        candidate_thetas = np.random.random(size=i)*2*np.pi
        A_candidates_thetas = np.zeros(candidate_thetas.shape, dtype=float)

        for ell in range(i):
            theta = candidate_thetas[ell]
            A_candidates_thetas[ell] = compute_attractiveness(i, theta, np.array(nodes), np.array(thetas), y)

        probs = A_candidates_thetas + V
        probs /= np.sum(probs)

        theta_i = np.random.choice(candidate_thetas, 1, p=list(probs))
        nodes.append(i)
        thetas.append(theta_i)
    return np.array(thetas).reshape((N,1))

def define_communities(SD, method):
    G = nx.from_numpy_matrix(np.matrix(SD.probs), create_using=nx.Graph())
    if method=='louvain':
        partition = community_louvain.best_partition(G)
    elif method=='infomap':
        im = Infomap()
        for node in G.nodes():
            im.add_node(int(node))
        for edge in list(G.edges(data=True)):
            n1, n2, data = edge
            im.add_link(int(n1), int(n2), data['weight'])
        im.run("-N10")
        partition = im.get_modules(depth_level=1)
    print(partition)
    communities = []
    for i in range(SD.N):
        communities.append(partition[i])
    SD.communities = np.array(communities)

# define model Class and optimize kappas
N=100
D=1
beta=3.5
mu = 0.01 
average_k = 10.
y=2.5
V=0.1

rng = np.random.default_rng()


global_params = get_global_params_dict(N, D, beta, mu)
target_degrees = get_target_degree_sequence(average_k, N, rng, 'pwl', sorted=True) ##vérifer algo ici
coordinates = get_angular_coordinates_GPA(N, y, V)
local_params = {'coordinates':coordinates, 
                'kappas': target_degrees+1e-3, 
                'target_degrees':target_degrees, 
                'nodes':np.arange(N)}
opt_params = {'tol':0.2, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}

SD = sample_model(global_params, local_params, opt_params, average_k, rng, optimize_kappas=True)
SD.build_probability_matrix() 

# identify communities

define_communities(SD, method='infomap')
n = len(set(SD.communities))

# sort prob matrix by communities, generate block mat

block_mat = get_community_block_matrix(SD, n)
new_order = get_order_theta_within_communities(SD, n)

SD.build_probability_matrix(order=new_order)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(131, projection='polar')
plot_coordinates_S1(SD, ax, n)

ax = fig.add_subplot(132)
im = ax.imshow(np.log10(SD.probs+1e-7), cmap='Greys')
ax.set_title('connection prob')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.axis('off')

ax = fig.add_subplot(133)
im = ax.imshow(np.log10(block_mat+1e-7), cmap='Greys')
ax.set_title('block_mat')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.axis('off')

plt.show()


# plot both matrices
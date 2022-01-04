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
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
args = parser.parse_args() 

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

#optimization stuff
tol = 1e-1
max_iterations = 1000
rng = np.random.default_rng()

#graph properties
average_k = 10.
N=1000

#container for the results
ratio_list = [1.5, 2.5, 3.5]

nc = args.nb_communities
filename = str(nc) + 'comms_'

theta_centers, sizes = get_cluster_coordinates(N, nc)

#compute and plot matrices
fig, axes = plt.subplots(len(ratio_list), 6, figsize=(10,15))
for i in range(len(ratio_list)):
    ratio = ratio_list[i]
    i_d = 0
    for dd in ['poisson', 'exp', 'pwl']:
        target_degrees = get_target_degree_sequence(average_k, N, rng, dd, sorted=False, y=2.5) 
        for D in [1,2]:
            beta = D*ratio
            title = r'$S^{}$'.format(D)+r', $\beta={}$'.format(beta)
            print(title)
            sigma_D = get_sigma_d(args.sigma, D)
            sigmas = [sigma_D for i in range (nc)]
            coordinates = sample_gaussian_clusters_at_equator(D, theta_centers, sigmas, sizes)
            order = np.argsort(coordinates.T[0])

            SD = ModelSD()
            SD.set_parameters({'dimension':D, 'N':N, 'beta':beta})
            SD.set_mu_to_default_value(average_k)
            SD.set_hidden_variables(coordinates, target_degrees+1e-3, target_degrees, nodes=None)
            SD.optimize_kappas(tol, max_iterations, rng, verbose=True, perturbation=0.1)
            SD.build_probability_matrix(order=order) 
            ax = axes[i, i_d+D-1]
            ax.imshow(np.log10(SD.probs+1e-5), vmin=-5, vmax=0, cmap='Greys')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        i_d+=2

filepath = '../figures/_'+filename
plt.tight_layout()
plt.savefig(filepath)
plt.show()


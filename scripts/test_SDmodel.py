#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Script to test my code as I restructure it

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *
import json
from numba import njit
from time import time

def plot_matrices(a, b, title_a, title_b):
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(a+1e-5), vmin=-5, vmax=0)
    plt.title(title_a)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(b+1e-5), vmin=-5, vmax=0)
    plt.title(title_b)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.clf() 

def get_sigma_d(sigma, D):
    sigma_d = 2*np.pi*np.exp(1-D)*sigma**2
    power = 1./(2*D)
    return (1./np.sqrt(2*np.pi))*sigma_d**power

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

def get_cluster_coordinates(N, nb_communities):
    sizes = [int(N/nb_communities) for i in range(nb_communities)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    theta_centers = np.linspace(0, 2*np.pi, nb_communities, endpoint=False)
    return list(theta_centers), sizes

#optimization stuff
tol = 1e-1
max_iterations = 1000
rng = np.random.default_rng()

#graph properties
average_k = 10.
N=1000
beta_r=3.5
sigma= 0.2
nc=4
theta_centers, sizes = get_cluster_coordinates(N, nc)

#container for result

matrices = []
titles = []

#perform experiment
target_degrees = get_target_degree_sequence(average_k, N, rng, 'exp', sorted=False, y=2.5) 
for D in [1,2]:
    beta = D*beta_r
    key = 'S{}'.format(D)+'beta{}'.format(beta)
    sigma_D = get_sigma_d(sigma, D)
    sigmas = [sigma_D for i in range (nc)]
    coordinates = sample_gaussian_clusters_at_equator(D, theta_centers, sigmas, sizes)
    order_array = np.argsort(coordinates.T[0])
    SD = ModelSD()
    SD.set_parameters({'dimension':D, 'N':N, 'beta':beta})
    SD.set_mu_to_default_value(average_k)
    SD.set_hidden_variables(coordinates, target_degrees+1e-3, target_degrees, nodes=None)
    print(np.min(SD.kappas), SD.mu, SD.N, SD.D, 'kappa0, mu, N, D')
    SD.optimize_kappas(tol, max_iterations, rng, verbose=True, perturbation=0.1)
    SD.build_probability_matrix(order=order_array)   
    matrices.append(SD.probs)
    titles.append(key)

plot_matrices(matrices[0], matrices[1], titles[0], titles[1])


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Generate community structure in the S1 model

Author: Béatrice Désy

Date : 29/07/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
import argparse
from time import time
from util import get_sigma_max, get_equal_communities_sizes, get_communities_coordinates
from util import get_communities_array_closest, get_order_theta_within_communities


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1000,
                        help='size of the graph')
parser.add_argument('-nc', '--nb_communities', type=int, default=8,
                        help='number of communities to put on the space')
parser.add_argument('-dd', '--degree_distribution', type=str, default='exp',
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-fs', '--sigma', type=float, default=0.1,
                        help='fraction of maximal sigma')
parser.add_argument('-br', '--beta_ratio', type=float, default=3.5,
                        help='value of beta for d=1')
parser.add_argument('-p', '--placement', type=str, default='uniformly',
                        choices = ['uniformly', 'randomly'],
                        help='clusters positions in the space')
parser.add_argument('-ok', '--optimize_kappas', type=bool, default=False)
args = parser.parse_args() 


def sample_custom_communities_S1(sizes, centers, sigmas):
    N = np.sum(sizes)
    thetas = np.zeros((N,1))
    ind = 0
    for i in range(len(centers)): #iterates on clusters
        n_i = sizes[i]
        coordinates_i = np.random.normal(loc=centers[i], scale=sigmas[i], size=n_i)%(2*np.pi)
        thetas[ind:ind+n_i] = coordinates_i.reshape((n_i, 1))
        ind+=n_i
    return thetas

def plot_model(S1):
    fig =plt.figure(figsize=(6,4))
    #plot circle
    ax = fig.add_subplot(121, projection='polar')
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(S1.communities==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=5, alpha=0.3)

    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')

    ax = fig.add_subplot(122)
    im = ax.imshow(np.log10(S1.probs+1e-5), cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    #setup
    N = args.N
    nb_com = args.nb_communities
    frac_sigma_max = args.sigma
    sigma1 = get_sigma_max(nb_com, 1)*frac_sigma_max 
    beta_r = args.beta_ratio
    rng = np.random.default_rng()

    sizes = get_equal_communities_sizes(nb_com, N) #a list, change to not have communities of ~equal size

    #sample angular coordinates on circle
    if args.placement=='uniformly':
        coordinates, centers = get_communities_coordinates(nb_com, N, sigma1, 
                                                            place='equator',
                                                            output_centers=True)
        coordinates = (coordinates.T[0]).reshape((N, 1)) #a 2D array with only one column, not a 1d array
        centers = (centers.T[0]).reshape((nb_com, 1)) #same here

    elif args.placement=='randomly':
        sigmas = np.ones(nb_com)*sigma1 #change to have communities of unequal angular spread
        centers = (np.random.random(size=nb_com)*(2*np.pi)).reshape((nb_com, 1))
        coordinates = sample_custom_communities_S1(sizes, centers, sigmas)

    #graph stuff
    mu = 0.01 #will be updated to default value within the model
    average_k = 10.
    target_degrees = get_target_degree_sequence(average_k, 
                                                N, 
                                                rng, 
                                                args.degree_distribution,
                                                sorted=False) 

    #optimization stuff for latent degrees
    opt_params = {'tol':1e-1, 
                'max_iterations': 1000, 
                'perturbation': 0.1,
                'verbose':True}


    S1 = ModelSD()
    D = 1
    global_params = get_global_params_dict(N, D, beta_r*D, mu)
    local_params = {'coordinates':coordinates, 
                    'kappas': target_degrees+1e-3, #initial value, will be optimized
                    'target_degrees':target_degrees, 
                    'nodes':np.arange(N)}

    S1.specify_parameters(global_params, local_params, opt_params)
    S1.set_mu_to_default_value(average_k)
    S1.reassign_parameters()

    if args.optimize_kappas:
        S1.optimize_kappas(rng)
        S1.reassign_parameters()

    labels = np.arange(nb_com)
    S1.communities = get_communities_array_closest(N, D, S1.coordinates, centers, labels)
    order = get_order_theta_within_communities(S1, nb_com)
    S1.build_probability_matrix(order=order) 

plot_model(S1)

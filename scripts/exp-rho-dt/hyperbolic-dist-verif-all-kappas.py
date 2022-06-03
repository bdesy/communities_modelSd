#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Assess my expression for the pdf of angular length of edges

Author: Béatrice Désy

Date : 13/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
from tqdm import tqdm
import argparse
from util import *

cmap = matplotlib.cm.get_cmap('viridis')
colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0)]


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int, default=1000,
                        help='number of nodes in the graph')
parser.add_argument('-k', '--kappa', type=float, default=4.,
                        help='value of average latent degree')
parser.add_argument('-br', '--beta_ratio', type=float, default=3.5,
                        help='value of beta for d=1')
parser.add_argument('-y', '--gamma', type=float, default=2.5,
                        help='value of gamma exponent for Pareto distribution')
parser.add_argument('-dd', '--degree_distribution', type=str, default='pwl',
                        help='shape of the latent degree distribution')
args = parser.parse_args()

N = args.nb_nodes
kappa = args.kappa
dd = args.degree_distribution
average_k = args.kappa

rng = np.random.default_rng()
opt_params = {'tol':0.2, 
        'max_iterations': 1000, 
        'perturbation': 0.1,
        'verbose':True}

for D in [2,1]:
    dist = []
    beta = args.beta_ratio*D
    if D<2.5:
        euclidean=False
    else:
        euclidean=True
    global_params = {'N':N, 
                    'dimension': D,
                    'mu': compute_default_mu(D, beta, kappa),
                    'radius':compute_radius(N, D),
                    'beta':beta, 
                    'euclidean':euclidean}
    coordinates = sample_uniformly_on_hypersphere(N, D)
    for i in tqdm(range(3)):
        print(D,'D', i, 'iteration')
        if len(dist)<1e6:
            target_degrees = get_target_degree_sequence(kappa, N, rng, dd, sorted=False, y=args.gamma)
            local_params = {'coordinates':coordinates,
                            'kappas':target_degrees,
                            'nodes':np.arange(N),
                            'target_degrees':target_degrees}
            SD = ModelSD()
            SD.specify_parameters(global_params, local_params, opt_params)
            SD.set_mu_to_default_value(average_k)
            SD.reassign_parameters()

            SD.compute_expected_degrees()
            plt.plot(np.sort(SD.expected_degrees), c=colors[D-1])
            #SD.optimize_kappas(rng)
            #SD.reassign_parameters()

            SD.build_probability_matrix()
            SD.build_hyperbolic_distance_matrix()
            
            A = SD.sample_random_matrix()
            m = np.sum(np.triu(A))
            connected_hyperbolic_distances = np.triu(A*SD.hyperbolic_distance_matrix)
            for ind in np.argwhere(connected_hyperbolic_distances>0.):
                dist.append(connected_hyperbolic_distances[ind[0], ind[1]])
    if dd=='pwl':
        filename = 'data/hyper-D{}-gamma{}-beta{}.txt'.format(D, args.gamma, args.beta_ratio)
    elif dd=='exp':
        filename = 'data/all-kappa-verif/hyper-exp-D{}-lambda{}-beta{}.txt'.format(D, kappa, args.beta_ratio)
    np.savetxt(filename, np.array(dist))

    #plt.hist(dist, bins=200, density=True, alpha=0.5, color=colors[D-1], label='D={}'.format(D))
plt.legend()
plt.show()
'''
Dthetas = np.linspace(1e-5, np.pi, 1000)
rho = np.sin(Dthetas)**(D-1)
pij = connection_prob(Dthetas, kappa, kappa, D, beta, R=SD.R, mu=SD.mu)
denum, error = integrated_connection_prob(kappa, kappa, D, beta, mu=SD.mu, R=SD.R)
plt.plot(Dthetas, pij*rho/denum, color='darkcyan')

plt.show()
'''

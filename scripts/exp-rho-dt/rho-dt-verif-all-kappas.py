#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Assess my expression for the pdf of angular length of edges

Author: Béatrice Désy

Date : 13/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
from tqdm import tqdm
import argparse
from util import *


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int, default=1000,
                        help='number of nodes in the graph')
parser.add_argument('-k', '--kappa', type=float, default=10.,
                        help='value of average latent degree')
parser.add_argument('-br', '--beta_ratio', type=float, default=3.5,
                        help='value of beta for d=1')
parser.add_argument('-y', '--gamma', type=float, default=2.5,
                        help='value of gamma exponent for Pareto distribution')
args = parser.parse_args()

N = args.nb_nodes
kappa = args.kappa

rng = np.random.default_rng()

for D in [2,1]:
    dist = []
    beta = args.beta_ratio*D
    if D<2.5:
        euclidean=False
    else:
        euclidean=True
    global_parameters = {'N':N, 
                    'dimension': D,
                    'mu': compute_default_mu(D, beta, kappa),
                    'radius':compute_radius(N, D),
                    'beta':beta, 
                    'euclidean':euclidean}
    coordinates = sample_uniformly_on_hypersphere(N, D)
    for i in tqdm(range(1000)):
        kappas = get_target_degree_sequence(kappa, N, rng, 'pwl', sorted=False, y=args.gamma)
        local_parameters = {'coordinates':coordinates,
                        'kappas':kappas,
                        'nodes':np.arange(N),
                        'target_degrees':np.ones(N)*args.kappa}
        SD = ModelSD()
        SD.gp.specify(global_parameters)
        SD.lp.specify(local_parameters)
        SD.reassign_parameters()
        SD.build_probability_matrix()
        SD.build_angular_distance_matrix()
        
        A = SD.sample_random_matrix()
        m = np.sum(np.triu(A))
        connected_angular_distances = np.triu(A*SD.angular_distance_matrix)
        for ind in np.argwhere(connected_angular_distances>0.):
            dist.append(connected_angular_distances[ind[0], ind[1]])
    filename = '../../../scratch/D{}-gamma{}-beta{}.txt'.format(D, args.gamma, args.beta_ratio)
    np.savetxt(filename, np.array(dist))
'''    
plt.hist(dist, bins=200, density=True, alpha=0.5, color='darkcyan')

Dthetas = np.linspace(1e-5, np.pi, 1000)
rho = np.sin(Dthetas)**(D-1)
pij = connection_prob(Dthetas, kappa, kappa, D, beta, R=SD.R, mu=SD.mu)
denum, error = integrated_connection_prob(kappa, kappa, D, beta, mu=SD.mu, R=SD.R)
plt.plot(Dthetas, pij*rho/denum, color='darkcyan')

plt.show()
'''

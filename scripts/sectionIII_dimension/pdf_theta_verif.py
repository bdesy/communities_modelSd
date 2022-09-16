#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Assess expression for the pdf of angular length of edges

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
parser.add_argument('-d', '--dimension', type=int, default=1,
                        help='dimension of sphere')
parser.add_argument('-k', '--kappa', type=float, default=10.,
                        help='value of both latent degree')
parser.add_argument('-br', '--beta_ratio', type=float, default=3.5,
                        help='value of beta for d=1')
args = parser.parse_args()

N = args.nb_nodes
D = args.dimension
kappa = args.kappa
beta = args.beta_ratio*D
if D<2.5:
	euclidean=False
else:
	euclidean=True

local_parameters = {'coordinates':sample_uniformly_on_hypersphere(N, D),
					'kappas':np.ones(N)*args.kappa,
					'nodes':np.arange(N),
					'target_degrees':np.ones(N)*args.kappa}
global_parameters = {'N':N, 
					'dimension': D,
					'mu': compute_default_mu(D, beta, 10.),
					'radius':compute_radius(N, D),
					'beta':beta, 
					'euclidean':euclidean}

SD = ModelSD()
SD.gp.specify(global_parameters)
SD.lp.specify(local_parameters)
SD.reassign_parameters()
SD.build_probability_matrix()
SD.build_angular_distance_matrix()
Dtheta = SD.angular_distance_matrix

dist = []

for i in tqdm(range(1000)):
	A = SD.sample_random_matrix()
	m = np.sum(np.triu(A))
	connected_angular_distances = np.triu(A*Dtheta)
	for ind in np.argwhere(connected_angular_distances>0.):
		dist.append(connected_angular_distances[ind[0], ind[1]])
plt.hist(dist, bins=200, density=True, alpha=0.5, color='darkcyan')

Dthetas = np.linspace(1e-5, np.pi, 1000)
rho = np.sin(Dthetas)**(D-1)
pij = connection_prob(Dthetas, kappa, kappa, D, beta, R=SD.R, mu=SD.mu)
denum, error = integrated_connection_prob(kappa, kappa, D, beta, mu=SD.mu, R=SD.R)
plt.plot(Dthetas, pij*rho/denum, color='darkcyan')
plt.show()
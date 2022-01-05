#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure experiment in the S1 and S1 model

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from time import time
import argparse
from comS2_util import *

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int,
                        help='number of nodes in the graph')
parser.add_argument('-nc', '--nb_communities', type=int,
                        help='number of communities to put on the sphere')
parser.add_argument('-dd', '--degree_distribution', type=str,
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-o', '--order', type=bool, default=False,
                        help='whether or not to order the matrix with theta')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
args = parser.parse_args() 

N = args.nb_nodes
nb_com = args.nb_communities
sigma = args.sigma

#sample angular coordinates on sphere and circle
coordinatesS2 = get_communities_coordinates_uniform(nb_com, N, sigma)
coordinatesS1 = project_coordinates_on_circle(coordinatesS2, N)
coordinates = [coordinatesS1, coordinatesS2]

#results
matrices = []

#graph stuff
beta = 3.5
mu = 0.01
average_k = 10.
rng = np.random.default_rng()
target_degrees = get_target_degree_sequence(average_k, N, rng, args.degree_distribution) 

#optimization stuff
tol = 1e-1
max_iterations = 1000
perturbation=0.1
verbose=True
opt_params = {'tol':tol, 
            'max_iterations': max_iterations, 
            'perturbation': perturbation,
            'verbose':verbose}

#create the SD models
for D in [1,2]:
    global_params = get_global_params_dict(N, D, beta*D, mu)
    local_params = {'coordinates':coordinates[D-1], 
                    'kappas': target_degrees+1e-3, 
                    'target_degrees':target_degrees, 
                    'nodes':np.arange(N)}

    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()

    SD.optimize_kappas(rng)
    SD.reassign_parameters()
    SD.build_probability_matrix(order=order_array)   
    matrices.append(SD.probs)

#plot 

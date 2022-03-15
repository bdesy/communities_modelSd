#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Script to generate samples of S1-S2 graphs and measure stuff on them

Author: Béatrice Désy

Date : 22/07/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *
from time import time

# set stuff

beta_list = [1.1, 2., 3., 4., 5., 10.,15.]
dico_keys = ['poissonS1', 'poissonS2', 'poissonS3', 'expS1', 'expS2', 'expS3', 'pwlS1', 'pwlS2', 'pwlS3']
degree_dist = ['exp', 'poisson']
nbg = 10

dico_dt = {}
dico_chi = {}

for key in dico_keys:
    dico_dt[key] = np.zeros((len(beta_list), nbg))
    dico_chi[key] = np.zeros((len(beta_list), nbg))

average_degree = 10.
N = 500

rng = np.random.default_rng()

R = [compute_radius(N, 1), compute_radius(N, 2), compute_radius(N, 3)]
mu = 0.04 #??????????????????????????????????????????????
tol = 1e-1
max_iterations = 300

ti=time()

# do the thing

i=0
for beta in beta_list:
    for j in tqdm(range(nbg)):
        for D in [3,2,1]:
            if D==3:
                euc=True
            else:
                euc=False
            coordinates = sample_uniformly_on_hypersphere(N=N, D=D)
            for dd in degree_dist:
                target_degrees = get_target_degree_sequence(average_degree, 
                                                        N, 
                                                        rng, 
                                                        dd, 
                                                        sorted=False)
                print(beta, j, dd, D, coordinates.shape)
                kappas, success = optimize_kappas(N, tol, max_iterations, coordinates, 
                        (target_degrees+1e-3)/5., R[D-1], beta, mu, 
                        target_degrees, rng, D, verbose=False)
                if success:
                    probs = build_probability_matrix(N, kappas, coordinates, R[D-1], beta, mu, D=D)
                    dthetas = build_angular_distance_matrix(N, coordinates, D, euclidean=euc)
                    chis = build_chi_matrix(N, coordinates, kappas, D, R[D-1], mu, euclidean=euc)

                    key = dd+'S'+str(D)
                    dico_chi[key][i,j] = d_1_theo(probs, chis)
                    dico_dt[key][i,j] = d_1_theo(probs, dthetas)
    i += 1



np.save('data/dico_average_chi_uniform3.npy', dico_chi)
np.save('data/dico_average_angular_distance_uniform3.npy', dico_dt)





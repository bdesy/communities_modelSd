#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Latent degrees optimization for tuning expected degrees
in the hyperbolic random graph model

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
from hyperbolic_random_graph import *

def optimize_kappas(N, tol, max_iterations, coordinates, kappas, R, beta, mu, target_degrees, rng, D, euclidean, verbose=False, perturbation=0.1):
    """Optimizes the hidden degrees given coordinates on S^D and target expected degree sequence

    Parameters
    ----------
    N : int
        Number of nodes in the graph
    coordinates : (N, D) array of floats for D=1,2, angular 
        (N, D+1) array of floats for D>2, euclidean
        coordinates of the nodes on the hypersphere S^D
    kappas : (N,) array of hidden degrees of the nodes
    R, beta, mu : floats
        Radius of the hypersphere, parameters of the model
    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    epsilon = 1.e3
    ell, m = 0, 0
    factor = 10
    stuck = 0
    while (epsilon > tol) and (m < max_iterations):
        for j in (tqdm(range(N))if verbose else range(N)):
            i = rng.integers(N)
            expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D, euclidean)
            while (abs(expected_k_i - target_degrees[i]) > tol*factor) and (ell < max_iterations):
                delta = rng.random()*perturbation
                kappas[i] = abs(kappas[i] + (target_degrees[i]-expected_k_i)*delta) 
                expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D, euclidean)
                ell += 1
            ell = 0
        expected_degrees = compute_all_expected_degrees(N, coordinates, kappas, R, beta, mu, D, euclidean)
        deviations = (target_degrees-expected_degrees)/target_degrees
        epsilon_m = np.max(np.array([np.max(deviations), abs(np.min(deviations))]))
        if abs(epsilon_m - epsilon)<1e-6:
            stuck += 1
        epsilon = epsilon_m
        factor = 1
        m += 1
        if verbose:
            print(m, epsilon)
        if stuck > 20:
            sign =  (rng.integers(2) - 0.5) * 2
            kappas += rng.random(size=(N,)) * perturbation * sign
            stuck = 0
    if m>max_iterations:
        success = False
        print('Max number of iterations, algorithm stopped at eps = {}'.format(epsilon))
    else:
        success = True
    return kappas, success

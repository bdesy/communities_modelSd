#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Functions specific to hyperbolic random graph model

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
from numba import njit
from scipy.special import gamma
from geometric_functions import *


def compute_default_mu(D, beta, average_kappa, *args_integral):
    if beta < D:
        mu = 0.01 #gamma(D/2.) / (average_kappa * 2 * np.pi**(D/2) ) POULET
        #mu /= integral_chi_normalization(D, beta, args_integral)
    else: 
        mu = gamma(D/2.) * np.sin(D*np.pi/beta) * beta
        mu /= np.pi**((D+1)/2)
        mu /= (2*average_kappa*D)
    return mu

@njit
def compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, global_parameters):
    D, N, mu, beta, R, euclidean = global_parameters
    chi = R * compute_angular_distance(coord_i, coord_j, D, euclidean)
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1. + chi**beta)

@njit
def compute_expected_degree(N, i, coordinates, kappas, global_parameters):
    coord_i = coordinates[i]
    kappa_i = kappas[i]
    expected_k_i = 0
    for j in range(N):
        if j!=(i):
            expected_k_i += compute_connection_probability(coord_i, coordinates[j], 
                                                           kappa_i, kappas[j], 
                                                           global_parameters)
    return expected_k_i


@njit
def compute_all_expected_degrees(N, coordinates, kappas, global_parameters):
    expected_degrees = np.zeros(N)
    for i in range(N):
        expected_degrees[i] = compute_expected_degree(N, i, coordinates, kappas, 
                                                      global_parameters)
    return expected_degrees


def get_target_degree_sequence(average_degree, N, rng, dist, sorted=True, y=2.5):
    if dist=='pwl':
        k_0 = (y-2) * average_degree / (y-1)
        a = y - 1.
        target_degrees = k_0 / rng.random(N)**(1./a)
    elif dist=='poisson':
        target_degrees = rng.poisson(average_degree-1., N)+1.
    elif dist=='exp':
        target_degrees = rng.exponential(scale=average_degree-1., size=N)+1.
        
    if sorted:
        target_degrees[::-1].sort()  
    
    return (target_degrees).astype(float)


@njit
def build_probability_matrix(N, coordinates, kappas, global_parameters, order=None):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i, kappa_i = coordinates[order[i]], kappas[order[i]]
        for j in range(i):
            coord_j, kappa_j = coordinates[order[j]], kappas[order[j]]
            mat[i,j] = compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, global_parameters)
    return mat+mat.T

@njit
def build_angular_distance_matrix(N, coordinates, D, euclidean, order=None):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i = coordinates[order[i]]
        for j in range(i):
            coord_j = coordinates[order[j]]
            mat[i,j] = compute_angular_distance(coord_i, coord_j, D, euclidean)
    return mat+mat.T

def get_global_params_dict(N, D, beta, mu):
    R=compute_radius(N, D)
    if D<2.5:
        euclidean=False
    else:
        euclidean=True
    global_params_dict={'N':N, 
                'dimension':D, 
                'beta':beta, 
                'mu':mu,
                'radius':R, 
                'euclidean':euclidean}
    return global_params_dict
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Samples Sd model with soft community structure, 
heterogeneous angular coordinates from geometric preferential
attachment and hidden degrees from a power-law distribution. 

Reproduction of algorithm from Garcia-Perez 2018 soft 
communitites paper (doi:https://doi.org/10.1007/s10955-018-2084-z)

Author: Béatrice Désy

Date : 17/03/2021
"""

import numpy as np
from tqdm import tqdm 
from numba import njit
from scipy.special import gamma

import matplotlib.pyplot as plt

# Define a few functions

def compute_radius(N, D=1):
    R = N * gamma((D+1.)/2.) / 2.
    R /= np.pi**((D+1.)/2.)
    return R**(1./D)

def compute_default_mu(beta, average_kappa, D=1):
    out = beta * np.sin(D * np.pi / beta) * gamma(D / 2.)
    out /= (2 * (np.pi**((D/2.)+1.)) * average_kappa)
    return out

@njit
def compute_angular_distance(phi, theta):
    return np.pi - abs(np.pi - abs(phi - theta))

@njit
def compute_attractiveness(i, phi, placed_nodes, placed_phis, y):
    rhs = 2./(placed_nodes**(1./(y-1.)))
    rhs /= i**((y-2.)/(y-1.))
    lhs = np.zeros(placed_nodes.shape)
    for j in range(len(placed_phis)):
        lhs[j] = compute_angular_distance(phi, placed_phis[j])
    return np.sum(np.where(lhs < rhs, 1, 0))

@njit
def compute_expected_degree(i, phis, kappas, R, beta, mu):
    phi_i = phis[i-1]
    kappa_i = kappas[i-1]
    expected_k_i = 0
    for j in range(N):
        if j!=(i-1):
            chi = R * compute_angular_distance(phi_i, phis[j])
            chi /= (mu * kappa_i * kappas[j])
            expected_k_i += 1./(1 + chi**beta)
    return expected_k_i

@njit
def get_candidates_probs(i, nodes, phis, y, V, candidates_phis):
    A_candidates_phis = np.zeros(candidates_phis.shape)
    for ell in range(i):
        phi = candidates_phis[ell]
        A_candidates_phis[ell] = compute_attractiveness(i, phi, nodes, phis, y)
    probs = A_candidates_phis + V + 1e-8 #epsilon to avoid the case where all A = 0
    probs /= np.sum(probs)
    return probs


def compute_angular_coordinates_gpa(N, y, V, rng):
    nodes = [1]
    phis = [(rng.random(size=1)*2*np.pi)[0]]     
    for i in tqdm(range(2, N+1)):
        candidates_phis = rng.random(size=i)*2*np.pi
        probs = get_candidates_probs(i, 
                                    np.array(nodes).flatten(), 
                                    np.array(phis).flatten(), 
                                    y, 
                                    V, 
                                    candidates_phis)
        phi_i = rng.choice(candidates_phis, 1, p=list(probs))[0]
        nodes.append(i)
        phis.append(phi_i)
    return np.array(phis)

def get_target_degree_sequence(average_degree, N, rng, dist='poisson', sorted=True, **kwargs):
    if dist=='power_law':
        k_0 = (y-2) * average_degree / (y-1)
        a = y - 1.
        target_degrees = k_0 / rng.random_sample(N)**(1./a)
    elif dist=='poisson':
        target_degrees = rng.poisson(average_degree, N)
    
    if sorted:
        target_degrees[::-1].sort()  
    
    return (target_degrees).astype(float)


if __name__ == "__main__":
    
    # Sets parameters

    N = 100
    y = 2.5
    V = 0.
    R = compute_radius(N)
    beta = 3.
    average_degree = 10.

    # Computes angular coordinates from GPA and Guille's algo
    rng = np.random.default_rng(12)
    phis = compute_angular_coordinates_gpa(N, y, V, rng)

    # Displays those angular coordinates

    plt.polar(phis, np.ones(N), 'o', ms=2)
    plt.show()

    plt.hist(phis, bins=120)
    plt.show()

    # Computes hidden degrees from Guille's algo

    target_degrees = get_target_degree_sequence(average_degree, 
                                                N, 
                                                rng, 
                                                dist='poisson', 
                                                sorted=True)
    print(np.min(target_degrees), 'min target degree')
    
    kappas = np.copy(target_degrees)

    mu = compute_default_mu(beta, np.mean(target_degrees))
    print('mu={}'.format(mu))

    tol = 10e-2
    epsilon = 10e-1
    iterations = 0
    max_iterations = 100*N
    while (epsilon > tol) and (iterations<max_iterations):
        for j in range(N):
            i = rng.integers(1,N+1)
            expected_k_i = compute_expected_degree(i, phis, kappas, R, beta, mu)
            delta = rng.random()*0.1
            kappas[i-1] = abs(kappas[i-1] + (target_degrees[i-1]-expected_k_i)*delta)

        deviations = np.zeros(N)
        expected_degrees = np.zeros(N)
        for i in range(1, N+1):
            k_i = target_degrees[i-1]
            expected_k_i = compute_expected_degree(i, phis, kappas, R, beta, mu)
            expected_degrees[i-1] = expected_k_i
            epsilon_i =  abs(k_i - expected_k_i)
            epsilon_i /= k_i
            deviations[i-1] = epsilon_i
        epsilon = np.max(deviations)
        iterations += 1

    if iterations==max_iterations:
        print('Max number of iterations, algorithm stopped at eps = {}'.format(epsilon))

    # Plots target degrees and kappas

    plt.plot(target_degrees, 'o', label='target degrees', c='purple')
    plt.plot(expected_degrees, 'o', label='expected degrees in ensemble', c='darkcyan')
    plt.plot(kappas, '^', label='kappas', c='coral')
    plt.legend()
    plt.show()

    # Saves the output in a format that the C++ code eats

    save=False

    if save:
        vertices = np.array(['v{:05d}'.format(i) for i in range(N)])
        data = np.column_stack((vertices, kappas, phis))
        filename = 'graph200_pwl_gpa_S1_hidvar.dat'
        np.savetxt(filename, data, delimiter='       ', fmt='%s',
                    header='vertex       kappa       theta      mu={}'.format(mu))

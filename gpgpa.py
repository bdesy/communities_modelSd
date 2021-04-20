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

import argparse
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
def compute_angular_distance(coord_i, coord_j, D=1):
    """Computes angular distancce between two points on an hypersphere

    Parameters
    ----------
    coord_i, coord_j : arrays of floats
        Coordinates of points i and j, (1, D) angular coordinates for D=1, D=2
        (1, D+1) euclidean coordinates for D>2

    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    if D==1:
        out = np.pi - abs(np.pi - abs(coord_i[0] - coord_j[0]))
    elif D==2:
        out = np.cos(abs(coord_i[0]-coord_j[0]))*np.cos(coord_i[1])*np.cos(coord_j[1])
        out += np.sin(coord_i[1])*np.sin(coord_j[1])
        out = np.arccos(out)
    else:
        denum = np.linalg.norm(coord_i)*np.linalg.norm(coord_j)
        out = np.arccos(np.dot(coord_i, coord_j)/denum)
    return out


@njit
def compute_attractiveness(i, phi, placed_nodes, placed_phis, y, D=1):
    """Computes attractiveness of a candidate coordinate

    Parameters
    ----------
    i : int
        Order of the new node in the network

    phi : array of floats
        Candidate coordinate for i, (1, D) angular coordinate for D=1, D=2
        (1, D+1) euclidean coordinate for D>2

    placed_nodes : array of int
        array of orders of previously placed nodes

    placed_phis : 2D array of floats
        Coordinates of i-1 previoulsy placed nodes, 
        (i-1, D) angular coordinate for D=1, D=2
        (i-1, D+1) euclidean coordinate for D>2

    y : float
        gamma parameter for the angular interval

    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    rhs = 2./(placed_nodes**(1./(y-1.)))
    rhs /= i**((y-2.)/(y-1.))
    lhs = np.zeros(placed_nodes.shape)
    for j in range(len(placed_phis)):
        lhs[j] = compute_angular_distance(phi, placed_phis[j], D)
    return np.sum(np.where(lhs < rhs, 1, 0))

@njit
def compute_expected_degree(i, phis, kappas, R, beta, mu, D=1):
    phi_i = phis[i-1]
    kappa_i = kappas[i-1]
    expected_k_i = 0
    for j in range(N):
        if j!=(i-1):
            chi = R * compute_angular_distance(phi_i, phis[j], D)
            chi /= (mu * kappa_i * kappas[j])
            expected_k_i += 1./(1 + chi**beta)
    return expected_k_i

@njit
def get_candidates_probs(i, nodes, phis, y, V, candidates_phis, D=1):
    A_candidates_phis = np.zeros(len(candidates_phis))
    for ell in range(i):
        phi = candidates_phis[ell]
        A_candidates_phis[ell] = compute_attractiveness(i, phi, nodes, phis, y, D)
    probs = A_candidates_phis + V + 1e-8 #epsilon to avoid the case where all A = 0
    probs /= np.sum(probs)
    return probs
'''

def generate_candidate_coordinates(n, D, rng):
    candidates_coord = rng.random(size=(n,1))*2*np.pi
    if D==2: 
        polar_coord = np.arccos(2 * rng.random(size=(n,1)) - 1)
        candidates_coord = np.column_stack((candidates_coord, polar_coord))
    elif D>2.5:
        candidates_coord = np.zeros((n, D+1))
        for i in range(n):
            pos = np.zeros(D+1)
            while np.linalg.norm(pos) < 1e-4:
                pos = rng.normal(size=D+1)
            candidates_coord[i] = pos / np.linalg.norm(pos)
    return candidates_coord
'''

@njit
def choose_random_coordinates_on_hypersphere(n, D, normal_coord):
    candidates_coord = np.zeros((n, D+1))
    j=0
    for i in range(n):
        pos = normal_coord[j]
        while np.linalg.norm(pos) < 1e-4:
            j+=1
            pos = normal_coord[j]
        candidates_coord[i] = pos / np.linalg.norm(pos)
        j+=1
    return candidates_coord

def generate_candidate_coordinates(n, D, rng):
    candidates_coord = rng.random(size=(n,1))*2*np.pi
    if D==2: 
        polar_coord = np.arccos(2 * rng.random(size=(n,1)) - 1)
        candidates_coord = np.column_stack((candidates_coord, polar_coord))
    elif D>2.5:
        normal_coord = rng.normal(size=(n+20,D+1))
        candidates_coord = choose_random_coordinates_on_hypersphere(n, D, normal_coord)
    return candidates_coord


def compute_angular_coordinates_gpa(N, y, V, rng, init=[], D=1):
    if len(init)==0:
        nodes = [1]
        phis = generate_candidate_coordinates(1, D, rng)
    else:
        nodes = list(np.arange(len(init))+1)
        phis = init
        print(phis.shape)
    for i in tqdm(range(len(phis)+1, N+1)):
        candidates_phis = generate_candidate_coordinates(i, D, rng)
        probs = get_candidates_probs(i, 
                                    np.array(nodes).flatten(), 
                                    np.array(phis), 
                                    y, 
                                    V, 
                                    candidates_phis,
                                    D)
        phi_i = rng.choice(candidates_phis, 1, p=list(probs))[0]
        nodes.append(i)
        phis = np.vstack((phis, phi_i))
    return np.array(phis)


def get_target_degree_sequence(average_degree, N, rng, dist='poisson', sorted=True, **kwargs):
    if dist=='pwl':
        k_0 = (y-2) * average_degree / (y-1)
        a = y - 1.
        target_degrees = k_0 / rng.random(N)**(1./a)
    elif dist=='poisson':
        target_degrees = rng.poisson(average_degree, N)
    elif dist=='exp':
        target_degrees = rng.exponential(scale=average_degree, size=N)
        
    if sorted:
        target_degrees[::-1].sort()  
    
    return (target_degrees).astype(float)

@njit
def compute_all_expected_degrees(N, phis, kappas, R, beta, mu):
    expected_degrees = np.zeros(N)
    for i in range(1, N+1):
        expected_degrees[i-1] = compute_expected_degree(i, phis, kappas, R, beta, mu)
    return expected_degrees


def optimize_kappas(N, tol, max_iterations, rng, phis, kappas, R, beta, mu, target_degrees):
    epsilon = tol+1.
    iterations = 0
    while (epsilon > tol) and (iterations<max_iterations):
        for j in range(N):
            i = rng.integers(1,N+1)
            expected_k_i = compute_expected_degree(i, phis, kappas, R, beta, mu)
            delta = rng.random()*0.1
            kappas[i-1] = abs(kappas[i-1] + (target_degrees[i-1]-expected_k_i)*delta)

        expected_degrees = compute_all_expected_degrees(N, phis, kappas, R, beta, mu)
        deviations = abs(target_degrees-expected_degrees)/target_degrees
        epsilon = np.max(deviations)
        iterations += 1

    if iterations==max_iterations:
        print('Max number of iterations, algorithm stopped at eps = {}'.format(epsilon))
    return kappas

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-p', type=str, default='hidvar',
                        help='path to save the hidden variable file')
    parser.add_argument('--size', '-N', type=int, default=1000, 
                        help='number of nodes')
    parser.add_argument('--beta', '-b', type=float, default=3.,
                        help='value of beta parameter')
    parser.add_argument('--average_degree', '-k', type=float, default=10.,
                        help='average degree of the target degree sequence')
    parser.add_argument('--Lambda', '-V', type=float, 
                        help='attractiveness parameter')
    parser.add_argument('--random_seed', '-rs', type=int, 
                        help='random seed for the RNG')
    parser.add_argument('--degree_distribution', '-dd', type=str, default='poisson',
                        choices=['poisson', 'exp', 'pwl'],
                        help='target degree distribution shape')
    parser.add_argument('--dimension', '-D', type=int, default=1, 
                        help='dimension of the sphere S^d')
    args = parser.parse_args()

    # Sets parameters

    N = args.size
    y = 2.5
    V = args.Lambda
    R = compute_radius(N)
    D = args.dimension
    beta = args.beta
    average_degree = args.average_degree

    # Computes angular coordinates from GPA and Guille's algo
    seed = args.random_seed
    rng = np.random.default_rng(seed)

    #init = np.array([[0.01,0.],[0.,np.pi]])
    init=np.array([[0.], [np.pi]])

    phis = compute_angular_coordinates_gpa(N, y, V, rng, D=D, init=init)

    # Displays those angular coordinates if D=1
    if D==1:
        plt.hist(phis, bins=120, color='darkcyan')
        plt.xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'2$\pi$'])
        plt.ylabel('Nombre de noeuds', fontsize=20)
        plt.xlabel('Angle', fontsize=20)
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 100)
        plt.show()

    # Computes hidden degrees from Guille's algo

    target_degrees = get_target_degree_sequence(average_degree, 
                                                N, 
                                                rng, 
                                                dist=args.degree_distribution, 
                                                sorted=True)

    kappas = np.copy(target_degrees)+1e-3

    # Set parameters for kappas optimization

    mu = compute_default_mu(beta, np.mean(target_degrees))
    tol = 10e-2
    max_iterations = 100*N

    kappas_opt = optimize_kappas(N, tol, max_iterations, rng, phis, kappas, R, beta, mu, target_degrees)
    expected_degrees = compute_all_expected_degrees(N, phis, kappas_opt, R, beta, mu)

    print(np.mean(expected_degrees), np.min(expected_degrees), np.sum(expected_degrees))

    # Plots target degrees and kappas

    plt.plot(target_degrees, 'o', label='target degrees', c='purple', ms=7)
    plt.plot(expected_degrees, 'o', label='expected degrees in ensemble', c='darkcyan', ms=3)
    plt.plot(kappas_opt, '^', label='kappas', c='coral')
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Saves the output in a format that the C++ code eats

    save=True

    if save:
        vertices = np.array(['v{:05d}'.format(i) for i in range(N)])
        data = np.column_stack((vertices, kappas, phis, target_degrees))
        filename = args.filename+'graph{}_'.format(int(N))
        filename += args.degree_distribution+'_gpa_S{}_rs{}'.format(D, seed)

        if D==1:
            header = 'vertex       kappa       theta      target degree'
        elif D==2:
            header = 'vertex       kappa       theta       phi       target degree'
        else:
            coords = ''
            for i in range(D+1):
                coords += 'x{}                       '.format(i)
            header = 'vertex       kappa        '+coords+'target degree'

        np.savetxt(filename+'.dat', data, delimiter='       ', fmt='%s',
                    header=header)

        params = {'mu':mu, 'beta':beta, 'dimension':D}
        params_file = open(filename+'_params.txt', 'a')
        params_file.write(str(params))
        params_file.close()
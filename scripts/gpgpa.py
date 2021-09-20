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

import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *

# Define a few functions

def compute_default_mu(beta, average_kappa, D=1):
    out = beta * np.sin(D * np.pi / beta) * gamma(D / 2.)
    out /= (2 * (np.pi**((D/2.)+1.)) * average_kappa)
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
def get_candidates_probs(i, nodes, phis, y, V, candidates_phis, D=1):
    A_candidates_phis = np.zeros(len(candidates_phis))
    for ell in range(i):
        phi = candidates_phis[ell]
        A_candidates_phis[ell] = compute_attractiveness(i, phi, nodes, phis, y, D)
    probs = A_candidates_phis + V + 1e-8 #epsilon to avoid the case where all A = 0
    probs /= np.sum(probs)
    return probs


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
    if D<2.5:
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
    D = args.dimension
    beta = args.beta
    average_degree = args.average_degree

    R = compute_radius(N, D)

    # Computes angular coordinates from GPA and Guille's algo
    seed = args.random_seed
    rng = np.random.default_rng(seed)

    init_4coms = np.array([[np.pi/6, np.pi/3], [5*np.pi/6, np.pi/3],[3*np.pi/2, np.pi/3], [0., np.pi]])
    init_3coms = np.array([[np.pi/6, np.pi/2], [5*np.pi/6, np.pi/2],[3*np.pi/2, np.pi/2]])
    #init = np.array([[0.01,0.],[np.pi/3,np.pi],[2*np.pi/3,np.pi]])
    init = np.array([[0.01,0.],[0.,np.pi]])
    #init=np.array([[0.], [np.pi]])

    phis = compute_angular_coordinates_gpa(N, y, V, rng, D=D, init=init_3coms)

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

    print('Mean target degree of {:.2f}'.format(np.mean(target_degrees)))

    kappas = np.copy(target_degrees)+1e-3

    # Set parameters for kappas optimization

    mu = compute_default_mu(beta, np.mean(target_degrees))
    tol = 10e-3
    max_iterations = 10*N
    print('optimizing kappas')
    kappas_opt = optimize_kappas(N, tol, max_iterations, 
                                                phis, kappas, 
                                                R, beta, mu, 
                                                target_degrees, rng,
                                                D=D, verbose=True)
    expected_degrees = compute_all_expected_degrees(N, phis, kappas_opt, R, beta, mu, D=D)

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

        params = {'mu':mu, 'beta':beta, 'dimension':D, 'radius':R}
        params_file = open(filename+'_params.txt', 'a')
        params_file.write(str(params))
        params_file.close()
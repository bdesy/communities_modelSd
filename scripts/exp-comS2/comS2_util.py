#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Functions for community structure experiment in the S1 and S1 model

Author: Béatrice Désy

Date : 03/01/2022
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *

from hrg_functions import *
from geometric_functions import *

from time import time

def get_order_theta_within_communities(SD, sizes):
    theta = SD.coordinates.T[0]
    i = 0
    order = np.argsort(theta[i:sizes[0]]).tolist()
    i += sizes[0]
    for s in sizes[1:]:
        order_s = (np.argsort(theta[i:i+s])+i).tolist()
        order += order_s
        i += s
    assert set(np.arange(SD.N).tolist())==set(order), 'not all indices in order'
    return np.array(order)

def sample_from_fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack((theta, phi))

def place_modes_coordinates_on_sphere(nb_com, place='uniformly'):
    if place=='uniformly':
        coordinates = sample_from_fibonacci_sphere(nb_com)
    elif place=='randomly':
        coordinates = sample_uniformly_on_hypersphere(nb_com, dimension=2)
    elif place=='equator':
        theta = np.linspace(0, 2*np.pi, nb_com, endpoint=False)
        phi = np.ones(theta.shape)*np.pi/2
        coordinates = np.column_stack((theta, phi))
    return coordinates

def get_equal_communities_sizes(nb_com, N):
    sizes = [int(N/nb_com) for i in range(nb_com)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    return sizes

def get_communities_coordinates(nb_com, N, sigma, place):
    sizes = get_equal_communities_sizes(nb_com, N)
    centers = place_modes_coordinates_on_sphere(nb_com, place=place)
    sigmas = np.ones(nb_com)*sigma
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes)
    return np.column_stack((thetas, phis))

def randomly_rotate_coordinates(coordinates, N, rng, reach=np.pi):
    sign = (rng.integers(2)-0.5)*2.
    x_angle, y_angle, z_angle = rng.random(size=3)*reach*sign
    R = get_xyz_rotation_matrix(x_angle, y_angle, z_angle)
    xyz = transform_angular_to_euclidean(coordinates)
    new_coordinates = rotate_euclidean_coordinates(xyz, N, R)
    return transform_euclidean_to_angular(new_coordinates), R

@njit
def rms_distance_to_equator(coordinates):
    phi = coordinates.T[1]
    return np.sqrt(np.mean(phi**2))

def project_coordinates_on_circle(coordinates, N, rng, verbose=False):
    rmsd = rms_distance_to_equator(coordinates)
    out = np.copy(coordinates)
    h=[]
    for i in tqdm(range(10000)):
        new_coordinates, R = randomly_rotate_coordinates(coordinates, N, rng, reach=np.pi)
        new_rmsd = rms_distance_to_equator(new_coordinates)
        h.append(new_rmsd)
        if new_rmsd < rmsd:
            out = new_coordinates
            rmsd = new_rmsd
    if verbose:
        #plt.hist(h, bins=40)
        #plt.show()
        print('Final RMS distance to equator is {}'.format(rmsd))
    return new_coordinates.T[0].reshape((N,1)), R

@njit
def KLD(p, q):
    mat = p * np.where(p*q>1e-14, np.log2(p/q), 0)
    mat += (1.-p) * np.where((1.-p)*(1.-q)>1e-14, np.log2((1.-p)/(1.-q)), 0)
    return np.sum(np.triu(mat))

@njit
def KLD_per_edge(p, q):
    n = p.shape[0]
    out = KLD(p, q)
    return out / (n*(n-1)/2)

@njit
def quick_build_sbm_matrix(n, comms_array, degree_seq, kappas, block_mat):
    probs = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            r, s = comms_array[i], comms_array[j]
            p_ij = degree_seq[i]*degree_seq[j]
            p_ij /= (kappas[r]*kappas[s])
            p_ij *= block_mat[r, s]
            probs[i,j] = p_ij
    probs_sym = probs + probs.T
    return np.where(probs_sym>1., 1., probs_sym)

def get_communities_array(n, sizes):
    comms_array = np.zeros(n, dtype=int)
    nc = len(sizes)
    i, c = 0, 0
    for g in range(nc):
        comms_array[i:i+sizes[g]] = c
        i+=sizes[g]
        c+=1
    return comms_array

def get_ordered_homemade_sbm(n, sizes, adj):
    comms_array = np.zeros(n, dtype=int)
    i, c = 0, 0
    nc = len(sizes)
    ig = []
    for g in range(nc):
        ig.append(i)
        comms_array[i:i+sizes[g]] = c
        i+=sizes[g]
        c+=1
    block_mat = np.zeros((nc, nc))
    for j in range(nc):
        indices_j = np.argwhere(comms_array==j)
        j_i, j_f = ig[j], ig[j]+sizes[j]
        for k in range(nc):
            indices_k = np.argwhere(comms_array==k)
            k_i, k_f = ig[k], ig[k]+sizes[k]
            block_mat[j,k] = np.sum(adj[j_i:j_f, k_i:k_f])

    kappas = np.sum(block_mat, axis=1)
    return comms_array, kappas, block_mat

def get_dcsbm_matrix(n, sizes, adj):
    comms_array, kappas, block_mat = get_ordered_homemade_sbm(n, sizes, adj)
    degree_seq = np.sum(adj, axis=1)
    dcSBM = quick_build_sbm_matrix(n, comms_array, degree_seq, kappas, block_mat)
    return dcSBM, comms_array

##tests
def test_randomly_rotate_coordinates():
    N=1000
    nb_tests=50
    rng = np.random.default_rng()
    res = np.zeros(nb_tests)
    for i in range(nb_tests):
        c = get_communities_coordinates_uniform(rng.integers(3, 20), N, 0.1)
        cp, R = randomly_rotate_coordinates(c, N, rng)
        d = build_angular_distance_matrix(N, c, D=2, euclidean=False)
        dp = build_angular_distance_matrix(N, cp, D=2, euclidean=False)
        res[i] = np.sum(abs(d-dp))
    assert np.max(res)<1e-8


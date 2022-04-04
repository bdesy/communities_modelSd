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
from scipy.integrate import quad
from scipy.stats import norm

from time import time

def normalize_block_matrix(block_mat, n):
    norm =  np.sum(block_mat)/2
    return block_mat/norm

def get_community_block_matrix(SD, n):
    labels = list(set(SD.communities))
    block_mat = np.zeros((n, n))
    for i in range(n):
        nodes_in_u = np.where(SD.communities==labels[i])[0]
        for j in range(i, n):
            nodes_in_v = np.where(SD.communities==labels[j])[0]
            block_mat[i,j] = get_block_sum(SD.probs, nodes_in_u, nodes_in_v)
    assert (np.sum(block_mat)-np.sum(SD.probs)<1e-5), 'sum of probs not equal'
    block_mat += np.triu(block_mat, k=1).T
    block_mat *= (1-np.eye(n))
    return block_mat

@njit
def get_block_sum(probs, nodes_in_u, nodes_in_v):
    indices = [[a,b] for a in nodes_in_u for b in nodes_in_v]
    block=0
    for i in indices:
        block+=probs[i[0], i[1]]
    return block

def get_coordinates(N, D, nc, sigma, output_centers=False):
    if output_centers==False:
        if D==1:
            coordinates = get_communities_coordinates(nc, N, sigma, place='equator')
            coordinates = (coordinates.T[0]).reshape((N, 1))
        elif D==2:
            coordinates = get_communities_coordinates(nc, N, sigma, place='uniformly')
        return coordinates
    else:
        if D==1:
            coordinates, centers = get_communities_coordinates(nc, N, sigma, 
                                place='equator', output_centers=output_centers)
            coordinates = (coordinates.T[0]).reshape((N, 1))
        elif D==2:
            coordinates, centers = get_communities_coordinates(nc, N, sigma, 
                                place='uniformly', output_centers=output_centers)
        return coordinates, centers


def get_order_theta_within_communities(SD, n):
    labels = list(set(SD.communities))
    theta = SD.coordinates.T[0]
    sorted_indices = np.argsort(theta)
    order = []
    for u in range(n):
        nodes_in_u = np.where(SD.communities==labels[u])[0]
        for i in sorted_indices:
            if i in nodes_in_u:
                order.append(i)
    assert set(np.arange(SD.N).tolist())==set(order), 'not all indices in order'
    return np.array(order)

def get_equal_communities_sizes(nb_com, N):
    sizes = [int(N/nb_com) for i in range(nb_com)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    return sizes

def get_communities_coordinates(nb_com, N, sigma, place, output_centers=False):
    sizes = get_equal_communities_sizes(nb_com, N)
    centers = place_modes_coordinates_on_sphere(nb_com, place=place)
    sigmas = np.ones(nb_com)*sigma
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes)
    if output_centers:
        return np.column_stack((thetas, phis)), centers
    else:
        return np.column_stack((thetas, phis))

def get_communities_array(n, sizes):
    comms_array = np.zeros(n, dtype=int)
    nc = len(sizes)
    i, c = 0, 0
    for g in range(nc):
        comms_array[i:i+sizes[g]] = c
        i+=sizes[g]
        c+=1
    return comms_array

def get_communities_array_closest(N, D, coordinates, centers, labels):
    comms_array = np.zeros(N, dtype=int)
    for i in range(N):
        c = find_closest_community(D, coordinates[i], centers, labels)
        comms_array[i] = c
    return comms_array

def find_closest_community(D, coordinate, centers, community_labels):
    closest_dist = np.pi
    c = None
    for u in range(len(centers)):
        dist_u = compute_angular_distance(coordinate, centers[u], D, euclidean=False)%np.pi
        if dist_u < closest_dist:
            c = community_labels[u]
            closest_dist = dist_u
    return c

def get_sigma_max(nc, D):
    if D==1:
        sigma = np.sqrt(2*np.pi)/nc
    elif D==2:
        sigma = np.sqrt(2./nc)
    return sigma

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

def project_coordinates_on_circle_with_R(coordinates, R, N):
    xyz = transform_angular_to_euclidean(coordinates)
    new_coordinates = rotate_euclidean_coordinates(xyz, N, R)
    return transform_euclidean_to_angular(new_coordinates)

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

# rank and disparity

def get_stable_rank(B):
    u, s, vh = np.linalg.svd(B)
    return np.sum(s**2)/(s[0])**2


def get_disparities(weights):
    strengths = np.sum(weights, axis=0)
    num = weights**2
    Y = np.sum(num, axis=0)
    return Y/strengths**2




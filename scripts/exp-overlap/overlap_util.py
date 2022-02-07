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

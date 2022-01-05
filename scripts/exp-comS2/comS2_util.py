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
    return coordinates

def get_equal_communities_sizes(nb_com, N):
    sizes = [int(N/nb_com) for i in range(nb_com)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    return sizes

def get_communities_coordinates_uniform(nb_com, N, sigma):
    sizes = get_equal_communities_sizes(nb_com, N)
    centers = place_modes_coordinates_on_sphere(nb_com, place='uniformly')
    sigmas = np.ones(nb_com)*sigma
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes)
    return np.column_stack((thetas, phis))

def randomly_rotate_coordinates(coordinates, N, rng):
    sign = (rng.integers(2)-0.5)*2.
    x_angle, y_angle, z_angle = rng.random(size=3)*np.pi*sign
    R = get_xyz_rotation_matrix(x_angle, y_angle, z_angle)
    xyz = transform_angular_to_euclidean(coordinates)
    new_coordinates = rotate_euclidean_coordinates(xyz, N, R)
    return transform_euclidean_to_angular(new_coordinates)

def project_coordinates_on_circle(coordinates, N):
    pass


##tests
def test_randomly_rotate_coordinates():
    N=1000
    nb_tests=50
    rng = np.random.default_rng()
    res = np.zeros(nb_tests)
    for i in range(nb_tests):
        c = get_communities_coordinates_uniform(rng.integers(3, 20), N, 0.1)
        cp = randomly_rotate_coordinates(c, N, rng)
        d = build_angular_distance_matrix(N, c, D=2, euclidean=False)
        dp = build_angular_distance_matrix(N, cp, D=2, euclidean=False)
        res[i] = np.sum(abs(d-dp))
    assert np.max(res)<1e-8


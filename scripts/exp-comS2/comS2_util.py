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

def project_coordinates_on_circle(coordinates, N):
    pass


##tests


from mpl_toolkits.mplot3d import Axes3D
def plot_points_on_sphere(coordinates, color='c'):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    #Import data
    theta, phi = coordinates.T[0], coordinates.T[1]
    xx = sin(phi)*cos(theta)
    yy = sin(phi)*sin(theta)
    zz = cos(phi)

    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color=color, alpha=0.3, linewidth=0)

    ax.scatter(xx,yy,zz,color="k",s=20)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.tight_layout()
    plt.show()

N=100
for i in range(5):
    R = get_rotation_matrix_x(0.2)
    c = get_communities_coordinates_uniform(16, N, 0.1)
    e = transform_angular_to_euclidean(c)
    rote = rotate_euclidean_coordinates(e, N, R)
    cp = transform_euclidean_to_angular(rote)
    d = build_angular_distance_matrix(N, c, D=2, euclidean=False)
    dp = build_angular_distance_matrix(N, cp, D=2, euclidean=False)
    print(np.sum(abs(d-dp)))
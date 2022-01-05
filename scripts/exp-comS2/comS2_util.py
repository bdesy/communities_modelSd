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

from mpl_toolkits.mplot3d import Axes3D
def plot_points_on_sphere(coordinates):
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
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.scatter(xx,yy,zz,color="k",s=20)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.tight_layout()
    plt.show()


@njit
def measure_upper_triangular_sum(a, n):
	total = 0
	for i in range(n):
		for j in range(i+1, n):
			total += a[i,j]
	return total

@njit
def measure_upper_triangular_mean(a, n):
	total = measure_upper_triangular_sum(a, n)
	return 2 * total / (n*(n-1))

@njit
def measure_distance_matrix_nonuniformity(D, n):
	mean = measure_upper_triangular_mean(D, n)
	nonuniformity = (D - mean)**2
	mse = measure_upper_triangular_mean(nonuniformity, n)
	return np.sqrt(mse)

def get_perturbed_coordinates(coordinates, rng, perturbation=0.2):
	sign = (rng.integers(2)-0.5)*2
	move = rng.random(coordinates.shape)*perturbation*sign
	return coordinates+move

def get_state(coordinates, n):
	D = build_angular_distance_matrix(n, coordinates, 2, False)
	rmse = measure_distance_matrix_nonuniformity(D, n)
	mean_distance = measure_upper_triangular_mean(D, n)
	return rmse, mean_distance

def relax_coordinates(coordinates, n, rng, tol):
	rmse, mean_distance = get_state(coordinates, n)
	while rmse > tol:
		print(rmse)
		prop_coordinates = get_perturbed_coordinates(coordinates, rng)
		prop_rmse, prop_mean_distance = get_state(prop_coordinates, n)
		if (prop_rmse < rmse) and (prop_mean_distance > mean_distance):
			coordinates = prop_coordinates
			rmse = prop_rmse
	return coordinates

def place_modes_coordinates_on_sphere(nb_com, uniform=True):
	coordinates = sample_uniformly_on_hypersphere(nb_com, dimension=2)
	if uniform==True:
		coordinates = relax_coordinates(coordinates, nb_com, np.random.default_rng(), tol=0.52)
	return coordinates

for i in range(5):
	c = place_modes_coordinates_on_sphere(8, uniform=True)
	plot_points_on_sphere(c)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Purely geometric functions for the hyperbolic random graph model

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
from scipy.special import gamma
from numba import njit

def compute_radius(N, dimension):
    '''Computes radius of hypersphere with surface area of N'''
    radius = N * gamma((dimension+1.)/2.) / 2.
    radius /= np.pi**((dimension+1.)/2.)
    return radius**(1./dimension)

@njit
def compute_angular_distance(coord_i, coord_j, dimension, euclidean):
    '''Computes angular distance between two points on an hypersphere'''
    if ((dimension==1) and (euclidean==False)):
        out = np.pi - abs(np.pi - abs(coord_i[0] - coord_j[0]))
    elif ((dimension==2) and (euclidean==False)):
        out = np.cos(abs(coord_i[0]-coord_j[0]))*np.sin(coord_i[1])*np.sin(coord_j[1])
        out += np.cos(coord_i[1])*np.cos(coord_j[1])
        out = np.arccos(out)
    else:
        denum = np.linalg.norm(coord_i)*np.linalg.norm(coord_j)
        out = np.arccos(np.dot(coord_i, coord_j)/denum)
    return out


#geometric transformation on unit sphere
@njit
def transform_angular_to_euclidean(coordinates):
    theta, phi = coordinates.T[0], coordinates.T[1]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x,y,z))

@njit
def transform_euclidean_to_angular(coordinates):
    x,y,z = coordinates.T[0], coordinates.T[1], coordinates.T[2]
    numerator = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(numerator, z)
    return np.column_stack((theta, phi))

@njit
def get_rotation_matrix_x(angle):
    R = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    R[1,1], R[2,2] = c, c
    R[1,2], R[2,1] = -s, s
    return R

@njit
def get_rotation_matrix_y(angle):
    R = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    R[0,0], R[2,2] = c, c
    R[2,0], R[0,2] = -s, s
    return R

@njit
def get_rotation_matrix_z(angle):
    R = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    R[0,0], R[1,1] = c, c
    R[0,1], R[1,0] = -s, s
    return R

@njit
def get_xyz_rotation_matrix(x_angle, y_angle, z_angle):
    R_x = get_rotation_matrix_x(x_angle)
    R_y = get_rotation_matrix_y(y_angle)
    R_z = get_rotation_matrix_z(z_angle)
    R_xy = np.dot(R_x, R_y)
    return np.dot(R_xy, R_z)

@njit
def rotate_euclidean_coordinates(coordinates, N, rotation_matrix):
    new_coordinates = np.zeros(coordinates.shape)
    for i in range(N):
        new_coordinates[i] = np.dot(rotation_matrix, coordinates[i].T)
    return new_coordinates

#ramdom sampling functions
def sample_gaussian_points_on_sphere(x_o, y_o, z_o, sigma):
    x = np.random.normal(loc=x_o, scale=sigma)
    y = np.random.normal(loc=y_o, scale=sigma)
    z = np.random.normal(loc=z_o, scale=sigma)
    point = np.array([x,y,z])
    point /= np.linalg.norm(point)
    num = np.sqrt(point[0]**2 + point[1]**2)
    theta = np.arctan2(point[1], point[0])
    phi = np.arctan2(num, point[-1])
    return theta, phi

def sample_gaussian_clusters_on_sphere(centers, sigmas, sizes):
    N = np.sum(sizes)
    thetas, phis = [],[]
    for i in range(len(centers)): #iterates on clusters
        n_i = sizes[i]
        theta, phi = centers[i]
        x_o = np.cos(theta)*np.sin(phi)
        y_o = np.sin(theta)*np.sin(phi)
        z_o = np.cos(phi)
        for j in range(n_i):
            s_theta, s_phi = sample_gaussian_points_on_sphere(x_o, y_o, z_o, sigmas[i])
            thetas.append(s_theta)
            phis.append(s_phi)
    return np.array(thetas), np.array(phis)

def sample_uniformly_on_hypersphere(N, dimension):
    if dimension == 1:
        coordinates = np.array([2 * np.pi * np.random.random(size=N)]).T
    elif dimension == 2:
        thetas = 2 * np.pi * np.random.random(size=N)
        phis = np.arccos(2 * np.random.random(size=N) - 1)
        coordinates = np.column_stack((thetas, phis))
    elif dimension > 2.5:
        coordinates = np.zeros((N, dimension+1))
        for i in range(N):
            pos = np.zeros(D+1)
            while np.linalg.norm(pos) < 1e-4:
                pos = np.random.normal(size=D+1)
            coordinates[i] = pos / np.linalg.norm(pos)
    return coordinates


# Tests

def test_compute_angular_distance_1D():
    theta_i, theta_j = np.array([0.]), np.array([0.5])
    assert abs(compute_angular_distance(theta_i, theta_j, 1, False) - 0.5) < 1e-5

def test_compute_angular_distance_2D():
    theta_i, theta_j = np.array([0., np.pi/2]), np.array([0.5, np.pi/2])
    assert abs(compute_angular_distance(theta_i, theta_j, 2, False) - 0.5) < 1e-5

def test_compute_angular_distance_3D():
    theta_i, theta_j = np.array([0., 0., 0., 1.,]), np.array([0., 0., 0., -1.,])
    assert abs(compute_angular_distance(theta_i, theta_j, 3, True) < np.pi) < 1e-5

def test_compute_radius_1D():
    assert abs(compute_radius(1000, 1) - 1000./(2*np.pi)) < 1e-5

def test_compute_radius_2D():
    assert abs(compute_radius(1000, 2) - np.sqrt(1000./(4*np.pi))) < 1e-5

def test_compute_expected_degree():
    pass
    #N, i, coordinates, kappas, R, beta, mu, D 
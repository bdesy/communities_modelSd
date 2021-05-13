#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Unable to write a description now

Author: Béatrice Désy

Date : 31/04/2021
"""

import ast
import numpy as np
import networkx as nx
from numba import njit
import matplotlib.pyplot as plt
from scipy.special import gamma


def compute_radius(N, D):
    """Computes radius of an hypersphere with N nodes on it and node density of 1

    Parameters
    ----------
    N : int
        Number of nodes in the graph

    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    R = N * gamma((D+1.)/2.) / 2.
    R /= np.pi**((D+1.)/2.)
    return R**(1./D)


@njit
def compute_angular_distance(coord_i, coord_j, D, euclidean=False):
    """Computes angular distancce between two points on an hypersphere

    Parameters
    ----------
    coord_i, coord_j : arrays of floats
        Coordinates of points i and j, (1, D) angular coordinates for D=1, D=2
        (1, D+1) euclidean coordinates for D>2

    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    if ((D==1) and (euclidean==False)):
        out = np.pi - abs(np.pi - abs(coord_i[0] - coord_j[0]))
    elif ((D==2) and (euclidean==False)):
        out = np.cos(abs(coord_i[0]-coord_j[0]))*np.sin(coord_i[1])*np.sin(coord_j[1])
        out += np.cos(coord_i[1])*np.cos(coord_j[1])
        out = np.arccos(out)
    else:
        denum = np.linalg.norm(coord_i)*np.linalg.norm(coord_j)
        out = np.arccos(np.dot(coord_i, coord_j)/denum)
    return out

@njit
def compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, R, beta, mu, D):
    chi = R * compute_angular_distance(coord_i, coord_j, D)
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1 + chi**beta)

@njit
def compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D):
    """Computes expected degree of a node in the S^D model

    Parameters
    ----------
    N : int
        Number of nodes in the graph
    i : int 
        Index of the node in the the hidden variables arrays
    coordinates : (N, D) array of floats for D=1,2, angular 
        (N, D+1) array of floats for D>2, euclidean
        coordinates of the nodes on the hypersphere S^D
    kappas : (N,) array of hidden degrees of the nodes
    R, beta, mu : floats
        Radius of the hypersphere, parameters of the model
    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    coord_i = coordinates[i]
    kappa_i = kappas[i]
    expected_k_i = 0
    for j in range(N):
        if j!=(i):
            expected_k_i += compute_connection_probability(coord_i, coordinates[j], kappa_i, kappas[j], R, beta, mu, D)
    return expected_k_i


@njit
def compute_all_expected_degrees(N, coordinates, kappas, R, beta, mu, D):
    """Computes expected degree of all nodes in the S^D model

    Parameters
    ----------
    N : int
        Number of nodes in the graph
    coordinates : (N, D) array of floats for D=1,2, angular 
        (N, D+1) array of floats for D>2, euclidean
        coordinates of the nodes on the hypersphere S^D
    kappas : (N,) array of hidden degrees of the nodes
    R, beta, mu : floats
        Radius of the hypersphere, parameters of the model
    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    expected_degrees = np.zeros(N)
    for i in range(N):
        expected_degrees[i] = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D=D)
    return expected_degrees



def get_target_degree_sequence(average_degree, N, rng, dist, sorted=True, y=2.5):
    if dist=='pwl':
        k_0 = (y-2) * average_degree / (y-1)
        a = y - 1.
        target_degrees = k_0 / rng.random(N)**(1./a)
    elif dist=='poisson':
        target_degrees = rng.poisson(average_degree-1., N)+1.
    elif dist=='exp':
        target_degrees = rng.exponential(scale=average_degree-1., size=N)+1.
        
    if sorted:
        target_degrees[::-1].sort()  
    
    return (target_degrees).astype(float)


def optimize_kappas(N, tol, max_iterations, coordinates, kappas, R, beta, mu, target_degrees, rng, D, verbose=False, perturbation=0.1):
    """Optimizes the hidden degrees given coordinates on S^D and target expected degree sequence

    Parameters
    ----------
    N : int
        Number of nodes in the graph
    coordinates : (N, D) array of floats for D=1,2, angular 
        (N, D+1) array of floats for D>2, euclidean
        coordinates of the nodes on the hypersphere S^D
    kappas : (N,) array of hidden degrees of the nodes
    R, beta, mu : floats
        Radius of the hypersphere, parameters of the model
    D : int
        Dimension of the hypersphere in a D+1 euclidean space
    """
    epsilon = 1.e3
    iterations = 0
    while (epsilon > tol) and (iterations < max_iterations):
        for j in range(N):
            i = rng.integers(N)
            expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D=D)
            while (abs(expected_k_i - target_degrees[i]) > tol):
                delta = rng.random()*perturbation
                kappas[i] = abs(kappas[i] + (target_degrees[i]-expected_k_i)*delta) 
                expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D=D)
        
        expected_degrees = compute_all_expected_degrees(N, coordinates, kappas, R, beta, mu, D=D)
        deviations = (target_degrees-expected_degrees)/target_degrees
        epsilon = np.max(np.array([np.max(deviations), abs(np.min(deviations))]))
        iterations += 1
        print(iterations, epsilon)

    if iterations==max_iterations:
        print('Max number of iterations, algorithm stopped at eps = {}'.format(epsilon))
    return kappas

@njit
def built_probability_matrix(N, kappas, coordinates, R, beta, mu, D, order=None):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i, kappa_i = coordinates[order[i]], kappas[order[i]]
        for j in range(i):
            coord_j, kappa_j = coordinates[order[j]], kappas[order[j]]
            mat[i,j] = compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, R, beta, mu, D)
    return mat+mat.T

@njit
def built_angular_distance_matrix(N, coordinates, D, order=None, euclidean=False):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i = coordinates[order[i]]
        for j in range(i):
            coord_j = coordinates[order[j]]
            mat[i,j] = compute_angular_distance(coord_i, coord_j, D, euclidean=euclidean)
    return mat+mat.T

@njit
def built_chi_matrix(N, coordinates, kappas, D, R, mu, order=None, euclidean=False):
    mat = np.zeros((N,N))
    if order is None:
        order = np.arange(N)
    for i in range(N):
        coord_i, kappa_i = coordinates[order[i]], kappas[order[i]]
        for j in range(i):
            coord_j, kappa_j = coordinates[order[j]], kappas[order[j]]
            delta_theta = compute_angular_distance(coord_i, coord_j, D, euclidean=euclidean)
            mat[i,j] = R * delta_theta / (mu*kappa_i*kappa_j)**(1./D)
    return mat+mat.T

class ModelSD():
    def load_parameters(self, path_to_dict):
        file = open(path_to_dict, 'r')
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        self.beta = dictionary['beta']
        self.mu = dictionary['mu']
        self.D = dictionary['dimension']
        self.R = dictionary['radius']

    def load_hidden_variables(self, path_to_hidvar):
        self.nodes = np.loadtxt(path_to_hidvar, dtype=str).T[0]
        self.kappas = (np.loadtxt(path_to_hidvar, dtype=str).T[1]).astype('float')
        self.target_degrees = (np.loadtxt(path_to_hidvar, dtype=str).T[-1]).astype('float')
        if self.D<2.5:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:2+self.D]).astype('float').T
        else:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:3+self.D]).astype('float').T
        self.N = len(self.kappas)

    def load_from_graphml(self, path_to_xml):
        self.G = nx.read_graphml(path_to_xml)
        self.D = self.G.graph['dimension']
        self.mu = self.G.graph['mu']
        self.R = self.G.graph['radius']
        self.beta = self.G.graph['beta']
        path_to_hidvar = self.G.graph['hidden_variables_file']
        self.load_hidden_variables(path_to_hidvar)


    def build_probability_matrix(self, order_theta=False):
        if order_theta:
            order = np.argsort(self.coordinates.T[0])
        else:
            order = None
        self.probs = built_probability_matrix(self.N, self.kappas, self.coordinates, 
                                              self.R, self.beta, self.mu, self.D, order=order)

    def build_angular_distance_matrix(self, order_theta=False, euclidean=False):
        if order_theta:
            order = np.argsort(self.coordinates.T[0])
        else:
            order = None
        self.angular_distance_matrix = built_angular_distance_matrix(self.N, self.coordinates, self.D, 
                                                                    order=order, euclidean=euclidean)

    def build_chi_matrix(self, order_theta=False, euclidean=False):
        if order_theta:
            order = np.argsort(self.coordinates.T[0])
        else:
            order = None
        self.chi_matrix = built_chi_matrix(self.N, self.coordinates, self.kappas, 
                                            self.D, self.R, self.mu,
                                            order=order, euclidean=euclidean)

    def compute_all_expected_degrees(self):
        self.expected_degrees = compute_all_expected_degrees(self.N, self.coordinates, self.kappas, 
                                    self.R, self.beta, self.mu, self.D)

# Tests

def test_compute_angular_distance_1D():
    theta_i, theta_j = np.array([0.]), np.array([0.5])
    assert abs(compute_angular_distance(theta_i, theta_j, D=1) - 0.5) < 1e-5

def test_compute_angular_distance_2D():
    theta_i, theta_j = np.array([0., np.pi/2]), np.array([0.5, np.pi/2])
    assert abs(compute_angular_distance(theta_i, theta_j, D=2) - 0.5) < 1e-5

def test_compute_angular_distance_3D():
    theta_i, theta_j = np.array([0., 0., 0., 1.,]), np.array([0., 0., 0., -1.,])
    assert abs(compute_angular_distance(theta_i, theta_j, D=3) < np.pi) < 1e-5

def test_compute_radius_1D():
    assert abs(compute_radius(1000, D=1) - 1000./(2*np.pi)) < 1e-5

def test_compute_radius_2D():
    assert abs(compute_radius(1000, D=2) - np.sqrt(1000./(4*np.pi))) < 1e-5

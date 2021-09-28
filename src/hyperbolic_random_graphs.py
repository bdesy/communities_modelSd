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
from tqdm import tqdm


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

def compute_default_mu(D, beta, average_kappa):
    if beta < D:
        print('Default value for mu is not valid if beta < D')
    else: 
        mu = gamma((D+1)/2.) * np.sin((D+1)*np.pi/beta) * beta
        mu /= np.pi**((D+2)/2)
        mu /= (2*average_kappa*(D+1))
    return mu

#@njit
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

#@njit
def compute_connection_probability(coord_i, coord_j, kappa_i, kappa_j, R, beta, mu, D):
    chi = R * compute_angular_distance(coord_i, coord_j, D)
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1. + chi**beta)

#@njit
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


#@njit
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
            x = np.random.normal(loc=x_o, scale=sigmas[i])
            y = np.random.normal(loc=y_o, scale=sigmas[i])
            z = np.random.normal(loc=z_o, scale=sigmas[i])
            point = np.array([x,y,z])
            point /= np.linalg.norm(point)
            num = np.sqrt(point[0]**2+ point[1]**2)
            thetas.append(np.arctan2(point[1], point[0]))
            phis.append(np.arctan2(num, point[-1]))
    return np.array(thetas), np.array(phis)


def sample_uniformly_on_hypersphere(N, D):
    if D == 1:
        coordinates = np.array([2 * np.pi * np.random.random(size=N)]).T
    elif D == 2:
        thetas = 2 * np.pi * np.random.random(size=N)
        phis = np.arccos(2 * np.random.random(size=N) - 1)
        coordinates = np.column_stack((thetas, phis))
    elif D > 2.5:
        coordinates = np.zeros((N, D+1))
        for i in range(N):
            pos = np.zeros(D+1)
            while np.linalg.norm(pos) < 1e-4:
                pos = np.random.normal(size=D+1)
            coordinates[i] = pos / np.linalg.norm(pos)
    return coordinates


def project_on_lower_dim(Df, prob_matrix, euclidean=False):
    '''
    Uses Laplacian Eigenmaps algorithm to project on a lower
    dimensional hypersphere
    '''
    N = prob_matrix.shape[0]
    D_matrix = np.diag(np.sum(prob_matrix, axis=1))
    L_matrix = D_matrix - prob_matrix
    iD_matrix = np.diag(1./np.sum(prob_matrix, axis=1))
    arr = np.matmul(iD_matrix, L_matrix)
    M, Y = np.linalg.eig(arr)

    if euclidean:
        ecoordinates = Y[:, 1:Df+2]
        norm = np.repeat(np.linalg.norm(ecoordinates, axis=1).reshape((N,1)), Df+1, axis=1)
        coordinates = ecoordinates / norm
    else:
        if Df==1:
            coordinates = (np.arctan2(Y[:,2], Y[:,1])+np.pi).reshape((N, 1))
        elif Df==2:
            phis = (np.arctan2(Y[:,2], Y[:,1])).reshape((N, 1))
            num = np.sqrt(Y[:,2]**2 + Y[:,1]**2)
            thetas = np.arctan2(num, Y[:,3]).reshape((N, 1))
            coordinates = np.column_stack((thetas, phis)) ## POULET MARCHE PAS À ARRANGER!!!
    return coordinates


@njit
def d_1_theo(probs, distances):
    num = np.sum(np.triu(probs*distances))
    denum = np.sum(np.triu(probs))
    return num / denum

@njit
def dkron(a, b, tol=1e-5):
    out=0
    if abs(a-b)<tol:
        out=1
    return out

@njit
def modularity(A, s_vector):
    degrees = np.sum(A, axis=1) 
    n = len(degrees)
    tm = np.sum(degrees)
    Q = 0
    for i in range(n):
        s_i = s_vector[i]
        k_i = degrees[i]
        for j in range(n):
            Q += ( A[i,j] - k_i*degrees[j]/(tm) ) * dkron(s_i, s_vector[j])
    return Q / (tm)


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
    ell, m = 0, 0
    factor = 10
    while (epsilon > tol):
        for j in (tqdm(range(N))if verbose else range(N)):
            i = rng.integers(N)
            expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D=D)
            while (abs(expected_k_i - target_degrees[i]) > tol*factor) and (ell < max_iterations):
                delta = rng.random()*perturbation
                kappas[i] = abs(kappas[i] + (target_degrees[i]-expected_k_i)*delta) 
                expected_k_i = compute_expected_degree(N, i, coordinates, kappas, R, beta, mu, D=D)
                ell += 1
            ell = 0
        expected_degrees = compute_all_expected_degrees(N, coordinates, kappas, R, beta, mu, D=D)
        deviations = (target_degrees-expected_degrees)/target_degrees
        epsilon = np.max(np.array([np.max(deviations), abs(np.min(deviations))]))
        factor = 1
        m += 1
        if verbose:
            print(m, epsilon)
        if m>max_iterations:
            success = False
        else:
            success = True

    if m==max_iterations:
        print('Max number of iterations, algorithm stopped at eps = {}'.format(epsilon))
    return kappas, success

#@njit
def build_probability_matrix(N, kappas, coordinates, R, beta, mu, D, order=None):
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
def build_angular_distance_matrix(N, coordinates, D, order=None, euclidean=False):
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
def build_chi_matrix(N, coordinates, kappas, D, R, mu, order=None, euclidean=False):
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
        self.set_parameters(dictionary)

    def set_parameters(self, params_dict):
        self.beta = params_dict['beta']
        if 'mu' in params_dict:
            self.mu = params_dict['mu']
        self.D = params_dict['dimension']
        if 'radius' in params_dict:
            self.R = params_dict['radius']
        if 'N' in params_dict:
            self.N = params_dict['N']
            self.R = compute_radius(self.N, self.D) 

    def set_mu_to_default_value(self, average_k):
        self.mu = compute_default_mu(self.D, self.beta, average_k)

    def load_hidden_variables(self, path_to_hidvar):
        self.nodes = np.loadtxt(path_to_hidvar, dtype=str).T[0]
        self.kappas = (np.loadtxt(path_to_hidvar, dtype=str).T[1]).astype('float')
        self.target_degrees = (np.loadtxt(path_to_hidvar, dtype=str).T[-1]).astype('float')
        if self.D<2.5:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:2+self.D]).astype('float').T
        else:
            self.coordinates = (np.loadtxt(path_to_hidvar, dtype=str).T[2:3+self.D]).astype('float').T
        self.N = len(self.kappas)

    def set_hidden_variables(self, coordinates, kappas, target_degrees, nodes=None):
        self.coordinates = coordinates
        self.kappas = kappas
        self.target_degrees = target_degrees
        self.N = len(self.kappas)
        if nodes is None:
            self.nodes = np.arange(self.N)
        else:
            self.nodes = nodes

    def optimize_kappas(self, tol, max_iterations, rng, verbose=True, perturbation=0.1):
        kappas, success = optimize_kappas(self.N, tol, max_iterations, 
                        self.coordinates, self.kappas, 
                        self.R, self.beta, self.mu, self.target_degrees, 
                        rng, self.D, verbose=verbose, perturbation=perturbation)
        print(kappas, self.kappas)
        self.kappas = kappas
        print('Optimization has succeeded : {}'.format(success))

    def load_from_graphml(self, path_to_xml):
        self.G = nx.read_graphml(path_to_xml)
        self.D = self.G.graph['dimension']
        self.mu = self.G.graph['mu']
        self.R = self.G.graph['radius']
        self.beta = self.G.graph['beta']
        path_to_hidvar = self.G.graph['hidden_variables_file']
        self.load_hidden_variables(path_to_hidvar)


    def build_probability_matrix(self, order=None):
        if (type(order) is str) and (order=='theta'):
            order = np.argsort(self.coordinates.T[0])
        self.probs = build_probability_matrix(self.N, self.kappas, self.coordinates, 
                                              self.R, self.beta, self.mu, self.D, order=order)

    def build_angular_distance_matrix(self, order=None, euclidean=False):
        if type(order) is str:
            if order=='theta':
                order = np.argsort(self.coordinates.T[0])
        self.angular_distance_matrix = build_angular_distance_matrix(self.N, self.coordinates, self.D, 
                                                                    order=order, euclidean=euclidean)

    def build_chi_matrix(self, order=None, euclidean=False):
        if type(order) is str:
            if order=='theta':
                order = np.argsort(self.coordinates.T[0])
        self.chi_matrix = build_chi_matrix(self.N, self.coordinates, self.kappas, 
                                            self.D, self.R, self.mu,
                                            order=order, euclidean=euclidean)

    def compute_all_expected_degrees(self):
        self.expected_degrees = compute_all_expected_degrees(self.N, self.coordinates, self.kappas, 
                                    self.R, self.beta, self.mu, self.D)

    def study_distance_distribution(self):
        average_angular_distance = np.sum(self.probs*self.angular_distance_matrix)/2
        average_chi = np.sum(self.probs*self.chi_matrix)/2
        num_average_angular_distance = []
        num_average_chi = []
        for i in tqdm(range(1000)):
            Ai = self.sample_random_matrix()
            m = np.sum(Ai)/2
            num_average_angular_distance.append(np.sum(Ai*self.angular_distance_matrix)/m)
            num_average_chi.append(np.sum(Ai*self.chi_matrix)/m)
        num_average_angular_distance = np.mean(np.array(num_average_angular_distance))
        num_average_chi = np.mean(np.array(num_average_chi))
        return average_angular_distance, num_average_angular_distance, average_chi, num_average_chi

    def sample_random_matrix(self):
        rand = np.random.random(size=(self.N, self.N))
        return np.where(self.probs>rand, 1, 0)

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

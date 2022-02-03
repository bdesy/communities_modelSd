#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure overlap experiment in the S1 and S1 model

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
import argparse
from time import time
from comS2_util import *
from scipy.integrate import quad
from scipy.stats import norm


def get_community_block_matrix(SD, sizes):
    nc = len(sizes)
    block_mat = np.zeros((nc, nc))
    i_i, j_i = 0, 0
    for i in range(nc):
        for j in range(nc):
            block_mat[i,j] = np.sum(SD.probs[i_i:i_i+sizes[i], j_i:j_i+sizes[j]])
            j_i += sizes[j]  
        i_i += sizes[i]
        j_i = 0
    return block_mat

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--nb_communities', type=int,
                        help='number of communities to put on the sphere')
parser.add_argument('-dd', '--degree_distribution', type=str,
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
parser.add_argument('-br', '--beta_ratio', type=float,
                        help='value of beta for d=1')
parser.add_argument('-p', '--placement', type=str, choices = ['uniformly', 'randomly'],
                        help='nodes placement in the spaces')
args = parser.parse_args() 

#setup
N = 1000
nb_com = args.nb_communities
sigma = args.sigma
beta_r = args.beta_ratio
rng = np.random.default_rng()

#sample angular coordinates on sphere and circle
coordinatesS2, centersS2 = get_communities_coordinates(nb_com, N, 
                                                    get_sigma_d(sigma, 2), 
                                                    place=args.placement, 
                                                    output_centers=True)
if args.placement=='uniformly':
    coordinatesS1, centers = get_communities_coordinates(nb_com, N, sigma, 
                                                        place='equator',
                                                        output_centers=True)
    coordinatesS1 = (coordinatesS1.T[0]).reshape((N, 1))
    centersS1 = (centers.T[0]).reshape((nb_com, 1))
else:
    coordinatesS1, R = project_coordinates_on_circle(coordinatesS2, N, rng, verbose=True)
    centersS1 = (project_coordinates_on_circle_with_R(centersS2, R, nb_com).T[0]).reshape((nb_com, 1))

coordinates = [coordinatesS1, coordinatesS2]
sizes = get_equal_communities_sizes(nb_com, N)

#compute overlap counting matrix
def normal_distribution_function(x, mu, sigma):
    value = norm.pdf(x, mu, sigma)
    return value

## DO TEST HERE
def integrate_overlap_probability_S1(mu1, mu2, sigma, show=False, factor=2.):
    dt = compute_angular_distance(mu1, mu2, dimension=1, euclidean=False)
    r = factor*sigma
    means = np.array([mu1, mu2])
    if dt > 2*r:
        p = 0.
    else:
        b_i = np.max(means)-r
        b_f = np.min(means)+r
        args = (mu1, sigma)
        p, err = quad(normal_distribution_function, b_i, b_f, args)
    return p

## DO TEST HERE
def integrate_overlap_probability_S2(mu1, mu2, sigma, show=False, factor=2.):
    r = factor*sigma
    thetas = np.array([mu1[0], mu2[0]])
    phis = np.array([mu1[1], mu2[1]])
    x = np.linspace(np.min(thetas)-r, np.max(thetas)+r, 1000)
    y = np.linspace(np.min(phis)-r, np.max(phis)+r, 1000)
    #mask
    xx, yy = np.meshgrid(x,y)
    c1 = np.where((xx-mu1[0])**2+(yy-mu1[1])**2<r, 1, 0)
    c2 = np.where((xx-mu2[0])**2+(yy-mu2[1])**2<r, 1, 0)
    mask = c1*c2
    #2d gaussian
    dst = ((xx-mu1[0])**2+(yy-mu1[1])**2)/sigma
    pdf = np.exp( - dst/2 )/(2*np.pi*sigma)
    area = np.diff(y)[0]*np.diff(x)[0]
    if show==True:
        plt.imshow(pdf*mask)
        plt.colorbar()
        plt.show()
    return np.sum(pdf*mask*area)

def get_overlap_matrix(centers, sigma, sizes, d, factor):
    nc = len(sizes)
    overlap_mat = np.zeros((nc, nc))
    #thetas = centers.T[0]%(2*np.pi)
    #phis = centers.T[1]%np.pi
    mus = centers#np.column_stack((thetas, phis))
    for i in range(nc):
        for j in range(i+1, nc):
            if d==2:
                p = integrate_overlap_probability_S2(mus[i], mus[j], sigma, factor)
            elif d==1:
                p = integrate_overlap_probability_S1(mus[i], mus[j], sigma, factor)
            overlap_mat[i,j] = p*(sizes[i]+sizes[j])
    return overlap_mat+overlap_mat.T

#graph stuff
mu = 0.01
average_k = 10.
target_degrees = get_target_degree_sequence(average_k, 
                                            N, 
                                            rng, 
                                            args.degree_distribution,
                                            sorted=False) 
print(np.max(target_degrees), 'max target degree')

#optimization stuff
opt_params = {'tol':1e-1, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':True}


S1, S2 = ModelSD(), ModelSD()
models = [S1, S2]
for D in [1,2]:
    global_params = get_global_params_dict(N, D, beta_r*D, mu)
    local_params = {'coordinates':coordinates[D-1], 
                    'kappas': target_degrees+1e-3, 
                    'target_degrees':target_degrees, 
                    'nodes':np.arange(N)}

    SD = models[D-1]
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()

    #SD.optimize_kappas(rng)
    SD.reassign_parameters()

    order = get_order_theta_within_communities(SD, sizes)

    SD.build_probability_matrix(order=order) 
    SD.communities = get_communities_array(N, sizes)

def plot_matrices(S1, S2, m1, m2):
    #the sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    #points on the sphere
    theta, phi = S2.coordinates.T[0], S2.coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    #plot sphere
    fig = plt.figure()
    ax = fig.add_subplot(321, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    for c in range(nb_com):
        color = plt.cm.tab10(c)
        nodes = np.where(S2.communities==c)
        ax.scatter(xx[nodes],yy[nodes],zz[nodes],color=color,s=10)
    ax.set_xlim([-1.,1.])
    ax.set_ylim([-1.,1.])
    ax.set_zlim([-1.,1.])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    #plot circle
    ax = fig.add_subplot(322, projection='polar')
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c)
        nodes = np.where(S2.communities==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=10)

    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')

    ax = fig.add_subplot(323)
    im = ax.imshow(np.log10(S2.probs+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(324)
    im = ax.imshow(np.log10(S1.probs+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(325)
    im = ax.imshow(m2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(326)
    im = ax.imshow(m1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


m1 = get_community_block_matrix(S1, sizes)
m2 = get_community_block_matrix(S2, sizes)
assert (np.sum(m1)-np.sum(S1.probs)<1e-3)
assert (np.sum(m2)-np.sum(S2.probs)<1e-3)

plot_matrices(S1, S2, m1*(1-np.eye(nb_com)), m2*(1-np.eye(nb_com)))

m1 = get_overlap_matrix(centersS1, sigma, sizes, d=1, factor=4)
m2 = get_overlap_matrix(centersS2, sigma, sizes, d=2, factor=4)
plot_matrices(S1, S2, m1, m2)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure overlap experiment example in the S1 and S1 model

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
import argparse
from time import time
from overlap_util import *

def get_strengths(mat):
    strengths = np.sum(mat, axis=0)
    return np.sort(strengths)[::-1]

def get_weights(mat):
    triu = np.where(np.triu(mat)>0)
    out = mat[triu]
    return np.sort(out.flatten())[::-1]

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--nb_communities', type=int, default=8,
                        help='number of communities to put on the sphere')
parser.add_argument('-dd', '--degree_distribution', type=str, default='exp',
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-fs', '--sigma', type=float,
                        help='fraction of maximal sigma')
parser.add_argument('-br', '--beta_ratio', type=float, default=3.5,
                        help='value of beta for d=1')
parser.add_argument('-p', '--placement', type=str, default='uniformly',
                        choices = ['uniformly', 'randomly'],
                        help='nodes placement in the spaces')
parser.add_argument('-ok', '--optimize_kappas', type=bool, default=False)
args = parser.parse_args() 


#setup
N = 1000
nb_com = args.nb_communities
frac_sigma_max = args.sigma
sigma1 = get_sigma_max(nb_com, 1)*frac_sigma_max
sigma2 = get_sigma_max(nb_com, 2)*frac_sigma_max

beta_r = args.beta_ratio
rng = np.random.default_rng()

#sample angular coordinates on sphere and circle
coordinatesS2, centersS2 = get_communities_coordinates(nb_com, N, 
                                                    sigma2, 
                                                    place=args.placement, 
                                                    output_centers=True)
if args.placement=='uniformly':
    coordinatesS1, centers = get_communities_coordinates(nb_com, N, sigma1, 
                                                        place='equator',
                                                        output_centers=True)
    coordinatesS1 = (coordinatesS1.T[0]).reshape((N, 1))
    centersS1 = (centers.T[0]).reshape((nb_com, 1))

else:
    coordinatesS1, R = project_coordinates_on_circle(coordinatesS2, N, rng, verbose=True)
    sigma1 = sigma2
    centersS1 = (project_coordinates_on_circle_with_R(centersS2, R, nb_com).T[0]).reshape((nb_com, 1))

coordinates = [coordinatesS1, coordinatesS2]
centers = [centersS1, centersS2]
sizes = get_equal_communities_sizes(nb_com, N)


#graph stuff
mu = 0.01
average_k = 10.
target_degrees = get_target_degree_sequence(average_k, 
                                            N, 
                                            rng, 
                                            args.degree_distribution,
                                            sorted=False) 

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

    if args.optimize_kappas:
        SD.optimize_kappas(rng)
        SD.reassign_parameters()

    labels = np.arange(nb_com)
    SD.communities = get_communities_array_closest(N, D, SD.coordinates, centers[D-1], labels)

    order = get_order_theta_within_communities(SD, nb_com)

    SD.build_probability_matrix(order=order) 
    #SD.communities = get_communities_array(N, sizes)
    


def plot_matrices(S1, S2, m1, m2, summ1, summ2):
    #the sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)*0.95
    y = np.sin(phi)*np.sin(theta)*0.95
    z = np.cos(phi)*0.95
    #points on the sphere
    theta, phi = S2.coordinates.T[0], S2.coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    #plot sphere
    fig =plt.figure(figsize=(6,7))
    ax = fig.add_subplot(321, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.7, linewidth=0, zorder=10)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(S2.communities==c)
        ax.scatter(xx[nodes],yy[nodes],zz[nodes],color=color,s=9)

    ax.set_xlim([-1.,1.])
    ax.set_ylim([-1.,1.])
    ax.set_zlim([-1.,1.])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    #plot circle
    ax = fig.add_subplot(322, projection='polar')
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(S1.communities==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=5, alpha=0.3)

    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')

    ax = fig.add_subplot(323)
    im = ax.imshow(np.log10(S2.probs+1e-5), cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    r = get_stable_rank(S2.probs)
    ax.set_title('connectivity S2, r={:.2f}'.format(r))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(324)
    im = ax.imshow(np.log10(S1.probs+1e-5), cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    r = get_stable_rank(S1.probs)
    ax.set_title('connectivity S1, r={:.2f}'.format(r))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    mi = np.min(m1)
    ma = np.max(np.array([np.max(m1), np.max(m2)]))
    ax = fig.add_subplot(325)
    im = ax.imshow(m2, cmap='Greys', vmin=mi, vmax=ma)
    ax.set_xticks([])
    ax.set_yticks([])
    r = get_stable_rank(m2)
    ax.set_title('S2, r={:.2f}, m={:.0f}'.format(r,summ1))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(326)
    im = ax.imshow(m1, cmap='Greys', vmin=mi, vmax=ma)
    ax.set_xticks([])
    ax.set_yticks([])
    r = get_stable_rank(m1)
    ax.set_title('S1, r={:.2f}, m={:.0f}'.format(r,summ2))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_quantities(B1, B2):
    fig, ax = plt.subplots(nrows=3, ncols=2, sharey='row', figsize=(6,6))
    models = [B2, B1]
    dims = [2,1]
    for i in [0,1]:
        D=dims[i]
        weights = get_weights(models[i])
        strengths = get_strengths(models[i])
        disparities = np.sort(get_disparities(models[i]))[::-1]
        r = get_stable_rank(models[i])
        ax[0, i].plot(weights, 'o', ms=2, label=r'$r={:2f}$'.format(r))
        ax[0, i].legend()
        ax[1, i].plot(strengths, 'o', ms=2)
        ax[2, i].axhline(np.mean(disparities), c='k', alpha=0.3)
        ax[2, i].plot(disparities, 'o', ms=2)

        ax[0, i].set_title(r'block matrix $S^{}$'.format(D))
    ax[0, 0].set_ylabel('weights')
    ax[1, 0].set_ylabel('strengths')
    ax[2, 0].set_ylabel('disparities')
    plt.show()

def plot_disparities(B1, B2):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(6,3))
    models = [B2, B1]
    dims = [2,1]
    for i in [0,1]:
        D=dims[i]
        disparities = np.sort(get_disparities(models[i]))[::-1]
        r = get_stable_rank(models[i])
        mu = np.mean(disparities)
        c_v = np.std(disparities)/mu
        ax[i].axhline(mu, c='k', alpha=0.3)
        ax[i].plot(disparities, 'o', ms=2, label='$\mu$={:0.2f}\n$\sigma$={:0.2f}\n$\sigma/\mu$={:0.2f}\n$\mu/\sigma$={:0.2f}'.format(mu, np.std(disparities), c_v, 1./c_v))
        ax[i].set_title(r'block matrix $S^{}$'.format(D))
        ax[i].legend()
    ax[0].set_ylabel('disparity of each community')
    plt.show()


m1 = get_community_block_matrix(S1, nb_com)
m2 = get_community_block_matrix(S2, nb_com)

summ1 = np.sum(m1*(1-np.eye(nb_com)))/2
summ2 = np.sum(m2*(1-np.eye(nb_com)))/2

m1 = normalize_block_matrix(m1, nb_com)
m2 = normalize_block_matrix(m2, nb_com)

plot_matrices(S1, S2, m1, m2, summ1, summ2)

plot_disparities(m1, m2)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure experiment in the S1 and S1 model

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from geometric_functions import *
from time import time
import argparse
from comS2_util import *
import os

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int, default=1000,
                        help='number of nodes in the graph')
parser.add_argument('-nc', '--nb_communities', type=int,
                        help='number of communities to put on the sphere')
parser.add_argument('-dd', '--degree_distribution', type=str,
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-o', '--order', type=bool, default=False,
                        help='whether or not to order the matrix with theta')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
parser.add_argument('-br', '--beta_ratio', type=float,
                        help='value of beta for d=1')
args = parser.parse_args() 

def get_order_theta_within_communities(SD, sizes):
    theta = SD.coordinates.T[0]
    i = 0
    order = np.argsort(theta[i:sizes[0]]).tolist()
    i += sizes[0]
    for s in sizes[1:]:
        order_s = (np.argsort(theta[i:i+s])+i).tolist()
        order += order_s
        i += s
    assert set(np.arange(N).tolist())==set(order), 'not all indices in order'
    return np.array(order)

@njit
def hadamard_distance(a, b, N):
    had = a*b / (b*b)
    return 2*had / (N*(N-1))

N = args.nb_nodes
nb_com = args.nb_communities
sigma = args.sigma
beta_r = args.beta_ratio

#specify random number generator
rng = np.random.default_rng()

#sample angular coordinates on sphere and circle
coordinatesS2 = get_communities_coordinates(nb_com, N, get_sigma_d(sigma, 2), place='randomly')
#coordinatesS1 = (get_communities_coordinates(nb_com, N, sigma, place='equator').T[0]).reshape((N, 1))
coordinatesS1, R = project_coordinates_on_circle(coordinatesS2, N, rng, verbose=True)
coordinates = [coordinatesS1, coordinatesS2]
sizes = get_equal_communities_sizes(nb_com, N)

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

#create the SD models

S1, S2 = ModelSD(), ModelSD()
models = [S1, S2]
path = 'data/assess/nc-{}-dd-{}-s{}-b{}-o{}'.format(nb_com, 
                                        args.degree_distribution, 
                                        str(sigma)[0]+str(sigma)[2], 
                                        int(beta_r),
                                        int(args.order))
os.mkdir(path)
path+='/'
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
    
    SD.optimize_kappas(rng)
    SD.reassign_parameters()

    if args.order:
        order = get_order_theta_within_communities(SD, sizes)
    else:
        order = None
    SD.build_probability_matrix(order=order) 
    SD.dcSBM, SD.communities = get_dcsbm_matrix(N, sizes, SD.probs)
    SD.save_all_parameters_to_file(path+'S{}-'.format(D))

#plot 
from mpl_toolkits.mplot3d import Axes3D
def plot_coordinates(S1, S2, save='fig'):
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
    #ax.view_init(elev=elev, azim=azi)
    #ax.set_box_aspect((1,1,1))
    #ax.grid(False)
    #plt.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])

    #plot circle
    ax = fig.add_subplot(322, projection='polar')
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c)
        nodes = np.where(S2.communities==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=10)
    #ax.plot(, , 'o', ms=2)
    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    #plot matrices POULETT
    ax = fig.add_subplot(323)
    ax.imshow(np.log10(S2.probs+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(324)
    ax.imshow(np.log10(S1.probs+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(325)
    ax.imshow(np.log10(S2.dcSBM+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(326)
    ax.imshow(np.log10(S1.dcSBM+1e-5))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save, dpi=600)
    plt.show()

plot_coordinates(S1, S2, save=path+'fig')

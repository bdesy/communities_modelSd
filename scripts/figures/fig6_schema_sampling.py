#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Schematic of sampling method

Author: Béatrice Désy

Date : 18/04/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
sys.path.insert(0, '../exp-overlap/')
from overlap_util import *
import argparse


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=16)

sampling = True

parser = argparse.ArgumentParser()
parser.add_argument('-ok', '--optimize_kappas', type=bool, default=False)
args = parser.parse_args() 

cmap = matplotlib.cm.get_cmap('viridis')
c1, c2 = cmap(1./20), cmap(1.1/3)

if sampling:
        #setup
    N = 500
    nb_com = 10
    frac_sigma_max = 0.4
    sigma1 = get_sigma_max(nb_com, 1)*frac_sigma_max
    sigma2 = get_sigma_max(nb_com, 2)*frac_sigma_max

    beta_r = 3.5
    rng = np.random.default_rng()

    #sample angular coordinates on sphere and circle
    coordinatesS2, centersS2 = get_communities_coordinates(nb_com, N, 
                                                        sigma2, 
                                                        place='uniformly', 
                                                        output_centers=True)
    coordinatesS1, centers = get_communities_coordinates(nb_com, N, sigma1, 
                                                            place='equator',
                                                            output_centers=True)
    coordinatesS1 = (coordinatesS1.T[0]).reshape((N, 1))
    centersS1 = (centers.T[0]).reshape((nb_com, 1))

    coordinates = [coordinatesS1, coordinatesS2]
    centers = [centersS1, centersS2]

    #graph stuff
    mu = 0.01
    average_k = 4.
    target_degrees = get_target_degree_sequence(average_k, 
                                                N, 
                                                rng, 
                                                'pwl',
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


    m1 = get_community_block_matrix(S1, nb_com)
    m2 = get_community_block_matrix(S2, nb_com)

    summ1 = np.sum(m1*(1-np.eye(nb_com)))/2
    summ2 = np.sum(m2*(1-np.eye(nb_com)))/2

    m1 = normalize_block_matrix(m1, nb_com)
    m2 = normalize_block_matrix(m2, nb_com)

    degs = np.linspace(0.01,60,1000)
    y = 2.5
    k0 = average_k*(y-2)/(y-1)
    pareto = (y-1)*k0**(y-1)*degs**(-y)

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    plt.hist(target_degrees, bins=100, color='k', alpha=0.5, density=True)
    plt.plot(degs, pareto, c='k')
    plt.yticks([],[])
    plt.xlabel('target degree')
    plt.xlim(0, 40)
    plt.ylim(0, 0.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('schematic/target.svg', dpi=600, format='svg')
    plt.show()

    #the circle
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='polar')
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    ax.plot(np.linspace(0, 2*np.pi, 1000), np.ones(1000), color='k', linewidth=0.5, zorder=10, alpha=0.3)
    ax.scatter(theta, np.ones(theta.shape), color=c1, s=40, linewidths=1, edgecolors='white',zorder=1)
    plt.ylim(0,1.5)
    plt.axis('off')
    plt.savefig('schematic/circle_coord_unif.svg', dpi=600, format='svg')
    plt.show()

    #the sphere
    fig = plt.figure(figsize=(3,3))
    l=0.9
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)*0.97
    y = np.sin(phi)*np.sin(theta)*0.97
    z = np.cos(phi)*0.97
    #points on the sphere
    theta, phi = S2.coordinates.T[0], S2.coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    #plot sphere
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.1, linewidth=0,zorder=0)
    #plt.plot(xx,yy,zz,'o', color='white',ms=9,zorder=8,alpha=0.5)
    #ax.plot_surface(x, y, z, color='white', antialiased=False)
    nodes = np.where(theta>np.pi/2)
    ax.scatter(xx[nodes],yy[nodes],zz[nodes], color=c2, s=40, linewidths=1, edgecolors='white')

    ax.set_xlim([-l,l])
    ax.set_ylim([-l,l])
    ax.set_zlim([-l,l])
    ax.axis('off')
    plt.savefig('schematic/sphere_coord.svg', dpi=600, format='svg')
    plt.show()

    fig = plt.figure(figsize=(3,3))
    plt.imshow(np.log10(S1.probs+1e-5), cmap='Purples', vmin=-5, vmax=0)
    plt.axis('off')
    plt.savefig('schematic/S1_probs_unif.svg', dpi=600, format='svg')
    plt.show()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(np.log10(S2.probs+1e-5), cmap='Blues', vmin=-5, vmax=0)
    plt.axis('off')
    plt.savefig('schematic/S2_probs_unif.svg', dpi=600, format='svg')
    plt.show()

    fig = plt.figure(figsize=(3,3))
    plt.imshow(np.log10(m1+1e-5), cmap='Purples', vmin=-5, vmax=0)
    plt.axis('off')
    plt.savefig('schematic/S1_B.svg', dpi=600, format='svg')
    plt.show()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(np.log10(m2+1e-5), cmap='Blues', vmin=-5, vmax=0)
    plt.axis('off')
    plt.savefig('schematic/S2_B.svg', dpi=600, format='svg')
    plt.show()





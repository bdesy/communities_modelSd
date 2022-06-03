#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Tests if RMI can lead to a numerical definition of 
maximum dispersion of communities on spheres

Author: Béatrice Désy

Date : 23/02/2022
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *
from overlap_util import * 
from scipy.optimize import fsolve

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=12)

def plot_hyperbolic_distance_dist(SD):
    SD.build_hyperbolic_distance_matrix()
    hd = SD.hyperbolic_distance_matrix
    dist = []
    for i in range(10):
        A = SD.sample_random_matrix()
        plt.imshow(A*hd)
        plt.colorbar()
        plt.show()
        realized_distances = np.triu(A*hd)
        for ind in np.where(realized_distances>0):
            dist.append(hd[ind[0], ind[1]])
    print(len(dist))


def plot_circle(ax, coordinates, labels, nb_com, N):
    theta = np.mod(coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels==c)
        ax.scatter(theta[nodes], np.ones(N)[nodes],
                    color=color, s=3, alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_ylim(0, 1.2)

def plot_sphere(ax, coordinates, labels, nb_com, N):
    #the sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    #points on the sphere
    theta, phi = coordinates.T[0], coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    #plot sphere
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.1, linewidth=0,zorder=0)
    plt.plot(xx,yy,zz,'o', color='white',ms=9,zorder=8,alpha=0.5)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels==c)
        ax.scatter(xx[nodes],yy[nodes],zz[nodes],color=color,s=3,zorder=10+c)
    l=0.7
    ax.set_xlim([-l,l])
    ax.set_ylim([-l,l])
    ax.set_zlim([-l+0.2,l])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')

def plot_coordinates(ax, SD, n):
    if SD.D==1:
        plot_circle(ax, SD.coordinates, SD.communities, n, SD.N)
    elif SD.D==2:
        plot_sphere(ax, SD.coordinates, SD.communities, n, SD.N)

def plot_blockmatrix(ax, SD, cmap):
    im = ax.imshow(np.log10(SD.blockmatrix+1e-5), cmap=cmap, vmin=-5, vmax=0)
    ax.axis('off')
    return im

def main():
    N = 1000
    n = 15
    frac_sigma_max = [0.2, 0.5, 0.8]
    br=3.5
    mu = 0.01
    average_k = 4.
    ok=False
    rng = np.random.default_rng()
    opt_params = {'tol':0.2, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}
    target_degrees = get_target_degree_sequence(average_k, N, 
                                            rng, 'pwl', sorted=False) 
    #sampling models
    models=[[],[],[]]
    for i in range(3):
        fs = frac_sigma_max[i]
        for D in [1,2]:
            sigma_max = get_sigma_max(n, D)
            sigma = fs*sigma_max
            beta = br*D
            global_params = get_global_params_dict(N, D, beta, mu)
            local_params = get_local_params(N, D, n, sigma, target_degrees)
            SD = sample_model(global_params, local_params, opt_params, average_k, rng, optimize_kappas=ok)
            define_communities(SD, n, reassign=True)
            order = get_order_theta_within_communities(SD, n)
            SD.build_probability_matrix(order=order) 
            plot_hyperbolic_distance_dist(SD)
            block_mat = get_community_block_matrix(SD, n)
            SD.blockmatrix = normalize_block_matrix(block_mat, n, all_edges=True)
            models[i].append(SD)
    #plot
    cmaps=['Purples', 'Blues']
    projections=['polar', '3d']
    num = [[],[],[]]
    cbar=False

    fig = plt.figure(figsize=(6,4))
    k=1
    for i in range(3):
        for j in range(4):
            D = int(j/2)+1
            if (j%2)==0:
                ax = fig.add_subplot(3,4,k, projection=projections[D-1])
                plot_coordinates(ax, models[i][D-1], n)
                k+=1
            else:
                ax = fig.add_subplot(3,4,k)
                im = plot_blockmatrix(ax, models[i][D-1], cmap=cmaps[D-1])
                if cbar:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                k+=1
    plt.savefig('mosaicpwl', dpi=600)
    plt.show()


if __name__=='__main__':
    main()

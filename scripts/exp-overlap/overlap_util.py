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
from scipy.integrate import quad
from scipy.stats import norm

from time import time

def get_dict_key(D, dd, nc, beta, f):
    return 'S{}-'.format(D)+dd+'-{}coms-{}beta-{:.2f}sigmam'.format(nc, beta, f)

def retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty, closest=False):
    y, err = [], []
    for f in frac_sigma_axis:
        key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
        if closest:
            key+='-closest'
        data = np.array(data_dict[key])  
        if qty == 'r':
            y.append(np.mean(data/nc))
            err.append(np.std(data/nc))
        elif qty == 'm':
            y.append(np.mean(data))
            err.append(np.std(data))
        elif qty=='Y':
            y.append(data)
        elif qty=='S':
            y.append(np.mean(data))
            err.append(np.std(data))
        elif qty=='degrees':
            y.append(data)
    return np.array(y), np.array(err)


def normalize_block_matrix(block_mat, n):
    norm =  np.sum(block_mat)/2
    return block_mat/norm

def get_community_block_matrix(SD, n):
    labels = list(set(SD.communities))
    block_mat = np.zeros((n, n))
    for i in range(n):
        nodes_in_u = np.where(SD.communities==labels[i])[0]
        for j in range(i, n):
            nodes_in_v = np.where(SD.communities==labels[j])[0]
            block_mat[i,j] = get_block_sum(SD.probs, nodes_in_u, nodes_in_v)
    assert (np.sum(block_mat)-np.sum(SD.probs)<1e-5), 'sum of probs not equal'
    block_mat += np.triu(block_mat, k=1).T
    block_mat *= (1-np.eye(n))
    return block_mat

@njit
def get_block_sum(probs, nodes_in_u, nodes_in_v):
    indices = [[a,b] for a in nodes_in_u for b in nodes_in_v]
    block=0
    for i in indices:
        block+=probs[i[0], i[1]]
    return block

def get_coordinates(N, D, nc, sigma, output_centers=False):
    if output_centers==False:
        if D==1:
            coordinates = get_communities_coordinates(nc, N, sigma, place='equator')
            coordinates = (coordinates.T[0]).reshape((N, 1))
        elif D==2:
            coordinates = get_communities_coordinates(nc, N, sigma, place='uniformly')
        return coordinates
    else:
        if D==1:
            coordinates, centers = get_communities_coordinates(nc, N, sigma, 
                                place='equator', output_centers=output_centers)
            coordinates = (coordinates.T[0]).reshape((N, 1))
        elif D==2:
            coordinates, centers = get_communities_coordinates(nc, N, sigma, 
                                place='uniformly', output_centers=output_centers)
        return coordinates, centers

def get_local_params(N, D, nc, sigma, target_degrees): 
    coordinates = get_coordinates(N, D, nc, sigma)
    local_params = {'coordinates':coordinates, 
                                'kappas': target_degrees+1e-3, 
                                'target_degrees':target_degrees, 
                                'nodes':np.arange(N)}
    return local_params

def sample_model(global_params, local_params, opt_params, average_k, rng, optimize_kappas=True):
    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()
    if optimize_kappas:
        SD.optimize_kappas(rng)
    SD.reassign_parameters()
    return SD

def define_communities(SD, n, reassign):
    if reassign==False:
        sizes = get_equal_communities_sizes(n, SD.N)
        SD.communities = get_communities_array(SD.N, sizes)
    elif reassign:
        labels = np.arange(n)
        if SD.D==2:
            misc, centers = get_communities_coordinates(n, SD.N, 0.01, place='uniformly', 
                                                    output_centers=True)
        elif SD.D==1:
            misc, centers = get_communities_coordinates(n, SD.N, 0.01, place='equator', 
                                                    output_centers=True)
            centers = (centers.T[0]).reshape((n, 1))
        SD.communities = get_communities_array_closest(SD.N, SD.D, SD.coordinates, centers, labels)


def get_order_theta_within_communities(SD, n):
    labels = list(set(SD.communities))
    theta = SD.coordinates.T[0]
    sorted_indices = np.argsort(theta)
    order = []
    for u in range(n):
        nodes_in_u = np.where(SD.communities==labels[u])[0]
        for i in sorted_indices:
            if i in nodes_in_u:
                order.append(i)
    assert set(np.arange(SD.N).tolist())==set(order), 'not all indices in order'
    return np.array(order)

def get_equal_communities_sizes(nb_com, N):
    sizes = [int(N/nb_com) for i in range(nb_com)]
    sizes[0] += (N - int(np.sum(np.array(sizes))))
    return sizes

def get_communities_coordinates(nb_com, N, sigma, place, output_centers=False):
    sizes = get_equal_communities_sizes(nb_com, N)
    centers = place_modes_coordinates_on_sphere(nb_com, place=place)
    sigmas = np.ones(nb_com)*sigma
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes)
    if output_centers:
        return np.column_stack((thetas, phis)), centers
    else:
        return np.column_stack((thetas, phis))

def get_communities_array(n, sizes):
    comms_array = np.zeros(n, dtype=int)
    nc = len(sizes)
    i, c = 0, 0
    for g in range(nc):
        comms_array[i:i+sizes[g]] = c
        i+=sizes[g]
        c+=1
    return comms_array

def get_communities_array_closest(N, D, coordinates, centers, labels):
    comms_array = np.zeros(N, dtype=int)
    for i in range(N):
        c = find_closest_community(D, coordinates[i], centers, labels)
        comms_array[i] = c
    return comms_array

def find_closest_community(D, coordinate, centers, community_labels):
    closest_dist = np.pi
    c = None
    for u in range(len(centers)):
        dist_u = compute_angular_distance(coordinate, centers[u], D, euclidean=False)%np.pi
        if dist_u < closest_dist:
            c = community_labels[u]
            closest_dist = dist_u
    return c

def get_sigma_max(nc, D):
    if D==1:
        sigma = np.sqrt(2*np.pi)/nc
    elif D==2:
        sigma = np.sqrt(2./nc)
    return sigma

def randomly_rotate_coordinates(coordinates, N, rng, reach=np.pi):
    sign = (rng.integers(2)-0.5)*2.
    x_angle, y_angle, z_angle = rng.random(size=3)*reach*sign
    R = get_xyz_rotation_matrix(x_angle, y_angle, z_angle)
    xyz = transform_angular_to_euclidean(coordinates)
    new_coordinates = rotate_euclidean_coordinates(xyz, N, R)
    return transform_euclidean_to_angular(new_coordinates), R

@njit
def rms_distance_to_equator(coordinates):
    phi = coordinates.T[1]
    return np.sqrt(np.mean(phi**2))

def project_coordinates_on_circle_with_R(coordinates, R, N):
    xyz = transform_angular_to_euclidean(coordinates)
    new_coordinates = rotate_euclidean_coordinates(xyz, N, R)
    return transform_euclidean_to_angular(new_coordinates)

def project_coordinates_on_circle(coordinates, N, rng, verbose=False):
    rmsd = rms_distance_to_equator(coordinates)
    out = np.copy(coordinates)
    h=[]
    for i in tqdm(range(10000)):
        new_coordinates, R = randomly_rotate_coordinates(coordinates, N, rng, reach=np.pi)
        new_rmsd = rms_distance_to_equator(new_coordinates)
        h.append(new_rmsd)
        if new_rmsd < rmsd:
            out = new_coordinates
            rmsd = new_rmsd
    if verbose:
        #plt.hist(h, bins=40)
        #plt.show()
        print('Final RMS distance to equator is {}'.format(rmsd))
    return new_coordinates.T[0].reshape((N,1)), R

# stuf to be measured

def get_stable_rank(B):
    u, s, vh = np.linalg.svd(B)
    return np.sum(s**2)/(s[0])**2


def get_disparities(weights):
    strengths = np.sum(weights, axis=0)
    num = weights**2
    Y = np.sum(num, axis=0)
    return Y/strengths**2

def get_entropy(B):
    assert abs(np.sum(B)-2.)<1e-4, 'sum of probs is not 1'
    logable_B = np.where(B>0, B, 1)
    hadamard = np.triu(B, k=1)*np.triu(np.log(logable_B), k=1)
    return -np.sum(hadamard)

#plotting functions

def plot_coordinates_S2(S2, ax, n):
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
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.7, linewidth=0, zorder=10)
    for c in range(n):
        color = plt.cm.tab10(c%10)
        nodes = np.where(S2.communities==c)
        ax.scatter(xx[nodes],yy[nodes],zz[nodes],color=color,s=9)
    l=0.85
    ax.set_xlim([-l,l])
    ax.set_ylim([-l,l])
    ax.set_zlim([-l,l])
    ax.axis('off')

def plot_coordinates_S1(S1, ax, n):
    theta = np.mod(S1.coordinates.flatten(), 2*np.pi)
    for c in range(n):
        color = plt.cm.tab10(c%10)
        nodes = np.where(S1.communities==c)
        ax.scatter(theta[nodes],np.ones(len(theta))[nodes],color=color,s=5, alpha=0.3)
    ax.set_ylim(0,1.5)
    ax.axis('off')

def plot_block_matrix(SD, ax, mi, ma):
    im = ax.imshow(np.log10(SD.block_matrix+1e-5), cmap='Greys', vmin=mi, vmax=ma)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

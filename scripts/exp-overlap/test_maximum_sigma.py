#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Tests if RMI can lead to a numerical definition of 
maximum dispersion of communities on spheres

Author: Béatrice Désy

Date : 23/02/2022
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *
from overlap_util import *

from rmi import reduced_mutual_information
from sklearn.cluster import DBSCAN

def get_eps(D, N):
    if D==1:
        vol = 2*np.pi
    elif D==2:
        vol = 4*np.pi
    return vol/N * 4 * D

def clean_noisy_samples(detected_communities):
    i = np.max(detected_communities)
    for j in np.where(detected_communities==-1):
        detected_communities[j]=i+1
    return detected_communities

def detect_angular_labels(N, D, coordinates):
    X = build_angular_distance_matrix(N, coordinates, D, 
                                        euclidean=False, order=None)
    eps = get_eps(D, N)
    clustering = DBSCAN(eps=eps, min_samples=2*D, metric='precomputed').fit(X)
    return clustering.labels_

def get_other_sigma1(nc):
    sigma = np.pi/(2*nc)
    return sigma

def plot_circles(coordinates, labels_a, labels_b, rmi, nb_com):
    N = len(labels_b)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='polar')
    theta = np.mod(coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels_a==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=4, alpha=0.7)
    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.title('coloring from initial sampling')

    ax = fig.add_subplot(122, projection='polar')
    theta = np.mod(coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels_b==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=2, alpha=0.3)

    plt.ylim(0,1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.title('coloring from detection on S1\nRMI={}'.format(rmi))
    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--nb_communities', type=int, default=8,
                            help='number of communities to put on the sphere')
    parser.add_argument('-s', '--sigma', type=float,
                            help='dispersion of points of angular clusters in d=1')
    args = parser.parse_args() 

    N = 1000
    nb_com = args.nb_communities
    if args.sigma is None:
        sigma1 = get_other_sigma1(nb_com)
    else:
        sigma1 = args.sigma
    sigma2 = get_sigma_d(sigma1, 2)

    rng = np.random.default_rng()

    coordinates = [get_coordinates(N, 1, nb_com, sigma1), get_coordinates(N, 2, nb_com, sigma2)]
    sizes = get_equal_communities_sizes(nb_com, N)

    my_communities = get_communities_array(N, sizes)
    detected_communities = detect_angular_labels(N, 1, coordinates[0])
    detected_communities = clean_noisy_samples(detected_communities)
    print(detected_communities, set(detected_communities))
    print(np.sum(np.where(my_communities!=detected_communities, 1, 0)))

    rmi = reduced_mutual_information(N, my_communities, detected_communities)

    plot_circles(coordinates[0], my_communities, detected_communities, rmi, nb_com)




if __name__=='__main__':
    main()

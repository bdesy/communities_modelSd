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
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *

from rmi import reduced_mutual_information

def detect_communities_Sd(N, D, coordinates):
	labels = np.zeros(N).astype(int)
	distance_matrix = build_angular_distance_matrix(N, coordinates, D, 
													euclidean=False, order=None)
	weights = np.pi - distance_matrix
	plt.imshow(weights)
	plt.show()
	return labels

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--nb_communities', type=int, default=8,
                        help='number of communities to put on the sphere')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
args = parser.parse_args() 

def get_other_sigma1(nc):
    sigma = np.pi/(2*nc)
    return sigma

def main():
	N = 1000
	nb_com = args.nb_communities
	if args.sigma is None:
	    sigma1 = get_other_sigma1(nb_com)
	else:
	    sigma1 = args.sigma
	sigma2 = get_sigma_d(sigma1, 2)

	rng = np.random.default_rng()

	coordinates = [get_coordinates(N, 1, nc, sigma1), get_coordinates(N, 2, nc, sigma2)]
	sizes = get_equal_communities_sizes(nb_com, N)

	my_communities = get_communities_array(N, sizes)
	detected_communities = detect_communities_Sd(N, 1, coordinates[0])

	rmi = reduced_mutual_information(N, my_communities, detected_communities)

if __name__=='__main__':
	main()

def plot_circles(coordinates, labels_a, labels_b, rmi):
    ax = fig.add_subplot(121, projection='polar')
    theta = np.mod(coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels_a==c)
        ax.scatter(theta[nodes],np.ones(N)[nodes],color=color,s=2, alpha=0.3)
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


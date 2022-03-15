#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots for community structure experiment in the S1 and S1 model

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
parser.add_argument('-n', '--nb_nodes', type=int,
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

path = 'data/nc-{}-dd-{}-s{}-b{}'.format(args.nb_communities, 
                                        args.degree_distribution, 
                                        str(args.sigma)[0]+str(args.sigma)[2], 
                                        int(args.beta_ratio))

def plot_distance_dist(path):
	S1, S2 = ModelSD(), ModelSD()
	models = [S1, S2]
	S1.load_all_parameters_from_file(path+'/S1-')
	S2.load_all_parameters_from_file(path+'/S2-')
	for D in [1,2]:
		mod = models[D-1]
		mod.build_angular_distance_matrix()
		dist = mod.angular_distance_matrix.flatten()
		plt.hist(dist, bins=50, alpha=0.5, label=D)
	plt.legend()
	plt.show()

plot_distance_dist(path)
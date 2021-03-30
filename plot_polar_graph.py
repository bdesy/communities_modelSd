#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot a network in hyperbolic plane whilst accentuating community structure.

Author: Béatrice Désy

Date : 29/04/2021
"""

import argparse
import numpy as np
import networkx as nx
from infomap import Infomap
import matplotlib.pyplot as plt
import community as community_louvain

# Define sub-recipes

def compute_radius(kappa, kappa_0, R_hat):
    return R_hat - 2*np.log(kappa/kappa_0)
'''
def get_node_coordinates(node, G, kappa_0, R_hat):
    theta = G.nodes(data=True)[node]['angle0']
    kappa = G.nodes(data=True)[node]['kappa']
    r = compute_radius(kappa, kappa_0, R_hat)
    return theta, r

def get_edge_coordinates(edge, G, kappa_0, R_hat):
    v1, v2 = edge
    theta1, r1 = get_node_coordinates(v1, G, kappa_0, R_hat)
    theta2, r2 = get_node_coordinates(v2, G, kappa_0, R_hat)
    thetas = [theta1, theta2]
    rs = [r1, r2]
    return thetas, rs
'''

# Par input parameters

parser = argparse.ArgumentParser()

parser.add_argument('--path', '-p', type=str,
                    help='path to the graph xml file')
parser.add_argument('--community', '-c', type=str,
                    help='community detection algorithm to use')
args = parser.parse_args()

# Load graph data and parameters

G = nx.read_graphml(args.path)
#G = nx.read_graphml('data/graph1000_poisson_gpa_S1_hidvar.xml')

path_to_hidvars = G.graph['hidden_variables_file']
D = G.graph['dimension']
mu = G.graph['mu']
R = G.graph['radius']
beta = G.graph['beta']

hidvars = np.loadtxt(path_to_hidvars, dtype=str).T

kappas_array = (hidvars[1]).astype('float')
thetas_array = (hidvars[2]).astype('float')
N = len(kappas_array)

# Compute radii 

kappa_0 = np.min(kappas_array)
R_hat = 2*np.log(N / (mu*np.pi*kappa_0**2))

radiuses_array = compute_radius(kappas_array, kappa_0, R_hat)

# Set dictionnaries for stuff

kappas = {}
thetas = {}
radiuses = {}

for node in G.nodes():
    kappa = G.nodes(data=True)[node]['kappa']
    
    thetas[node] = G.nodes(data=True)[node]['angle0']
    kappas[node] = kappa
    radiuses[node] = compute_radius(kappa, kappa_0, R_hat)

# Compute desired partition
if args.community=='louvain':
    partition = community_louvain.best_partition(G)
elif args.community=='infomap':
    im = Infomap()
    for node in G.nodes():
        im.add_node(int(node[1:]))
    for edge in G.edges():
        n1, n2 = edge
        im.add_link(int(n1[1:]), int(n2[1:]))
    im.run("-N10")
    partition_im = im.get_modules(depth_level=2)
    partition = {}
    for node_int in partition_im:
        node = 'n{}'.format(node_int)
        partition[node] = partition_im[node_int]

# Create color dictionnary

color_list=['teal', 'coral', 'limegreen', 'crimson', 'midnightblue', 'turquoise', 'plum', 'darkorchid', 'indigo', 'darkslategrey', 
            'dimgray', 'darkgreen',  'peru', 'greenyellow', 'saddlebrown']

colors = {}
for comm in set(partition.values()):
    colors[comm] = color_list[comm]

# Plot figure

plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='polar')

for edge in G.edges():
    n1, n2 = edge
    theta, r = [thetas[n1], thetas[n2]], [radiuses[n1], radiuses[n2]]
    ax.plot(theta, r, c='k', linewidth=0.5, alpha=0.2)

ax.plot(thetas.values(), radiuses.values(), 'o', ms=3, c='white')
for node in G.nodes():
    community = partition[node]
    color = colors[community]
    ax.plot(thetas[node], radiuses[node], 'o', ms=2, c=color)

ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['polar'].set_visible(False)
ax.set_ylim(np.min(radiuses_array)/2, np.max(radiuses_array))
plt.tight_layout()
plt.show()
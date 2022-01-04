#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot a network in hyperbolic plane whilst accentuating community structure and node densities.

Author: Béatrice Désy

Date : 29/04/2021
"""

import argparse
import matplotlib
import numpy as np
import networkx as nx
from infomap import Infomap
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import community as community_louvain

# Define sub-recipes

def compute_radius(kappa, kappa_0, R_hat):
    return R_hat - 2*np.log(kappa/kappa_0)

# Parse input parameters

parser = argparse.ArgumentParser()

parser.add_argument('--filepath', '-p', type=str,
                    help='path to the graph xml file')
parser.add_argument('--community', '-c', type=str, choices=['louvain', 'infomap', 'SBM'],
                    help='community detection algorithm to use')
parser.add_argument('--mode', '-m', type=str, default='normal',
                    help='optional presentation mode for bigger stuff')
parser.add_argument('--density', '-d', type=bool, default=False,
                    help='to turn off angular density plot on the circumference')
parser.add_argument('--save', '-s', type=bool, default=False,
                    help='to save or not the figure')
args = parser.parse_args()

# Load graph data and parameters

G = nx.read_graphml(args.filepath)
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
    partition_im = im.get_modules(depth_level=1)
    partition = {}
    for node_int in partition_im:
        node = 'n{}'.format(node_int)
        partition[node] = partition_im[node_int]
elif args.community=='SBM':
    import graph_tool.all as gt
    g = gt.load_graph(args.filepath)
    state = gt.minimize_blockmodel_dl(g)
    partition_sbm = state.get_blocks()
    partition = {}
    for i in range(N):
        node = 'n{}'.format(i)
        partition[node] = partition_sbm[i]



# Create color dictionnary
print(len(set(partition.values())))
color_list=['teal', 'coral', 'limegreen', 'crimson', 'midnightblue', 'turquoise', 'plum', 'darkorchid', 'indigo', 'darkslategrey', 
            'dimgray', 'darkgreen',  'peru', 'greenyellow', 'saddlebrown', 'teal', 'coral', 'limegreen', 'crimson', 'midnightblue',
             'turquoise', 'plum', 'darkorchid', 'indigo', 'darkslategrey', 
            'dimgray', 'darkgreen',  'peru', 'greenyellow', 'saddlebrown',
             'turquoise', 'plum', 'darkorchid', 'indigo', 'darkslategrey', 
            'dimgray', 'darkgreen',  'peru', 'greenyellow', 'saddlebrown',]

colors = {}
for comm in set(partition.values()):
    colors[comm] = color_list[comm]

# Set plotting sizes
if args.mode=='normal':
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    ms=[5,3]
elif args.mode=='presentation':
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    ms=[10,8]

# Plot figure

fig = plt.figure(figsize=(6,6))
rect = [0.1, 0.1, 0.8, 0.8]
ax = fig.add_axes(rect, projection='polar')

for edge in G.edges():
    n1, n2 = edge
    theta, r = [thetas[n1], thetas[n2]], [radiuses[n1], radiuses[n2]]
    ax.plot(theta, r, c='k', linewidth=0.5, alpha=0.4)

for node in G.nodes():
    community = partition[node]
    color = colors[community]
    ax.plot(thetas[node], radiuses[node], 'o', ms=ms[0], c='white')
    ax.plot(thetas[node], radiuses[node], 'o', ms=ms[1], c=color)

ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)

if args.density:
    tt = np.linspace(0.001,2*np.pi, 1000)
    kde = gaussian_kde(thetas_array, bw_method=0.02)
    upper = kde(tt)
    upper /= np.max(upper)
    upper *= (R_hat*1.1 - R_hat)
    upper += R_hat
    lower = np.ones(1000)*R_hat
    ax.fill_between(tt, lower, upper, alpha=0.3, color='darkcyan')
    ax.plot(tt, upper, c='darkcyan', linewidth=1)

    xx = np.linspace(0.01, R_hat, 1000)
    kde_r = gaussian_kde(radiuses_array, bw_method=0.07)
    yy = np.array(kde_r(xx))
    yy = yy/np.max(yy)*R_hat/3

    r_den = np.sqrt(xx**2 + yy**2)
    th_den = np.arccos(xx / r_den)
    print(len(th_den), th_den.shape)
    print(type(r_den), type(xx), type(yy))
    
    #ax.plot(th_den, r_den, '-', c='darkcyan', ms=0.5)
    #ax.fill_between(th_den, 0, r_den, alpha=0.3, color='darkcyan')
    plt.ylim(0.01, R_hat*1.2)
if args.save:
    plt.savefig('fig1_'+args.community, dpi=600)

plt.show()




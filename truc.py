#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Various graph measures implemented simpy and to be used with Numba 

Author: Béatrice Désy

Date : 31/07/2020
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import networkx as nx


def initialize(path_to_G, space):
    if type(path_to_G) == str:
        G = nx.read_weighted_edgelist(path_to_G, nodetype=int)
    elif type(path_to_G) == nx.classes.graph.Graph:
        G = path_to_G
    else:
        print('wrong input graph, either filepath to edge list or networkx graph')
    
    A = nx.to_numpy_matrix(G)
    if space=='multigraph':
        elist_str = [line.split(' ') for line in nx.generate_edgelist(G, data=['weight'])]
        multiple_edge_list = [[int(e[0]), int(e[1]), int(float(e[2]))] for e in elist_str]
        edge_list = []
        for e in multiple_edge_list:
            edge_list += [[e[0], e[1]]]*e[2]
        A = np.where(A>0.5, A, 0)
    
    elif space=='simple':
        A = np.where(A>1, 1, A)
        G = nx.Graph(G)
        edge_list = [list(map(int, line.split(' '))) for line in nx.generate_edgelist(G, data=False)]
    else:
        print('wrong space, either simple or multigraph')

    m = len(edge_list)
    n = A.shape[0]
    
    # index between nodes labels and position in adjacency matrix
    nodes_ind = np.array(list(G.nodes))
    index_A = np.zeros(np.max(nodes_ind)+1, dtype=int)
    for i in range(len(nodes_ind)):
        node = nodes_ind[i]
        index_A[node] = i

    return G, np.array(edge_list), A, m, n, index_A

@njit
def adjacency_to_edgelist(A, n, loopy):
    edgelist=[]
    for i in range(n):
        for j in range(i+int(loopy)):
            if A[i][j]>0.5:
                for ell in range(int(A[i][j])):
                    edgelist.append([i,j])
    return np.array(edgelist)

@njit
def nb_edges_from_adjacency(A):
    return np.sum(np.triu(A, k=0))

@njit
def measure_assortativity(edgelist, m, A, index):
    k = np.sum(A, axis=1)
    degdeg, mu, var = 0, 0, 0
    for i in range(m):
        ku, kv = k[index[int(edgelist[i, 0])]], k[index[int(edgelist[i, 1])]]
        degdeg += ku*kv
        mu += ku + kv
        var += ku**2 + kv**2
    mu /= (2*m)
    sigma = var/(2*m) - mu**2
    if sigma > 1.e-5:
        out = (degdeg/m - mu**2)/sigma
    else:
        out=0
    return out

@njit
def measure_nb_self_loops(A):
    return np.trace(A)

@njit
def measure_nb_multiedges(A):
    multi = np.triu(np.where(A>1, 1, 0))
    return np.sum(multi)

@njit
def measure_transitivity(A, n):
    triangles = 0
    triplets = 0
    k = np.sum(np.where(A>=0.5, 1., 0.), axis=1)
    triplets = np.sum(k*(k-1))
    for i in range(n):
        for j in range(i):  
            if A[i,j]>=0.5:
                for ell in range(n):
                    if A[j,ell]>=0.5 and A[i, ell]>=0.5:
                        triangles += 1
    triplets/=2
    if triplets > 1.e-5:
        out = triangles/triplets
    else:
        out = 0
    return out
    
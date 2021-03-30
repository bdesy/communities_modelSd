#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot a network in hyperbolic plane whilst accentuating community structure.

Author: Béatrice Désy

Date : 29/04/2021
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml('data/graph1000_poisson_gpa_S1_hidvar.xml')

path_to_hidvars = G.graph['hidden_variables_file']
D = G.graph['dimension']
mu = G.graph['mu']
R = G.graph['radius']
beta = G.graph['beta']

hidvars = np.loadtxt(path_to_hidvars, dtype=str).T

kappas = (hidvars[1]).astype('float')
thetas = (hidvars[2]).astype('float')
N = len(kappas)

kappa_0 = np.min(kappas)
R_hat = 2*np.log(N / (mu*np.pi*kappa_0**2))

radiuses = R_hat - 2*np.log(kappas/kappa_0)

plt.polar(thetas, radiuses, 'o', ms=2, c='darkcyan')
plt.show()
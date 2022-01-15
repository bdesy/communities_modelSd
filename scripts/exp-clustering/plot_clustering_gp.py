#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plot clustering figure 2.6b from Garcia-Perez PhD thesis

Author: Béatrice Désy

Date : 14/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
rc('font',**{'serif':['Computer Modern Roman'], 'size':14})
rc('text', usetex=True)

dimensions = np.arange(1,11)
gammas = [2.1, 2.325, 2.55, 2.775, 3.0]
markers = ['s', '^', 'D', '*', 's']
colors = ['blueviolet', 'dodgerblue', 'turquoise', 'lightgreen', 'darkorange']

with open('figure2', 'rb') as file:
    res = pickle.load(file)

plt.figure(figsize=(7,5))
i=0
for y in gammas:
    data = []
    for D in dimensions:
        key = 'S{}_gamma{}'.format(D, y)
        data.append(res[key])
    data = np.array(data)
    plt.errorbar(dimensions, data.T[0], 
    		yerr=data.T[1], elinewidth=1., capsize=2.,
            marker=markers[i], mec='k', mew=0.4,
            c=colors[i], 
            linewidth=2,
            label=r'$\gamma={}$'.format(y))
    i+=1
plt.legend()
plt.ylim(0.2, 0.9)
plt.xlim(1, 10)
plt.xticks(dimensions.tolist())
plt.ylabel(r'$\bar{c}_{\mathrm{max}}$')
plt.xlabel(r'$D$')
plt.show()






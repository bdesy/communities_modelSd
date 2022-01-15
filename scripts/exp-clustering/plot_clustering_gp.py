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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

dimensions = np.arange(1,11)
gammas = [2.1, 2.325, 2.55, 2.775, 3.0]
markers = ['s', '^', 'D', '*', 's']
colors = ['blueviolet', 'dodgerblue', 'turquoise', 'lightgreen', 'darkorange']

with open('test', 'rb') as file:
    res = pickle.load(file)

i=0
for y in gammas:
    data = []
    for D in dimensions:
        key = 'S{}_gamma{}'.format(D, y)
        data.append(res[key])
    plt.plot(dimensions, data, 
            marker=markers[i], mec='k', mew=0.4,
            c=colors[i], 
            linewidth=2,
            label=r'$\gamma={}$'.format(y))
    i+=1
plt.legend()
plt.ylim(0.2, 0.9)
plt.xlim(1, 10)
plt.xticks(dimensions.tolist())
#plt.ylabel(r'$\bar{c}_{\text{max}}$')
plt.xlabel(r'D')
plt.show()






#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure overlap experiment run in the S1 and S1 model

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import json
import matplotlib.pyplot as plt

from overlap_run import get_dict_key

nc_list = [5, 15, 25]
dd_list = ['exp', 'pwl']
beta_ratio_list = [3.5]
sigma_axis = np.linspace(0.01, 0.3, 10)

with open('data/sample10.json', 'r') as read_file:
    data_dict = json.load(read_file)

colors = [plt.cm.viridis(0.2), plt.cm.viridis(0.5), plt.cm.viridis(0.8)]
formats = [0, '--', '-']

for qty in ['r', 'Y']:
    for br in beta_ratio_list:
        for dd in dd_list:
            plt.figure(figsize=(7,5))
            for c in range(len(nc_list)):
                nc = nc_list[c]
                for D in [1,2]:
                    beta = br*D
                    y, err = [], []
                    for sigma in sigma_axis:
                        key = get_dict_key(D, dd, nc, beta, sigma)+'-'+qty
                        data = np.array(data_dict[key])
                        y.append(np.mean(data))
                        err.append(np.std(data))
                    plt.errorbar(sigma_axis, y, yerr=err, fmt=formats[D],
                                     capsize=2, elinewidth=1, capthick=1, 
                                     color=colors[c], label=r'$S^{}$, $n_c={}$'.format(D, nc))
            plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
            plt.ylabel(qty)
            plt.xlabel(r'$\sigma$ (en unités de $\pi$)')
            plt.legend()
            plt.show()


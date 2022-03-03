#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure overlap experiment run in the S1 and S1 model

Author: Béatrice Désy

Date : 03/02/2022
"""


import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

from overlap_run import get_dict_key, get_sigma_max

font = {'size'   : 13, 
    'family': 'serif'}

matplotlib.rc('font', **font)

nc_list = [5, 15, 25]
dd = 'exp'
br = 3.5
frac_sigma_axis = np.linspace(0.1, 0.9, 10)

with open('data/sample10_allbeta_fracsigma.json', 'r') as read_file:
    data_dict = json.load(read_file)

cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.), cmap(2.2/5)]
formats = [':', '--', '-']

bidon = np.linspace(100, 110, 1000)

qty = 'Y'
plt.figure(figsize=(5.5, 5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    fmt = formats[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = [], []
        for f in frac_sigma_axis:
            key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
            data = np.array(data_dict[key])
            y.append(np.mean(data))
            err.append(np.std(data))
        y = np.array(y)
        err = np.array(err)
        y *= nc
        err *= nc
        if nc == 25:
            lab = label=r'$S^{}$'.format(D)
        else:
            lab=None

        plt.errorbar(frac_sigma_axis, y, yerr=err, fmt=fmt,
                         capsize=2, elinewidth=1, capthick=1, 
                         color=colors[D-1], label=lab)
    plt.plot(bidon, np.ones(bidon.shape), 
                fmt, c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))
#plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel('average disparity')
plt.xlabel(r'angular dispersion')
plt.xlim(0.05,0.95)
plt.legend(loc=1, ncol=2)
#plt.savefig('figures/sample10_allbeta_fracsigma/beta{}-'.format(br)+dd+'-'+qty+'.png')
plt.show()

'''
qty = 'r'
plt.figure(figsize=(7,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = [], []
        for f in frac_sigma_axis:
            key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
            data = np.array(data_dict[key])
            y.append(np.mean(data))
            err.append(np.std(data))
        y = np.array(y)
        err = np.array(err)
        if qty =='r':
            y /= nc
            err /= nc
        elif qty=='Y':
            y *= nc
            err *= nc
        plt.errorbar(frac_sigma_axis, y, yerr=err, fmt=formats[D],
                         capsize=2, elinewidth=1, capthick=1, 
                         color=colors[c], label=r'$S^{}$, $n={}$'.format(D, nc))
plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel('r/n')
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0,1)
plt.legend()
#plt.savefig('figures/sample10_allbeta_fracsigma/beta{}-'.format(br)+dd+'-'+qty+'.png')
plt.show()
'''

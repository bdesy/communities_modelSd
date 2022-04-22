#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plot for the article

Author: Béatrice Désy

Date : 07/04/2022
"""


import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

from overlap_run import get_sigma_max
from overlap_util import *

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)

nc_list = [5, 15,25]
dd = 'exp'
br = 3.5
frac_sigma_axis = np.linspace(0.05, 0.95, 20)

with open('data/experiment_entropy.json', 'r') as read_file:
    data_dict = json.load(read_file)

cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.), cmap(2.2/5)]
formats = [':', '--', '-']

bidon = np.linspace(100, 110, 1000)

qty='S'
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
for c in range(len(nc_list)):
    nc = nc_list[c]
    plt.plot(bidon, np.ones(bidon.shape), 
                formats[c], c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty)
        if nc==25:
            lab = r'$D = {}$'.format(D)
        else:
            lab=None
        plt.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=lab)
        plt.fill_between(frac_sigma_axis, y-err, y+err, alpha=0.3, color=colors[D-1], linewidth=0.0)

plt.ylabel(r'$S$')
plt.xlabel(r'$\sigma$')
plt.xlim(0.05,0.95)
plt.ylim(0.5, 5.2)
#plt.legend(ncol=2, loc=4)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figures/article-'+qty+'.png', dpi=600)
plt.show()

qty='r'
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot()
for c in range(len(nc_list)):
    nc = nc_list[c]
    plt.plot(bidon, np.ones(bidon.shape), 
                formats[c], c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty)
        if nc==25:
            lab = r'$D = {}$'.format(D)
        else:
            lab=None
        plt.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=lab)
        plt.fill_between(frac_sigma_axis, y-err, y+err, alpha=0.3, color=colors[D-1], linewidth=0.0)

plt.ylabel(r'$r$')
plt.xlabel(r'$\sigma$')
plt.xlim(0.05,0.95)
plt.ylim(0.,0.7)
plt.tight_layout()
plt.legend(ncol=2, frameon=False, loc=(0.05, 0.75),columnspacing=1.)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figures/article-'+qty+'.png', dpi=600)
plt.show()

qty='degrees'
fig = plt.figure(figsize=(3.375, 3))
ax = fig.add_subplot()
for c in range(len(nc_list)):
    nc = nc_list[c]
    plt.plot(bidon, np.ones(bidon.shape), 
                formats[c], c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty, False)
        y_mean = np.mean(y, axis=(1,2))
        err = np.std(y, axis=(1,2))
        if nc==25:
            lab = r'$D = {}$'.format(D)
        else:
            lab=None
        plt.plot(frac_sigma_axis, y_mean, linestyle=formats[c], color=colors[D-1], 
                label=lab)

        plt.fill_between(frac_sigma_axis, y_mean-err, y_mean+err, 
                        alpha=0.3, color=colors[D-1], linewidth=0.0)

#plt.title('block matrix <k>\n'+r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel(r'$<k>$')
plt.xlabel(r'$\sigma$')
plt.xlim(0.05,0.95)
plt.ylim(0, 21)
plt.legend(ncol=2, frameon=False, columnspacing=1.)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figures/article-'+qty+'.png')
plt.show()

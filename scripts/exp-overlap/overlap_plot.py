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
import matplotlib

from overlap_run import get_dict_key, get_sigma_max
from overlap_util import *

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=12)

nc_list = [5,15,25]
dd = 'exp'
br = 3.5
frac_sigma_axis = np.linspace(0.05, 0.95, 20)

o_fsa = np.linspace(0.1, 0.9, 10)

with open('data/experiment0604.json', 'r') as read_file:
    data_dict = json.load(read_file)
with open('data/experiment_entropy.json', 'r') as read_file:
    entropy_data_dict = json.load(read_file)

cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.), cmap(2.2/5)]
formats = [':', '--', '-']
m = ['s', '^']
ms = [5.5, 6.]

qty='Y'
plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty, closest=True)
        y_mean = np.mean(y, axis=(1,2))
        err_mean = np.std(y, axis=(1,2))

        plt.plot(frac_sigma_axis, y_mean, linestyle=formats[c], color=colors[D-1], 
                label=r'$S^{}$, $n={}$'.format(D, nc))
        plt.fill_between(frac_sigma_axis, y_mean-err_mean, y_mean+err_mean, alpha=0.3, color=colors[D-1])
        
plt.title('arithmetic mean of disparities\n B/D={}, '.format(br)+dd+' degree distribution')
plt.ylabel(qty)
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.legend()
#plt.savefig('figures/experiment0604'+qty+'.png')
plt.show()


qty='r'
plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty, True)
        
        plt.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=r'$S^{}$, $n={}$'.format(D, nc))

        plt.fill_between(frac_sigma_axis, y-err, y+err, alpha=0.3, color=colors[D-1])

plt.title('block matrix stable rank\n'+r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel('r')
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.0,1.0)
plt.legend()
#plt.savefig('figures/experiment0604'+qty+'.png')
plt.show()

qty='m'

plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, qty, True)
        plt.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=r'$S^{}$, $n={}$'.format(D, nc))
        plt.fill_between(frac_sigma_axis, y-err, y+err, alpha=0.3, color=colors[D-1])

plt.title('number of inter-community edges\n'+r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel(qty)
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.legend(loc=4)
#plt.savefig('figures/experiment0604'+qty+'.png')
plt.show()

qty='S'
plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(entropy_data_dict, D, dd, nc, beta, frac_sigma_axis, qty, False)

        plt.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=r'$S^{}$, $n={}$'.format(D, nc))

        plt.fill_between(frac_sigma_axis, y-err, y+err, alpha=0.3, color=colors[D-1])

plt.title('block matrix entropy\n'+r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel('S')
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.0,1.0)
plt.legend()
#plt.savefig('figures/experiment0604'+qty+'.png')
plt.show()


bidon = np.linspace(100, 110, 1000)

qty='degrees'
plt.figure(figsize=(5.5, 5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    plt.plot(bidon, np.ones(bidon.shape), 
                formats[c], c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = retrieve_data(entropy_data_dict, D, dd, nc, beta, frac_sigma_axis, qty, False)
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
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.ylim(0, 22)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
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

nc_list = [5, 15, 25]
dd = 'exp'
br = 3.5
frac_sigma_axis = np.linspace(0.1, 0.9, 10)

with open('data/sample50_fracsigma_abstract.json', 'r') as read_file:
    data_dict = json.load(read_file)

cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.), cmap(2.2/5)]
formats = [':', '--', '-']
m = ['s', '^']
ms = [5.5, 6.]

def transform_Y(Y, n):
    Y = np.array(Y)
    Y = Y - 1./(n)
    Y /= (1.-1./n)
    return Y

qty='Y'
plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y = np.zeros((50, len(frac_sigma_axis)))
        j=0
        for f in frac_sigma_axis:
            key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
            data = np.array(data_dict[key])
            y[:,j] = data[:]
            j+=1
        for i in range(len(data)):
            plt.plot(frac_sigma_axis, y[i, :],
                    linewidth=1, alpha=0.1, c='grey')
        y, err = np.mean(y, axis=0), np.std(y, axis=0)
        #y = transform_Y(y, nc)
        plt.errorbar(frac_sigma_axis, y, yerr=err, fmt=formats[c],
                         capsize=2, elinewidth=1, capthick=1, 
                         color=colors[D-1], label=r'$S^{}$, $n_c={}$'.format(D, nc))
plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel(qty)
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.legend()
#plt.savefig('figures/sample10_allbeta_fracsigma/beta{}-'.format(br)+dd+'-'+qty+'.png')
plt.show()

qty='Y'
plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y = np.zeros((len(data), len(frac_sigma_axis)))
        j=0
        for f in frac_sigma_axis:
            key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
            data = np.array(data_dict[key])
            y[:,j] = data[:]
            j+=1
        for i in range(len(data)):
            if i==1:
                plt.plot(frac_sigma_axis, y[i, :], linestyle=formats[c],
                    linewidth=1, alpha=0.2, c=colors[D-1], 
                    label=r'$S^{}$, $n_c={}$'.format(D, nc))
            else:
                plt.plot(frac_sigma_axis, y[i, :], linewidth=1, alpha=0.2, c=colors[D-1])

plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel(qty)
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.legend()
#plt.savefig('figures/sample10_allbeta_fracsigma/beta{}-'.format(br)+dd+'-'+qty+'.png')
plt.show()

def r_toeplitz(n):
    res = 0.
    for i in range(n):
        arg = (i+1)*np.pi/(n+1)
        res += np.cos(arg)**2
    arg = np.pi/(n+1)
    return res / np.cos(arg)**2

qty='r'

def transform_r(r, n):
    r_max = r_toeplitz(n)/n
    r -= 1./n
    r /= (r_max - 1./n)
    return r 

plt.figure(figsize=(5.5,5))
for c in range(len(nc_list)):
    nc = nc_list[c]
    for D in [1,2]:
        sigma_max = get_sigma_max(nc, D)
        beta = br*D
        y, err = [], []
        for f in frac_sigma_axis:
            key = get_dict_key(D, dd, nc, beta, f)+'-'+qty
            data = np.array(data_dict[key])
            y.append(np.mean(data/nc))
            err.append(np.std(data/nc))
        y = np.array(y)
        err = np.array(err)
        #y = transform_r(y, nc)
        plt.errorbar(frac_sigma_axis, y, yerr=err, fmt=formats[c],
                         capsize=2, elinewidth=1, capthick=1, 
                         color=colors[D-1], label=r'$S^{}$, $n={}$'.format(D, nc))
    #plt.axhline(r_toeplitz(nc)/nc, linestyle=formats[c], c='k', alpha=0.2)
plt.title(r'$\beta/D={}, $'.format(br)+dd+' degree distribution')
plt.ylabel(qty)
plt.xlabel(r'$\sigma/\sigma_m$')
plt.xlim(0.05,0.95)
plt.legend()
#plt.savefig('figures/sample10_allbeta_fracsigma/beta{}-'.format(br)+dd+'-'+qty+'.png')
plt.show()


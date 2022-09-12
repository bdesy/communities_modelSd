#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Test various binarization for computing community degree

Author: Béatrice Désy

Date : 03/05/2022
"""


import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)

#import sys
#sys.path.insert(0, '../../src/')
#from hyperbolic_random_graph import *
#from hrg_functions import *
#from geometric_functions import *
#from overlap_util import *

def get_dict_key(D, dd, nc, beta, f):
    return 'S{}-'.format(D)+dd+'-{}coms-{}beta-{:.2f}sigmam'.format(nc, beta, f)

def retrieve_data(data_dict, D, dd, nc, beta, frac_sigma_axis, bins):
    y = []
    for f in frac_sigma_axis:
        key = get_dict_key(D, dd, nc, beta, f)+'-degrees-'+bins
        data = np.array(data_dict[key])
        y.append(data)
    y_data = np.array(y)
    if False:#bins=='backbone':
        y = np.mean(np.max(y_data, axis=2), axis=1)
        err = np.std(np.max(y_data, axis=2), axis=1)
    else:
        y = np.mean(np.mean(y_data, axis=2), axis=1)
        err = np.std(np.mean(y_data, axis=2), axis=1)
    return y, err

def measure_community_degrees_gthreshold(matrices_list, t):
    data = []
    for m in matrices_list:
        m = np.array(m)
        binary_mat = np.where(m>t, 1, 0)
        degrees = list(np.sum(binary_mat, axis=0).astype(float))
        data.append(degrees)
    return data

def measure_community_degrees_proportional(matrices_list):
    pass

def measure_community_degrees_backbone(matrices_list, alpha, show=False):
    data = []
    for m in matrices_list:
        mat = 2*np.array(m)/np.sum(m)
        binary_mat = binarize_using_backbone_method(mat, alpha)
        degrees = list(np.sum(binary_mat, axis=0).astype(float))
        data.append(degrees)
    if show:
        plt.imshow(binary_mat)
        plt.colorbar
        plt.show()
    return data

def binarize_using_backbone_method(m, alpha):
    n = m.shape[0]
    binary_mat = np.zeros(m.shape)
    for i in range(n):
        for j in range(i):
            p_ij = m[i,j] / np.sum(m[i, :])
            binary_mat[i,j] = int((1-p_ij)**(n-2) < alpha)
    out = binary_mat + binary_mat.T
    assert np.max(out)<1.1, 'max is greater than 1'
    return out

dd='pwl'
frac_sigma_axis = np.linspace(0.05, 0.95, 30)
beta_r = 3.5
nc_list = [5,15,25]

t = 10

compute = True
if compute:
    with open('data/experiment_entropy_pwl_deg4_blockmatrices.json', 'r') as read_file:
        matrices_dict = json.load(read_file)
    with open('data/experiment_entropy_pwl_deg4.json', 'r') as read_file:
        initial_data_dict = json.load(read_file)

    
    res = {}
    for D in [1,2]:
        beta = beta_r*D
        for nc in nc_list:
            for f in frac_sigma_axis:
                key = get_dict_key(D, dd, nc, beta, f)
                dist_gthreshold = measure_community_degrees_gthreshold(matrices_dict[key], t)
                #print(key)
                dist_backbone = measure_community_degrees_backbone(matrices_dict[key], 0.2, show=False)
                key+='-degrees'
                dist_first = initial_data_dict[key]
                res[key+'-first'] = dist_first
                res[key+'-gthreshold'+str(int(t))] = dist_gthreshold
                res[key+'-backbone'] = dist_backbone
    
    #with open('data/community_degrees_various_binarizations.json', 'w') as write_file:
    #    json.dump(res, write_file, indent=4)

else:
    with open('data/community_degrees_various_binarizations.json', 'r') as read_file:
        data_dict = json.load(read_file)

cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.), cmap(2.2/5)]
formats = [':', '--', '-']
bidon = np.linspace(100, 110, 1000)

sb = [121, 122]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 3.), sharey=True)
i=0
for bins in ['gthreshold'+str(int(t)), 'backbone']:
    ax = axes[i]
    for c in range(len(nc_list)):
        nc = nc_list[c]
        ax.plot(bidon, np.ones(bidon.shape), 
                    formats[c], c='k', alpha=0.5,
                    label=r'$n = {}$'.format(nc))
        for D in [1,2]:
            beta = beta_r*D
            y, err = retrieve_data(res, D, dd, nc, beta, frac_sigma_axis, bins)

            if nc==25:
                lab = r'$D = {}$'.format(D)
            else:
                lab=None

            ax.plot(frac_sigma_axis, y, linestyle=formats[c], color=colors[D-1], 
                label=lab)
            ax.fill_between(frac_sigma_axis, y-err, y+err, 
                        alpha=0.3, color=colors[D-1], linewidth=0.0)
            ax.set_xlabel(r'$\sigma$')
            ax.set_xlim(0.05, 0.95)
    i+=1

axes[0].set_ylabel(r'$\langle k\rangle$')
axes[0].legend(loc=(0.053, 0.532), frameon=False)
plt.ylim(0,7.5)
plt.tight_layout()
axes[0].set_rasterized(True)
axes[1].set_rasterized(True)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.savefig('figures/binarization.eps', dpi=600, format='eps')
plt.show()

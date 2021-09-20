#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots a curve of the modularity and the average realised 
effective distance with beta and x-axis and various degree distributions
and dimensions 1 and 2.

Author: Béatrice Désy

Date : 28/07/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

font = {'size'   : 18}

matplotlib.rc('font', **font)

dico_chi = np.load('data/dico_average_chi_uniform3.npy', allow_pickle=True).flat[0]
dico_dt = np.load('data/dico_average_angular_distance_uniform3.npy', allow_pickle=True).flat[0]
betas =  [1.1, 2., 3., 4., 5., 10.,15.]

colors={'poissonS1':'yellowgreen', 'poissonS2':'coral', 
        'expS1':'g',  'expS2':'firebrick', 'poissonS3':'dodgerblue',  'expS3':'darkblue', }


plt.figure(figsize=(8,6))

for dd in ['poisson', 'exp']:
    for model in ['S1', 'S2']:
        key = dd+model
        data = dico_chi[key]
        means, stds = np.mean(data, axis=1), np.std(data, axis=1)
        plt.plot(betas, means, label=key, color=colors[key])
        #plt.fill_between(betas, means+stds, means-stds, color=colors[key], alpha=0.5)
    for model in ['S3']:
        key = dd+model
        data = dico_chi[key]
        means, stds = np.mean(data, axis=1), np.std(data, axis=1)
        plt.plot(betas, means, '-.', label=key, color=colors[key])

plt.ylabel(r'average $\chi$ of existing edges')
plt.xlabel(r'$\beta$')
plt.ylim(0,6)
plt.xlim(1, 10)
plt.legend()
plt.show()

plt.figure(figsize=(8,6))

for dd in ['poisson', 'exp']:
    for model in ['S1', 'S2']:
        key = dd+model
        data = dico_dt[key]
        means, stds = np.mean(data, axis=1), np.std(data, axis=1)
        plt.plot(betas, means, label=key, color=colors[key])
        #plt.fill_between(betas, means+stds, means-stds, color=colors[key], alpha=0.5)
    for model in ['S3']:
        key = dd+model
        data = dico_dt[key]
        means, stds = np.mean(data, axis=1), np.std(data, axis=1)
        plt.plot(betas, means, '-.', label=key, color=colors[key])

plt.ylabel(r'average $\Delta\theta$ of existing edges')
plt.xlabel(r'$\beta$')
plt.ylim(0,1.2)
plt.xlim(1, 10)
plt.legend()
plt.show()

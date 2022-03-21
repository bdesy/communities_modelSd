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
from mpl_toolkits.mplot3d import Axes3D

from overlap_run import get_dict_key, get_sigma_max
from overlap_util import get_coordinates

font = {'size'   : 13, 
    'family': 'serif'}

matplotlib.rc('font', **font)

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
                         elinewidth=0.7, alpha=0.7,
                         marker=m[D-1], ms=ms[D-1],
                         color=colors[D-1], label=lab)
    plt.plot(bidon, np.ones(bidon.shape), 
                fmt, c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))

plt.ylabel('average disparity')
plt.xlabel(r'angular dispersion')
plt.xlim(0.05,0.95)
plt.ylim(0.35, 1.62)
plt.legend(loc=1)
plt.savefig('figure_disparities_empty', dpi=600)
plt.show()


qty = 'r'
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
                         elinewidth=0.7, alpha=0.7,
                         marker=m[D-1], ms=ms[D-1],
                         color=colors[D-1], label=lab)
    plt.plot(bidon, np.ones(bidon.shape), 
                fmt, c='k', alpha=0.5,
                label=r'$n = {}$'.format(nc))

plt.ylabel('average disparity')
plt.xlabel(r'angular dispersion')
plt.xlim(0.05,0.95)
plt.ylim(0.35, 1.62)
plt.legend(loc=1)
plt.savefig('figure_disparities_empty', dpi=600)
plt.show()


schema = False
N=300
s=0.04
ms = 4

if schema:
    thetas = get_coordinates(N, 1, 20, s)
    circ = np.linspace(0, 2*np.pi, 1500)
    plt.figure(figsize=(3,3))
    plt.polar(circ, np.ones(circ.shape), c='k', alpha=0.3)
    plt.polar(thetas, np.ones(thetas.shape), 'o', c='white', ms=ms+2)
    plt.polar(thetas, np.ones(thetas.shape), 'o', c=cmap(1./20), ms=ms, alpha=0.4)
    plt.axis('off')
    plt.savefig('circle', transparent=True, dpi=600)
    plt.show()

    coordinates = get_coordinates(N, 2, 20, s+0.03)

    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    theta, phi = coordinates.T[0], coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='silver', alpha=0.2, linewidth=0, zorder=0)
    ax.scatter(xx,yy,zz,color='white',s=16, zorder=8)
    ax.scatter(xx,yy,zz,color=cmap(1.1/3),s=10, zorder=10)
    l=0.8
    ax.set_xlim(-l,l)
    ax.set_ylim(-l,l)
    ax.set_zlim(-l,l)
    plt.axis('off')
    plt.savefig('sphere', transparent=True, dpi=600)
    plt.show()


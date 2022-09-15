#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Tests if RMI can lead to a numerical definition of 
maximum dispersion of communities on spheres

Author: Béatrice Désy

Date : 23/02/2022
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *
from util import * 

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=14)

def get_max_sigma1_unif(n):
    sigma = np.sqrt(2*np.pi)/n
    return sigma

def get_max_sigma1_mixed(n):
    sigma = np.pi/n
    return sigma

def plot_circle(ax, coordinates, labels, nb_com, N):
    theta = np.mod(coordinates.flatten(), 2*np.pi)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels==c)
        ax.scatter(theta[nodes], np.ones(N)[nodes],
                    color=color, s=3, alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_ylim(0, 1.5)

def plot_sphere(ax, coordinates, labels, nb_com, N):
    #the sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    #points on the sphere
    theta, phi = coordinates.T[0], coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    #plot sphere
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.3, linewidth=0)
    for c in range(nb_com):
        color = plt.cm.tab10(c%10)
        nodes = np.where(labels==c)
        ax.scatter(xx[nodes],yy[nodes],zz[nodes],color=color,s=3)
    ax.set_xlim([-1.,1.])
    ax.set_ylim([-1.,1.])
    ax.set_zlim([-1.,1.])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

def sum_vmf(x, n):
    s = np.exp(1./x**2)
    if int(n)%2==0:
        for i in range(1, int(n/2)):
            s += 2*np.exp(np.cos( 2*np.pi*i / n ) / x**2)
        s += np.exp(-1./x**2)
    else:
        for i in range(1, int((n+1)/2)):
            s += 2*np.exp(np.cos( 2*np.pi*i / n ) / x**2)
    return s

def vmf_zero(x, n):
    return sum_vmf(x, n) / (n*np.i0(1./x**2)) - 1

def get_max_sigma_vmf_1D(n):
    res = fsolve(vmf_zero, get_max_sigma1_unif(n), args=(n))
    print(res[0]- get_max_sigma1_unif(n))
    return res[0]
'''
print(get_max_sigma_vmf_1D(16), get_max_sigma1_unif(16))

colors=['k', 'cyan', 'coral', 'olivedrab', 'tomato']
#i=0
for n in range(3, 100):
    plt.plot(n, get_max_sigma_vmf_1D(n), 'o', c='cyan', ms=7)
#    #plt.axhline(np.cos(np.pi/n), label='n={}'.format(n), c=colors[i])
    plt.plot(n, get_max_sigma1_unif(n), 'o', c='k', ms=3)

#sss = np.linspace(0.0001, np.pi, 1000)
#plt.plot(sss, sss**2*np.log(np.i0(1./sss**2)))
#plt.legend()
plt.show()
'''

def main():
    N = 1000
    nb_com = [5, 10, 15]
    frac_sigma_max = [0.05, 0.5, 0.7]

    #plot for S^1
    fig, ax = plt.subplots(3,3, figsize=(7,7),
                            subplot_kw=dict(projection='polar'))
    for i in range(3):
        n = nb_com[i]
        sigma_max = get_max_sigma1_unif(n)
        for j in range(3):
            f = frac_sigma_max[j]
            sigma = f*sigma_max
            coordinates = get_coordinates(N, 1, n, sigma)
            sizes = get_equal_communities_sizes(n, N)
            my_communities = get_communities_array(N, sizes)
            plot_circle(ax[i,j], coordinates, my_communities, n, N)
            if j==0:
                ax[i,j].text(np.pi, 2.5, r'$n = {}$'.format(n))
            if i==0:
                ax[i,j].set_title(r'$\sigma = {}$'.format(f))
    plt.axis('off')
    plt.savefig('S1_sigma_max.pdf', dpi=600, format='pdf')
    plt.show()

    #plot for S^1
    fig, ax = plt.subplots(3,3, figsize=(7,7),
                            subplot_kw=dict(projection='3d'))
    for i in range(3):
        n = nb_com[i]
        sigma_max = np.sqrt(2./n)
        for j in range(3):
            f = frac_sigma_max[j]
            sigma = f*sigma_max
            coordinates = get_coordinates(N, 2, n, sigma)
            sizes = get_equal_communities_sizes(n, N)
            my_communities = get_communities_array(N, sizes)
            plot_sphere(ax[i,j], coordinates, my_communities, n, N)
            #if j==0:
            #    ax[i,j].text(r'$n = {}$'.format(n))
            if i==0:
                ax[i,j].set_title(r'$\sigma = {}$'.format(f))
    plt.axis('off')
    plt.savefig('S2_sigma_max.pdf', dpi=600, format='pdf')
    plt.show()
'''
    coordinatesS2, centersS2 = get_coordinates(N, 2, nb_com, sigma2, output_centers=True)
    coordinates = [coordinatesS1, coordinatesS2]
    sigma2 = get_sigma_d(sigma1, 2)
'''
if __name__=='__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plot of number of neighbors with distance on S1 and S2 

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.special import gamma
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=12)

def number_nearest_neighbors_S2_disks_cos(nc):
    angle = 3*np.arccos(1 - 2./nc)
    angle = np.where(angle>np.pi, np.pi, angle)
    return nc*0.5*(1-np.cos(angle)) - 1

def number_nearest_neighbors_S2_disks(nc):
    return 16./nc**2 -24./nc + 8

def phi_n(phi, n):
    return phi - np.sin(phi)*np.cos(phi) - (np.pi / n)

def number_nearest_neighbors_S3_exact(n):
    phi = fsolve(phi_n, np.pi/2, args=(n))
    angle = 3*phi[0]
    if angle > np.pi:
        angle = np.pi
    out = angle - np.sin(angle)*np.cos(angle)
    return (n / np.pi) * out -1

def number_nearest_neighbors_S3(nc):
    y = 3*(3*nc/2)**(1./3)
    out = y - np.sin(y)*np.cos(y)
    return nc*out/np.pi - 1

nc_list = np.array([5,10,15,20,25])
cmap = matplotlib.cm.get_cmap('viridis')
dthetas = np.linspace(0, np.pi, 1000)

def get_approximate_phi_n(n, D):
    phi = D/n * np.sqrt(2*np.pi/(D-1))
    return phi**(1./D)

def assert_volume_of_ball_with_n(phi, D):
    out = gamma(D/2) * np.sqrt(np.pi) / gamma((D+1)/2)
    integral, error = quad(power_sine, 0, phi, args=(D))
    return out/integral

def number_nearest_neighbors_SD(n, D):
    if D==1:
        nnn = np.ones(n.shape)*2
        nnn[0]=0
        nnn[1]=1
    elif D==2:
        nnn = number_nearest_neighbors_S2_disks_cos(n)
    elif D==3:
        nnn = []
        for i in n:
            nnn.append(number_nearest_neighbors_S3_exact(i))
    else:
        nnn = [0, 1]
        for i in n[2:]:
            phi_i = find_characteristic_angle(i, D)
            i_exp = assert_volume_of_ball_with_n(phi_i, D)
            #assert abs(i-i_exp)<1e-5, 'characteristic volume not ok'
            #print(assert_volume_of_ball_with_n(phi_i, D))
            #print('D={}, n={}'.format(D,i), phi_i, get_approximate_phi_n(i, D))
            nnn.append(compute_number_nearest_neighbors_general(i, D, phi_i))
    return np.array(nnn)

def phi_n_general(phi, n, D):
    integral, err = quad(power_sine, 0, phi, args=(D))
    res = gamma(D/2) * np.sqrt(np.pi) / (gamma((D+1)/2) * n)
    return integral - res

def find_characteristic_angle(n, D):
    phi = fsolve(phi_n_general, np.pi/D, args=(n, D))
    return phi[0]

def power_sine(theta, D):
    return np.sin(theta)**(D-1)

def compute_number_nearest_neighbors_general(n, D, phi_n):
    integral, err = quad(power_sine, 0, 3*phi_n, args=(D))
    nnn = integral * n * gamma((D+1)/2) / (gamma(D/2)*np.sqrt(np.pi))
    return nnn - 1


nc = np.arange(100)+1
#nc = np.array([1000, 1e4, 1e5, 1e6, 1e7, 1e8])

xmin, xmax = 1., 37.5
ymin, ymax = 0., 37.
fig, ax = plt.subplots(figsize=(5,4)) #(3.375, 3) À FAIRE
plt.rcParams.update({
    "text.usetex": True,})
plt.plot(nc, nc-1, ':', c='k', linewidth=2)
colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0), 'orange', 'tomato']
angle=[0,2,5,19,27]
for D in [5,4,3,2,1]:
    curve = number_nearest_neighbors_SD(nc, D)
    plt.plot(nc, curve,
            c='white', 
            linewidth = 7)
    plt.plot(nc, curve, 
            c=colors[D-1], label=r'$D={}$'.format(D),
            linewidth = 3.5)

    ax.text(32.2, curve[32]-0.8, r'$D={}$'.format(D), rotation=angle[D-1], backgroundcolor='white')
    #plt.axhline(3**D-1, linestyle=':', color=colors[D-1])

plt.xlabel(r'$n$')
plt.ylabel(r'$n_{\mathrm{nn}}$')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#handles, labels = plt.gca().get_legend_handles_labels()
#order = [4,3,2,1,0]
#plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], framealpha=1.)



plt.tight_layout()
plt.savefig('figure_neighbors_lines', dpi=600)
plt.show()

schema = False
if schema:
    thetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    circ = np.linspace(0, 2*np.pi, 1500)
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(circ, np.ones(circ.shape), c='k', alpha=0.3)
    ax.plot(thetas, np.ones(thetas.shape), 'o', c='white', ms=9)
    ax.plot(thetas, np.ones(thetas.shape), 'o', c=cmap(1./20), ms=7)
    plt.axis('off')
    plt.tight_layout()
    plt.ylim(0.,1.1)
    plt.savefig('circle', transparent=True, dpi=600)
    plt.show()

    coordinates = place_modes_coordinates_on_sphere(30, place='uniformly')

    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    theta, phi = coordinates.T[0], coordinates.T[1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=1, linewidth=0, zorder=1)
    nodes = np.where(theta>0)
    plt.plot(xx[nodes],yy[nodes],zz[nodes],'o', color='white',ms=9,zorder=8,alpha=0.5)
    plt.plot(xx[nodes],yy[nodes],zz[nodes],'o', color=cmap(1.1/3),ms=7,zorder=10)
    l=0.7
    ax.set_xlim(-l,l)
    ax.set_ylim(-l,l)
    ax.set_zlim(-l,l)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('sphere', transparent=True, dpi=600)
    plt.show()

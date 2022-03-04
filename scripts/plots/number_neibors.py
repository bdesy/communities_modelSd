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
import sys
sys.path.insert(0, '../../src/')
from geometric_functions import *

font = {'size'   : 13, 
    'family': 'serif'}

matplotlib.rc('font', **font)

def number_neighbors_S2(nc, dt):
	return (nc/2.)*(1-np.cos(dt))

def number_neighbors_S1(nc, dt):
	return (nc/np.pi)*dt

def number_nearest_neighbors_S2_disks_cos(nc):
	angle = 3*np.arccos(1 - 2./nc)
	return nc*0.5*(1-np.cos(angle)) - 1

def number_nearest_neighbors_S2_disks(nc):
	return 16./nc**2 -24./nc + 8


def number_nearest_neighbors_S3(nc):
	y = 3*(3*nc/2)**(1./3)
	out = y - np.sin(y)*np.cos(y)
	return nc*out/np.pi - 1

nc_list = np.array([5,10,15,20,25])
cmap = matplotlib.cm.get_cmap('viridis')
dthetas = np.linspace(0, np.pi, 1000)

'''
fig = plt.figure(figsize=(5,7))

fig.add_subplot(211)
for i in range(5):
	nc = nc_list[i]
	c = cmap(nc/np.max(nc_list))
	plt.plot(dthetas, number_neighbors_S2(nc, dthetas), c=c, label='{} points'.format(nc))
plt.legend()
plt.xticks([0, np.pi/2, np.pi], ['0', r'$\pi/2$', r'$\pi$'])
plt.xlabel(r'$\Delta\theta$')

fig.add_subplot(212)
for i in range(5):
	nc = nc_list[i]
	c = cmap(nc/np.max(nc_list))
	plt.plot(dthetas, number_neighbors_S1(nc, dthetas), c=c, label='{} points'.format(nc))
plt.legend()
plt.xticks([0, np.pi/2, np.pi], ['0', r'$\pi/2$', r'$\pi$'])
plt.xlabel(r'$\Delta\theta$')
plt.show()


plt.plot(dthetas, number_neighbors_S1(25, dthetas)/25., c='darkcyan', label=r'$S^1$')
plt.plot(dthetas, number_neighbors_S2(25, dthetas)/25., c='coral', label=r'$S^2$')
plt.ylabel(r'$\%$ of all neighbors')
plt.xticks([0, np.pi/2, np.pi], ['0', r'$\pi/2$', r'$\pi$'])
plt.legend()
plt.xlabel(r'$\Delta\theta$')
plt.show()
'''

nc = np.arange(25)+1

nnS1 = np.ones(nc.shape)*2
nnS1[0]=0
nnS1[1]=1

plt.figure(figsize=(5.5,5))
plt.plot(nc, nnS1, '^', c=cmap(0.), ms=5, label=r'$D=1$')
plt.plot(nc, (number_nearest_neighbors_S2_disks(nc)), 's', c=cmap(2.2/5), ms=5, label=r'$D=2$')

#plt.plot(nc, (number_nearest_neighbors_S3(nc)), 'o', c='coral', ms=5, label=r'$n_{nn}$ in $S^3$')
#plt.plot(nc, (number_nearest_neighbors_S2_disks(nc)).astype(int), 'o', c='tomato', ms=2, label=r'$\lfloor n_{nn}\rfloor$')


plt.xlabel('nb points')
plt.ylabel('nb near neighbors')
#plt.xlim(2.5, 25.5)
plt.ylim(1., 14)
plt.grid()
plt.legend(loc=2)
plt.show()


thetas = np.linspace(0, 2*np.pi, 20)
circ = np.linspace(0, 2*np.pi, 1500)
plt.figure(figsize=(3,3))
plt.polar(circ, np.ones(circ.shape), c='k', alpha=0.3)
plt.polar(thetas, np.ones(thetas.shape), 'o', c=cmap(0.))
plt.axis('off')
plt.show()

coordinates = place_modes_coordinates_on_sphere(20, place='uniformly')

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
    x, y, z,  rstride=1, cstride=1, color='k', alpha=0.1, linewidth=0)
ax.scatter(xx,yy,zz,color=cmap(2.2/5),s=20)
l=0.8
ax.set_xlim(-l,l)
ax.set_ylim(-l,l)
ax.set_zlim(-l,l)
plt.axis('off')
plt.show()

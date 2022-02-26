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
from scipy.optimize import fsolve

def number_neighbors_S2(nc, dt):
	return (nc/2.)*(1-np.cos(dt))

def number_neighbors_S1(nc, dt):
	return (nc/np.pi)*dt

def number_nearest_neighbors_S2_disks(nc):
	angle = 3*np.arccos(1 - 2./nc)
	return nc*0.5*(1-np.cos(angle)) - 1

def number_nearest_neighbors_S3(nc):
	y = 3*(3*nc/2)**(1./3)
	out = y - np.sin(y)*np.cos(y)
	return nc*out/np.pi - 1

nc_list = np.array([5,10,15,20,25])
cmap = matplotlib.cm.get_cmap('viridis')
dthetas = np.linspace(0, np.pi, 1000)

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


nc = np.arange(50)+1
plt.figure(figsize=(10,4))
plt.plot(nc, (number_nearest_neighbors_S2_disks(nc)), 'o', c='olivedrab', ms=5, label=r'$n_{nn}$ in $S^2$')
plt.plot(nc, (number_nearest_neighbors_S3(nc)), 'o', c='coral', ms=5, label=r'$n_{nn}$ in $S^3$')
#plt.plot(nc, (number_nearest_neighbors_S2_disks(nc)).astype(int), 'o', c='tomato', ms=2, label=r'$\lfloor n_{nn}\rfloor$')

plt.xlabel(r'$n$')
plt.grid()
plt.legend()
plt.show()


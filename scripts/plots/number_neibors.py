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

font = {'size'   : 12, 
    'family': 'serif'}
matplotlib.rc('font', **font)

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
		nnn = []
		for i in n:
			phi_i = find_characteristic_angle(i, D)
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

def find_characteristic_angle_cust(n, D):
	phi_n = np.pi/D
	res = gamma(D/2) * np.sqrt(np.pi) / (gamma((D+1)/2) * n)
	integral, err = quad(power_sine, 0, phi_n, args=(D))
	while abs(res-integral)>1e-5:
		phi_n += (res-integral)/(10*D)
		integral, err = quad(power_sine, 0, phi_n, args=(D))
	return phi_n

def power_sine(theta, D):
	return np.sin(theta)**(D-1)

def compute_number_nearest_neighbors_general(n, D, phi_n):
	integral, err = quad(power_sine, 0, 3*phi_n, args=(D))
	nnn = integral * n * gamma(D/2) / gamma((D+1)/2)
	return nnn - 1


nc = np.arange(1000)+1

m=4

fig, ax1 = plt.subplots(figsize=(5,4))
plt.rcParams.update({
    "text.usetex": True,})

colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0), 'coral']
for D in [1,2,3,4,5, 6]:
	plt.plot(nc, number_nearest_neighbors_SD(nc, D), 'o', c=colors[D-1], ms=m-.4, label=r'$D={}$'.format(D))

plt.xlabel(r'$n$')
plt.ylabel(r'$n_{\mathrm{nn}}$')
plt.xlim(0., 1000.)
#plt.ylim(0., 100.)
plt.grid()
plt.legend(loc=2)

#ax2 = ax1.twinx()
#ax2.tick_params(axis ='y')
#ax2.set_yticks([2.,8.,26.], [2.,8.,26.])

plt.tight_layout()
#plt.savefig('figure_neighbors_empty', dpi=600)
plt.show()

schema = False
if schema:
	thetas = np.linspace(0, 2*np.pi, 20)
	circ = np.linspace(0, 2*np.pi, 1500)
	plt.figure(figsize=(3,3))
	plt.polar(circ, np.ones(circ.shape), c='k', alpha=0.3)
	plt.polar(thetas, np.ones(thetas.shape), 'o', c=cmap(1./20))
	plt.axis('off')
	plt.savefig('circle', transparent=True, dpi=600)
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
	ax.scatter(xx,yy,zz,color=cmap(1.1/3),s=20)
	l=0.8
	ax.set_xlim(-l,l)
	ax.set_ylim(-l,l)
	ax.set_zlim(-l,l)
	plt.axis('off')
	plt.savefig('sphere', transparent=True, dpi=600)
	plt.show()

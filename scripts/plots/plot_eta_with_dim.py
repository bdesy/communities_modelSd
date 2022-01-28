#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots threshold of connection probability with regards 
to angular distance against dimension

Author: Béatrice Désy

Date : 28/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *

from matplotlib import rc
rc('font',**{'serif':['Computer Modern Roman'], 'size':11})
rc('text', usetex=True)

dimensions = np.arange(1,100)
N = 1000
colors = ['coral', 'darkcyan', 'olivedrab']

def eta(kkp, mu, N, d):
	out = mu*kkp*2*np.pi**((d+1)/2)
	out /= N*gamma((d+1)/2)
	return out**(1./d)

def get_mu(str, D=1, beta=3.5, average_kappa=10.):
	if str=='default':
		mu = compute_default_mu(D, beta, average_kappa)
	elif str=='fixed':
		mu = 0.1 / average_kappa
	return mu
i=0
for kkp in [1, np.sqrt(N), N]:
	c = colors[i]
	for mu in ['default', 'fixed']:
		if mu=='default':
			fmt='-'
		elif mu=='fixed':
			fmt=':'
		etas = []
		for d in dimensions:
			etas.append(eta(kkp, get_mu(mu, D=d, beta=d*2.), N, d))
		plt.plot(dimensions, etas, fmt, c=c, label=r'$\kappa\kappa^\prime={},\, \mu$ '.format(int(kkp))+mu)
	i+=1
plt.ylabel(r'$\eta$')
plt.xlabel(r'$d$')
plt.yticks([0, np.pi/16, np.pi/8, 3*np.pi/16, np.pi/4],['0', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$'])
plt.legend()
plt.grid()
plt.xlim(1, 49)
plt.yl im(0,1)
plt.show()
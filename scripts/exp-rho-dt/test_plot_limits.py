#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots probability density functions of angular distance 
between connected nodes in different dimensions

Author: Béatrice Désy

Date : 28/07/2021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma




def eta(kappa_i, kappa_j, average_kappa, N, beta, d):
	out = kappa_i*kappa_j/(average_kappa*N)
	out *= gamma(d/2)/gamma((d+1)/2)
	out *= beta/d * np.sin(d*np.pi/beta)
	return out**(1./d)

kappa_i, kappa_j, average_kappa = 100., 10., 10.
N = 1000

beta = np.linspace(1., 1000, 10000000)
print(np.max(beta))

r=20

plt.plot(r*beta, eta(kappa_i, kappa_j, average_kappa, N, beta*r, beta))
plt.axhline(1.)
plt.xlabel(r'$\beta$')
plt.ylim(0, 2)
plt.title(r'$\beta/d=2$')
plt.show()


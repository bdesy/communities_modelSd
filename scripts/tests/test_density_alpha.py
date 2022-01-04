#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Validate density of alpha calculations
Author: Béatrice Désy

Date : 21/10/2021
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *

#setup stuff
rng = np.random.default_rng()
average_degree = 10.
y = 3.0
N = 1000
D = 10
mu = 0.05
R = compute_radius(N, D)
print(R)

#sample kappas, kappas'
k_0 = (y-2) * average_degree / (y-1)
a = y - 1.
kappas = k_0 / rng.random(500000)**(1./a)
kappasp = k_0 / rng.random(500000)**(1./a)

#compute alphas
alphas = (mu*kappas*kappasp)**(1./D) / R

#define theoretical density
def theoretical_density_alpha(a, R, D, y, mu, k_0):
	out = D * R**(D*(1-y)) * mu**(y-1) * (y-1)**2
	out *= k_0**(2*(y-1))
	out *= np.log(((a*R)**D)/(mu*k_0**2))
	out *= a**(D*(1-y)-1)
	return out

#plot it
count, bins, _ = plt.hist(alphas, N, density=True)
alpha_fit = theoretical_density_alpha(bins, R, D, y, mu, k_0)

plt.plot(bins, max(count)*alpha_fit/max(alpha_fit))
plt.axvline((mu*k_0*k_0)**(1./D)/R, color='coral')
plt.show()

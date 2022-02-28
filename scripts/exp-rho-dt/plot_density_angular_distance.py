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
import matplotlib 
from scipy.special import gamma
from scipy.integrate import quad, RK45
from tqdm import tqdm

import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *

from math import factorial, gamma
from scipy.special import hyp2f1

from util import *

#matplotlib.rcParams['text.usetex']= True
font = {'size'   : 13, 
    'family': 'serif'}

matplotlib.rc('font', **font)

cmap = matplotlib.cm.get_cmap('viridis')



Dthetas = np.linspace(1e-5, np.pi, 100000)
kappa_i, kappa_j = 10., 10.
ratio = 3.5
N = 10000
average_kappa = 10.
Dmin, Dmax = 1, 5

plt.figure(figsize=(5.5,5))

for D in range(Dmax, Dmin-1, -1):
    beta = ratio * D
    R = compute_radius(N, D)
    mu = compute_default_mu(D, beta, average_kappa)
    print(mu, 'mu')

    rho = np.sin(Dthetas)**(D-1)
    pij = connection_prob(Dthetas, kappa_i, kappa_j, D, beta, R=R, mu=mu)
    denum, error = integrated_connection_prob(kappa_i, kappa_j, D, beta, mu=mu, R=R)
    eta = compute_eta(kappa_i, kappa_j, mu, R, D)
    print('D {}, eta = {}'.format(D, eta))
    #other_denum = integrated_connection_prob_eta(D, beta, eta)
    other_denum = normalization_2f1(D, beta, eta)
    print('int is', denum, error)
    print('denum 2f1 is ', other_denum)
    c = cmap((D)/Dmax)

    plt.plot(Dthetas, pij*rho/denum, label=r'$D = {}$'.format(D), color=c)
    plt.plot(Dthetas, pij*rho/other_denum, ':', color=c)
    print('normalisation verif : ', np.sum((pij*rho/denum)[:-1]*np.diff(Dthetas)))

    plt.axvline(eta**(1./D), color=c, alpha=0.8, linestyle='--')

plt.xticks([0, np.pi/8, np.pi/4],['0', r'$\pi/8$', r'$\pi/4$'])
plt.xlabel(r'$\Delta\theta$')
plt.ylabel(r'$\rho(\Delta\theta\ |\ \kappa, \kappa\')$')
plt.legend(loc=(0.03, 0.6))
#plt.title(r'$\kappa={}$, $\kappa^,={}$, $\beta/d={}$'.format(kappa_i, kappa_j, ratio))
#plt.savefig('pdf', dpi=600)
#plt.ylim(0,10.)
plt.xlim(-0.01, np.pi/4+0.01)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

#probability averaged over kappas

rng = np.random.default_rng()
average_degree = 10.
y = 2.5
N = 1000
k_0 = (y-2) * average_degree / (y-1)

Dthetas = np.linspace(0.001, np.pi/2, 200)
prob_Dthetas = np.zeros(Dthetas.shape)

def other_integrand(a, D, R, y, mu, k_0, beta, Dt):
    out = a**(D*(1-y)-1)
    out *= np.log(((a*R)**D)/(mu*k_0**2))
    out /= (1 + (Dt/a)**beta)
    out /= integrated_connection_prob_alpha(a, beta, D)
    return out

def prefactor(D, R, y, mu, k_0):
    out = D * R**(D*(1-y)) * mu**(y-1) * (y-1)**2
    out *= k_0**(2*(y-1))
    return out

def integrand_alpha(Dtheta, alpha, beta, D):
    A = 1 / (1 + (Dtheta/alpha)**beta)
    B = angular_prob(Dtheta, D)
    return A*B

def integrated_connection_prob_alpha(alpha, beta, D):
    args = (alpha, beta, D)
    I, error = quad(integrand_alpha, 0, np.pi, args=args)
    return I

ratio = 20.

for D in range(10, 0, -1):
    beta = ratio * D
    mu = compute_default_mu(D, beta, average_degree)
    R = compute_radius(N, D)
    a_0 = (mu*k_0**2)**(1./D)/R
    pref = prefactor(D, R, y, mu, k_0)

    for i in tqdm(range(len(Dthetas))):
        Dt = Dthetas[i]
        args = (D, R, y, mu, k_0, beta, Dt)
        integral, error = quad(other_integrand, a_0, np.inf, args=args)
        integral *= pref
        integral *= (np.sin(Dt))**(D-1)
        prob_Dthetas[i] = integral
    c = cmap((D+1)/11)
    plt.plot(Dthetas, prob_Dthetas, color=c, label=r'$D={}$'.format(D))
plt.xlabel(r'$\Delta\theta$')
plt.ylabel(r'$\rho(\Delta\theta)$')
plt.title(r'$\gamma={}$, $<\kappa>={}$, $\beta/D={}$'.format(y, average_degree, ratio))
plt.legend()
plt.show()







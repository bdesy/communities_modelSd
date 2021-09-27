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
from scipy.integrate import quad
from tqdm import tqdm

import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *

from math import factorial, gamma

font = {'size'   : 13}

matplotlib.rc('font', **font)

cmap = matplotlib.cm.get_cmap('viridis')

def connection_prob(Dtheta, kappa_i, kappa_j, D, beta, R=1, mu=1):
    chi = R * Dtheta
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1 + chi**beta)

def angular_prob(Dtheta, D):
    a = gamma((D+1)/2) / gamma(D/2)
    b = np.sin(Dtheta)**(D-1)
    return a*b/np.sqrt(np.pi)

def integrand(Dtheta, kappa_i, kappa_j, D, beta, R=1, mu=1):
    a = connection_prob(Dtheta, kappa_i, kappa_j, D, beta, R, mu)
    b = angular_prob(Dtheta, D)
    return a*b

def integrated_connection_prob(kappa_i, kappa_j, D, beta, R=1, mu=1):
    args = (kappa_i, kappa_j, D, beta, R, mu)
    return quad(integrand, 0, np.pi, args=args)

def bfk_term(n, m, a, b):
    t = a**(-2*n-2) * (-1)**m / (b*m + 2*n +2)
    tt = np.pi**(-b*(m+1) + 2*n + 2) * (-1)**m / ((-b*(m+1) + 2*n + 2) * a**(b*(m+1)))
    ttt = a**(-2*n-2) * (-1)**m / (-b*(m+1) + 2*n + 2)
    out = (t + tt - ttt)*(-1)**n
    return out / factorial(2*n + 1)

def alpha(R, kappa_i, kappa_j, mu, D):
    num = (mu*kappa_i*kappa_j)**(1./D)
    return num / R

def default_mu(D, beta, average_kappa):
    if beta < D:
        print('Default value for mu is not valid if beta < D')
    else: 
        mu = gamma((D+1)/2.) * np.sin((D+1)*np.pi/beta) * beta
        mu /= np.pi**((D+2)/2)
        mu /= (2*average_kappa*(D+1))
    return mu


def mode_taylor_alpha(D, a, b):
    x = (D-1)/b - a/(2*np.tan(a))
    x *= 2*np.tan(a)
    x /= (1 + b/2 - a/(np.sin(a)*np.cos(a)))
    x += a
    return x

def mode_taylor_pisur2(D, a, b):
    x = 1 - (D-1)/b * (2/np.pi) 
    x *= (1 + 2*a/np.pi)**b
    x += np.pi/2
    return x

Dthetas = np.linspace(0.001, np.pi, 10000)
kappa_i, kappa_j = 300., 300.
beta = 12.
N=1000
average_kappa = 10.

for D in range(10, 0, -1):
    R = compute_radius(N, D)
    print(D)
    mu = default_mu(D, beta, average_kappa)
    a = alpha(R, kappa_i, kappa_j, mu, D)
    n,m = 10, 100
    truc = 0
    #for i in range(n):
    #    for j in range(m):
    #        truc += bfk_term(i, j, a, beta)
    #print(D, 'sum is ', truc/2)
    rho = angular_prob(Dthetas, D)
    pij = connection_prob(Dthetas, kappa_i, kappa_j, D, beta, R=R, mu=mu)
    denum, error = integrated_connection_prob(kappa_i, kappa_j, D, beta, mu=mu, R=R)
    print('int is', denum, error)
    c = cmap((D+1)/11)
    plt.plot(Dthetas, pij*rho/denum, label=r'$D = {}$'.format(D), color=c)
    print(D, a, beta)
    print((D-1)/beta, '(D-1)/beta', a*np.tan(a), 'a tan a')
    #plt.plot(Dthetas, (a/Dthetas)**beta, label=D)
    #plt.axvline(a, color=c)
    #plt.plot(np.arctan((D-1)/beta), D/10,'o', ms=12, color=c) 

    if D>1:
        f = (D-1)*Dthetas / (beta*np.tan(Dthetas))
        g = 1. / ((1+(a/Dthetas)**beta))

        idth = np.argwhere(np.diff(np.sign(f - g))).flatten()
        #plt.axvline(Dthetas[idth], color=c) 
        plt.axvline(mode_taylor_alpha(D, a, beta), linestyle=':', color=c)
        plt.axvline(a, color=c)

plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
plt.xlabel(r'$\Delta\theta$')
plt.ylabel(r'$\rho(\Delta\theta)$')
plt.legend()
plt.title(r'$\kappa={}$, $\kappa^,={}$, $\beta={}$, $\mu={}$'.format(kappa_i, kappa_j, beta, mu))
#plt.savefig('pdf', dpi=600)
#plt.ylim(0,10.)
plt.xlim(0, np.pi)

plt.show()

'''
for D in range(6, 0, -1):
    R = compute_radius(1000, D)
    print(D)
    mu = default_mu(D, beta, average_kappa)
    a = alpha(R, kappa_i, kappa_j, mu, D)
    f = (D-1)*Dthetas / (beta*np.tan(Dthetas))
    g = 1. / ((1+(a/Dthetas)**beta))

    plt.plot(Dthetas, f, label='LHS')
    plt.plot(Dthetas, g, label='RHS')
    plt.axhline((D-1)/beta)
    plt.title(D)
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()'''


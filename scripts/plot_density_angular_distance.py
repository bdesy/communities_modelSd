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
    if D>1 :
        out = quad(integrand, 0, np.pi, args=args)
    elif D==1:
        out = quad(integrand, 0, np.pi, args=args)
    return out

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

Dthetas = np.linspace(1e-5, np.pi, 100000)
kappa_i, kappa_j = 10., 10.
ratio = 2.5
N = 1000
average_kappa = 10.

for D in range(10, 0, -1):
    beta = ratio * D
    R = compute_radius(N, D)
    mu = default_mu(D, beta, average_kappa)
    print(mu*kappa_i*kappa_j)
    a = alpha(R, kappa_i, kappa_j, mu, D)
    print(D, 'D', a, 'alpha', mu, 'mu')
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
    print(np.sum((pij*rho/denum)[:-1]*np.diff(Dthetas)))
    #print(D, a, beta)
    #print((D-1)/beta, '(D-1)/beta', a*np.tan(a), 'a tan a')
    plt.axvline(a, color=c, alpha=0.5)
    #if D==1 :
    #    plt.axhline(1./a)
    #if D==2:
    #    plt.axhline(1./np.tan(a/2))
    if D>100:
        f = (D-1)*Dthetas / (beta*np.tan(Dthetas))
        g = 1. / ((1+(a/Dthetas)**beta))

        idth = np.argwhere(np.diff(np.sign(f - g))).flatten()
        #plt.axvline(Dthetas[idth], color=c) 
        plt.axvline(mode_taylor_alpha(D, a, beta), linestyle=':', color=c)
        plt.axvline(a, color=c)

#plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
plt.xlabel(r'$\Delta\theta$')
plt.ylabel(r'$\rho(\Delta\theta\ |\ \kappa, \kappa\')$')
plt.legend()
plt.title(r'$\kappa={}$, $\kappa^,={}$, $\beta/d={}$'.format(kappa_i, kappa_j, ratio))
#plt.savefig('pdf', dpi=600)
#plt.ylim(0,10.)
plt.xlim(-0.01, 1.0)
plt.ylim(0, 40)
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







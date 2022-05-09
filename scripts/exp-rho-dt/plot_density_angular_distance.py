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
from scipy.optimize import fsolve
from tqdm import tqdm

import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *

from math import factorial
from scipy.special import hyp2f1

from util import *


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)

cmap = matplotlib.cm.get_cmap('viridis')
colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0), 'coral']

limit=True

Dthetas = np.linspace(1e-5, np.pi, 100000)
kappa_i, kappa_j = 10., 10.
ratio = 3.5
N = 1000
average_kappa = 10.
Dmin, Dmax = 1, 5
#colors = [cmap(D/Dmax) for D in range(Dmin, Dmax+1)]


def get_approx_max_location(eta, beta, D):
    res = (D-1)/(beta - D + 1)
    return eta*res**(1./beta)

def funk(x, eta, D, beta):
    res = (D-1)*((eta/x)**beta + 1)
    res -= beta*np.tan(x)/x
    #res = np.sin(x) / (1 + (x/eta)**beta) * beta * (x/eta)**beta / x
    #res -= (D-1)*np.cos(x)
    return res

def get_exact_max_location(eta, beta, D):
    res = fsolve(funk, eta, args=(eta, D, beta))
    print(res)
    return res[0]

def limite_beta_eta(Dthetas, eta, D):
    nonzero = D*Dthetas**(D-1) / eta**D
    out = np.where(Dthetas<eta, nonzero, 0)
    return out


fig = plt.figure(figsize=(3.375, 3))
ax = fig.add_subplot(111)

for D in range(Dmax, Dmin-1, -1):
    beta = ratio * D
    R = compute_radius(N, D)
    mu = compute_default_mu(D, beta, average_kappa)
    eta = (mu*kappa_i*kappa_j)**(1./D) / R
    #print(mu, 'mu')

    rho = np.sin(Dthetas)**(D-1)
    pij = 1./(1 + (Dthetas/eta)**beta)
    denum, error = integrated_connection_prob_eta(D, beta, eta)
    

    print('D={}, eta = {}'.format(D, eta))
    #other_denum = integrated_connection_prob_eta(D, beta, eta)
    #other_denum = normalization_2f1(D, beta, eta)
    #print('int is', denum, error)
    #print('denum 2f1 is ', other_denum)
    
    c = colors[D-1]

    plt.plot(Dthetas, pij*rho/denum, color='white', linewidth=4)
    plt.plot(Dthetas, pij*rho/denum, label=r'$D = {}$'.format(D), 
                color=c, linewidth=2.5)

    #approx_max = get_approx_max_location(eta, beta, D)
    #print(approx_max, get_exact_max_location(eta, beta, D))
    #plt.axvline(approx_max, color=c)
    #plt.plot(Dthetas, pij*rho/other_denum, ':', color=c)
    #print('normalisation verif : ', np.sum((pij*rho/denum)[:-1]*np.diff(Dthetas)))


    if limit:
        beta = 1000.*D
        mu = compute_default_mu(D, beta, average_kappa)
        pij = connection_prob(Dthetas, kappa_i, kappa_j, D, beta, R=R, mu=mu)
        denum, error = integrated_connection_prob(kappa_i, kappa_j, D, beta, mu=mu, R=R)
        #plt.plot(Dthetas, pij*rho/denum, '-', color='white', linewidth=3, zorder=0)
        plt.plot(Dthetas, pij*rho/denum, '--', color=c, zorder=0)
        #plt.plot(Dthetas, limite_beta_eta(Dthetas, eta, D), c='k', linewidth=1)
        #plt.ylim(0, 32)
    else:
        pass
        #plt.axvline(eta, color=c, alpha=0.8, linestyle='--')
        #plt.ylim(0, 60)

#plt.xticks([0, np.pi/8, np.pi/4],['0', r'$\pi/8$', r'$\pi/4$'])
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$f_{X|Y,A}(\theta\ |\ \eta,1)$')

handles, labels = plt.gca().get_legend_handles_labels()
order = [4,3,2,1,0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
            loc=(0.09, 0.553), frameon=False)

#plt.title(r'$\kappa={}$, $\kappa^,={}$, $\beta/d={}$'.format(kappa_i, kappa_j, ratio))
plt.ylim(0., 22.)
plt.xlim(0., 0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
#plt.savefig('figure_densities_article', dpi=600)
plt.show()

def puissance_sin(dt, D):
    return np.sin(dt)**(D-1)

fig = plt.figure(figsize=(3.375, 2.5))
ax = fig.add_subplot(111)
for D in [6,5,4,3,2,1]:
    c = colors[D-1]
    if D==6:
        D=50
    norm, err = quad(puissance_sin, 0, np.pi, args=(D))#gamma((D+1)/2.)/(np.sqrt(np.pi)*gamma(D/2.))
    plt.plot(Dthetas, np.sin(Dthetas)**(D-1)/norm, color='white', linewidth=4)
    plt.plot(Dthetas, np.sin(Dthetas)**(D-1)/norm, label=r'$D = {}$'.format(D), 
                color=c, linewidth=2.5)
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$f_{X}(\theta)$')
plt.xticks([0, np.pi/2, np.pi],['0', r'$\pi/2$', r'$\pi$'])
handles, labels = plt.gca().get_legend_handles_labels()
order = [5,4,3,2,1,0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
            loc=(0.02, 0.35), frameon=False)

#plt.title(r'$\kappa={}$, $\kappa^,={}$, $\beta/d={}$'.format(kappa_i, kappa_j, ratio))
plt.ylim(0., 2.9)
plt.xlim(0., np.pi)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('f_X_theta', dpi=600)
plt.show()

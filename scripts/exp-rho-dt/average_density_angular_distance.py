#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots probability density functions of angular distance 
between connected nodes in different dimensions, for all the distribution kappa

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

from math import factorial, gamma
from scipy.special import hyp2f1

from util import *


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)

cmap = matplotlib.cm.get_cmap('viridis')
colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0)]

Dmin, Dmax = 1, 5

#probability averaged over kappas

rng = np.random.default_rng()
average_degree = 10.
y = 2.5
N = 1000
k_0 = (y-2) * average_degree / (y-1)

Dthetas = np.linspace(0.001, np.pi, 1000)
prob_Dthetas = np.zeros(Dthetas.shape)
prob_Dthetas_infinite_beta = np.zeros(Dthetas.shape)

def prefactor(D, R, y, mu, k_0):
    out = D * R**(D*(1-y)) * mu**(y-1) * (y-1)**2
    out *= k_0**(2*(y-1))
    return out

def pdf_eta_pareto(eta, D, R, y, mu, k_0):
    out = eta**(D*(1-y)-1)
    out *= np.log(((eta*R)**D)/(mu*k_0**2))
    out *= prefactor(D, R, y, mu, k_0)
    return out

def other_integrand(eta, D, R, y, mu, k_0, beta, Dt):
    out = pdf_eta_pareto(eta, D, R, y, mu, k_0)
    if beta<np.inf:
        out /= (1 + (Dt/eta)**beta)
    return out

def compute_expectation_Z_eta(eta_0, beta, D, R, y, mu, k_0):
    args = (beta, D, R, y, mu, k_0)
    integral, error = quad(expectation_Z_eta, 
                            eta_0, np.inf, args=args)
    return integral

def Z_eta(eta, beta, D):
    integral, error = quad(integrand_eta, 0., np.pi, args=(eta, beta, D))
    return integral

def expectation_Z_eta(eta, beta, D, R, y, mu, k_0):
    return Z_eta(eta, beta, D)*pdf_eta_pareto(eta, D, R, y, mu, k_0)

def integrand_eta(Dtheta, eta, beta, D):
    A = 1 / (1 + (Dtheta/eta)**beta)
    B = np.sin(Dtheta)**(D-1)
    return A*B

def integrand_eta_infinite_beta(Dtheta, D):
    return np.sin(Dtheta)**(D-1)

def integrated_connection_prob_eta(eta, beta, D):
    if beta<np.inf:
        args = (eta, beta, D)
        I, error = quad(integrand_eta, 0, np.pi, args=args)
    else:
        I, error = quad(integrand_eta_infinite_beta, 0, eta, args=(D))
    return I

fig = plt.figure(figsize=(3.375, 3))
ax = fig.add_subplot(111)
ratio=3.5

compute=False
verif=False
limit=False

for D in range(Dmax, Dmin-1, -1):
    if compute:
        beta = ratio * D
        mu = compute_default_mu(D, beta, average_degree)
        R = compute_radius(N, D)
        eta_0 = (mu*k_0**2)**(1./D)/R
        norm = compute_expectation_Z_eta(eta_0, beta, D, R, y, mu, k_0)
        print(r'$D={}$'.format(D))
        prob_Dthetas_infinite_beta = np.sin(Dthetas)**(D-1)/norm
        for i in tqdm(range(len(Dthetas))):
            Dt = Dthetas[i]
            args = (D, R, y, mu, k_0, beta, Dt)
            integral, error = quad(other_integrand, eta_0, np.inf, args=args)
            integral *= (np.sin(Dt))**(D-1)
            #integral /= norm
            prob_Dthetas[i] = integral
            if limit:
                if Dt > eta_0:
                    args = (D, R, y, mu, k_0)
                    integral_infinite_beta, error = quad(pdf_eta_pareto, Dt, np.inf, args=args)
                    prob_Dthetas_infinite_beta[i] *= integral_infinite_beta

        prob_Dthetas /= norm
        print(norm, 'analytical normalisation')
        data = np.column_stack((Dthetas, prob_Dthetas, prob_Dthetas_infinite_beta))
        np.savetxt('data/D{}_beta{}.txt'.format(D, ratio), data, header='theta    prob_theta    prob_theta_inf_beta')
        #plt.axvline(eta_0, color=colors[D-1])
    else:
        data = np.loadtxt('data/D{}_beta{}.txt'.format(D, ratio)).T
        Dthetas = data[0]
        prob_Dthetas = data[1]
        prob_Dthetas_infinite_beta = data[2]
    
    plt.plot(Dthetas, prob_Dthetas, color='white', linewidth=5)
    plt.plot(Dthetas, prob_Dthetas, color=colors[D-1], 
            label=r'$D={}$'.format(D), linewidth=3)
    print(np.sum(np.diff(Dthetas)*prob_Dthetas[:-1]), 'should be 1.')

    if verif:
        dist = np.loadtxt('data/all-kappa-verif/D{}-gamma{}-beta{}.txt'.format(D, 2.5, ratio))
        if len(dist)>1e6:
            dist = dist[:1000000]
        plt.hist(dist, bins=600, density=True, alpha=0.5, color=colors[D-1])
    
    if limit:
        plt.plot(Dthetas, prob_Dthetas_infinite_beta, ':', color='k', linewidth=1.5)

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$f_{X|A}(\theta\,|\,1)$')
#plt.title(r'$\gamma={}$, $<\kappa>={}$, $\beta/D={}$'.format(y, average_degree, ratio))

handles, labels = plt.gca().get_legend_handles_labels()
order = [4,3,2,1,0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
            loc=(0.09, 0.583), frameon=False) 

plt.ylim(0., 22.)
plt.xlim(0., 0.9)
log=True
if log:
    plt.ylim(0.1, 50.)
    plt.yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('densities_all_kappas', dpi=600)
plt.show()


'''
#verif pdf eta
for D in [5,4,3,2,1]:
    mu = compute_default_mu(D, ratio*D, average_degree)
    R = compute_radius(N, D)
    eta_0 = (mu*k_0**2)**(1./D)/R
    etas = []
    for truc in range(100):
        kappas = get_target_degree_sequence(average_degree, N, rng, 'pwl', sorted=False, y=2.5)
        for i in range(N):
            for j in range(i):
                etas.append((mu*kappas[i]*kappas[j])**(1./D)/R)
    eta_axis = np.linspace(eta_0, 1., 1000)
    theo = pdf_eta_pareto(eta_axis, D, R, y, mu, k_0)
    plt.hist(etas, bins=100, density=True, alpha=0.5, color=colors[D-1], range=(eta_0, 2.))
    plt.plot(eta_axis, theo, color=colors[D-1], label=D)
plt.legend()
plt.show()
'''




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
matplotlib.rc('font', size=12)

cmap = matplotlib.cm.get_cmap('viridis')
colors =[cmap(1./20), cmap(1.1/3), cmap(2./3), cmap(9./10), cmap(1.0)]

Dmin, Dmax = 1, 5

#probability averaged over kappas

rng = np.random.default_rng()
average_degree = 10.
y = 2.5
N = 1000
k_0 = (y-2) * average_degree / (y-1)

Dthetas = np.linspace(0.001, np.pi/4, 100)
prob_Dthetas = np.zeros(Dthetas.shape)
limit=True
prob_Dthetas_infinite_beta = np.zeros(Dthetas.shape)

def pdf_eta_pareto(eta, D, R, y, mu, k_0):
    out = eta**(D*(1-y)-1)
    out *= np.log(((eta*R)**D)/(mu*k_0**2))
    out *= prefactor(D, R, y, mu, k_0)
    return out

def other_integrand(eta, D, R, y, mu, k_0, beta, Dt):
    out = pdf_eta_pareto(eta, D, R, y, mu, k_0)
    out /= integrated_connection_prob_eta(eta, beta, D)
    if beta<np.inf:
        out /= (1 + (Dt/eta)**beta)
    return out

def prefactor(D, R, y, mu, k_0):
    out = D * R**(D*(1-y)) * mu**(y-1) * (y-1)**2
    out *= k_0**(2*(y-1))
    return out

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

ratio = 3.5
plt.figure(figsize=(5,4))

ratio=3.5

for D in []:
    print(D)
    mu = compute_default_mu(D, ratio*D, average_degree)
    R = compute_radius(N, D)
    eta_0 = (mu*k_0**2)**(1./D)/R
    etas = []
    for truc in range(0):
        kappas = get_target_degree_sequence(average_degree, N, rng, 'pwl', sorted=False, y=2.5)
        for i in range(N):
            for j in range(i):
                etas.append((mu*kappas[i]*kappas[j])**(1./D)/R)
    eta_axis = np.linspace(eta_0, 1., 1000)
    theo = pdf_eta_pareto(eta_axis, D, R, y, mu, k_0)
    #plt.hist(etas, bins=1000, density=True, alpha=0.5, color=colors[D-1], range=(eta_0, 2.))
    plt.plot(eta_axis, theo, color=colors[D-1], label=D)
plt.legend()
plt.show()

compute=False
verif=True

for D in [1,3,5]:#range(Dmax, Dmin-1, -1):
    if compute:
        beta = ratio * D
        mu = compute_default_mu(D, beta, average_degree)
        R = compute_radius(N, D)
        eta_0 = (mu*k_0**2)**(1./D)/R
        print(r'$D={}$'.format(D))
        for i in tqdm(range(len(Dthetas))):
            Dt = Dthetas[i]
            args = (D, R, y, mu, k_0, beta, Dt)
            integral, error = quad(other_integrand, eta_0, np.inf, args=args)
            integral *= (np.sin(Dt))**(D-1)
            prob_Dthetas[i] = integral
            if limit:
                args = (D, R, y, mu, k_0, np.inf, Dt)
                lower_bound = np.max(np.array([eta_0, Dt]))
                integral_infinite_beta, error = quad(other_integrand, lower_bound, np.inf, args=args)
                prob_Dthetas_infinite_beta[i] = integral_infinite_beta
        data = np.column_stack((Dthetas, prob_Dthetas, prob_Dthetas_infinite_beta))
        np.savetxt('data/D{}_beta{}.txt'.format(D, ratio), data, header='theta    prob_theta    prob_theta_inf_beta')
    else:
        data = np.loadtxt('data/D{}_beta{}.txt'.format(D, ratio)).T
        Dthetas = data[0]
        prob_Dthetas = data[1]
        prob_Dthetas_infinite_beta = data[2]

    plt.plot(Dthetas, prob_Dthetas, color='white', linewidth=6)
    plt.plot(Dthetas, prob_Dthetas, color=colors[D-1], 
            label=r'$D={}$'.format(D), linewidth=3.5)

    if verif:
        dist = np.loadtxt('data/all-kappa-verif/D{}-gamma{}-beta{}.txt'.format(D, 2.5, ratio))
        plt.hist(dist, bins=400, density=True, alpha=0.5, color=colors[D-1])
    #plt.plot(Dthetas, prob_Dthetas_infinite_beta, ':', color=colors[D-1], linewidth=1.5)
    
plt.xticks([0, np.pi/8, np.pi/4],['0', r'$\pi/8$', r'$\pi/4$'])

plt.xlabel(r'$\Delta\theta$')
plt.ylabel(r'$\rho(\Delta\theta)$')
plt.title(r'$\gamma={}$, $<\kappa>={}$, $\beta/D={}$'.format(y, average_degree, ratio))

#handles, labels = plt.gca().get_legend_handles_labels()
#order = [4,3,2,1,0]
#plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=(0.02, 0.553))
plt.ylim(0., 100.)
plt.xlim(0., np.pi/4)
plt.show()






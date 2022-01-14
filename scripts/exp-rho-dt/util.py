#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Functions for the pdf of angular length of edges

Author: Béatrice Désy

Date : 13/01/2022
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1

def compute_eta(kappa_i, kappa_j, mu, R, D):
    return mu*kappa_i*kappa_j/(R**D)

def connection_prob(Dtheta, kappa_i, kappa_j, D, beta, R, mu):
    chi = R * Dtheta
    chi /= (mu * kappa_i * kappa_j)**(1./D)
    return 1./(1 + chi**beta)

def integrand(Dtheta, kappa_i, kappa_j, D, beta, R, mu):
    a = connection_prob(Dtheta, kappa_i, kappa_j, D, beta, R, mu)
    b = np.sin(Dtheta)**(D-1)
    return a*b

def integrand_eta(theta, D, beta, eta):
    a = 1 / (1 + (theta**D / eta)**(beta/D))
    b = np.sin(theta)**(D-1)
    return a*b

def integrated_connection_prob_eta(D, beta, eta):
    args = (D, beta, eta)
    out = quad(integrand_eta, 0, np.pi, args=args)
    return out

def integrated_connection_prob(kappa_i, kappa_j, D, beta, R=1, mu=1):
    args = (kappa_i, kappa_j, D, beta, R, mu)
    if D>1 :
        out = quad(integrand, 0, np.pi, args=args)
    elif D==1:
        out = quad(integrand, 0, np.pi, args=args)
    return out

def alpha(R, kappa_i, kappa_j, mu, D):
    num = (mu*kappa_i*kappa_j)**(1./D)
    return num / R

def normalization_2f1(D, beta, eta):
    tau = D/beta
    um = (np.pi**D)/eta
    return eta * um * hyp2f1(1., tau, 1.+tau, -um**(1./tau)) / D

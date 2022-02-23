#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Functions for reduced mutual information calculations, 
from https://doi.org/10.1103/PhysRevE.101.042304 
using their notation for variables.

Author: Béatrice Désy

Date : 23/02/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from numba import njit

log = np.log

def reduced_mutual_information(N, labels_a, labels_b):
    R, S = get_labels_indices(labels_a), get_labels_indices(labels_b)
    a, b = get_labels_totals(labels_a, R), get_labels_totals(labels_b, R)
    MI = compute_mutual_information(N, labels_a, labels_b, R, S)
    logOmega = approximate_log_Omega(N, a, b, R, S)
    return MI - logOmega / N

def get_labels_indices(labels):
    assert issubclass(labels.dtype.type, np.integer), 'labels not integers'
    indices = set(labels)
    R = len(indices)
    assert indices == set(np.arange(R)), 'labels not smallest integers'
    return R

@njit
def get_labels_totals(labels, R):
    labels_totals = np.zeros(R, dtype=np.float64)
    for r in range(R):
        labels_totals[r] = np.sum(np.where(labels==r, 1, 0))
    return labels_totals

@njit
def get_colabels_totals(labels_a, labels_b, R, S):
    c_matrix = np.zeros((R, S))
    for r in range(R):
        for s in range(S):
            in_r = np.where(labels_a==r, 1, 0)
            in_s = np.where(labels_b==s, 1, 0)
            c_matrix[r, s] = np.sum(in_r*in_s)
    return c_matrix

@njit
def compute_mutual_information(N, labels_a, labels_b, R, S):
    c_matrix = get_colabels_totals(labels_a, labels_b, R, S)
    a = get_labels_totals(labels_a, R)
    b = get_labels_totals(labels_b, S)
    MI = 0.
    for r in range(R):
        for s in range(S):
            c_rs = c_matrix[r,s]
            if c_rs > 0.:
                MI += c_rs * log(N * c_rs / (a[r]*b[s]))
    return MI / N


def approximate_log_Omega(N, a, b, R, S):
    out = (R-1)*(S-1)*log(N + 0.5*R*S)
    w = N / (N + 0.5*R*S)
    x = change_of_variable(a, R, w, N)
    y = change_of_variable(b, S, w, N)
    mu = kindof_normalisation(R, y)
    nu = kindof_normalisation(S, x)

    out += term_with_sums(R, nu, y)
    out += term_with_sums(S, mu, x)
    out += term_with_gammas(R, S, mu, nu)
    return out

@njit
def change_of_variable(a, R, w, N):
    x = w*a / N
    return x + (1-w)/R

@njit
def kindof_normalisation(R, y):
    out = (R+1) / (R * np.sum(y**2))
    return out - 1./R

@njit
def term_with_sums(R, nu, y):
    out = np.sum(log(y))
    return 0.5*(R + nu - 2)*out

def term_with_gammas(R, S, mu, nu):
    t = gamma(mu*R)/(gamma(nu)*gamma(R))**S
    tt = gamma(nu*S)/(gamma(mu)*gamma(S))**R
    return 0.5*log(t*tt)


# -------------------------------Scrap

from time import time

def test_reduced_mutual_information():
    labels_a = []
    for i in range(12):
        labels_a+=[i]*100
    labels_a = np.array(labels_a)
    N = len(labels_a)
    for i in [0, 1, 11, 255]:
        labels_b = np.roll(labels_a, i)
        ti = time()
        print(reduced_mutual_information(N, labels_a, labels_b), time()-ti)


def test_mutual_information():
    labels_a = []
    for i in range(10):
        labels_a+=[i]*10
    labels_a = np.array(labels_a)
    R = get_labels_indices(labels_a)
    N = len(labels_a)
    for i in [0, 1, 11, 25]:
        labels_b = np.roll(labels_a, i)
        S = get_labels_indices(labels_b)
        ti = time()
        print(compute_mutual_information(N, labels_a, labels_b, R, S), time()-ti)

def test_approximate_logOmega():
    labels_a = []
    for i in range(8):
        labels_a+=[i]*100
    labels_a = np.array(labels_a)
    R = get_labels_indices(labels_a)
    N = len(labels_a)
    for i in [0, 1, 11, 25]:
        labels_b = np.roll(labels_a, i)
        S = get_labels_indices(labels_b)
        a, b = get_labels_totals(labels_a, R), get_labels_totals(labels_b, S)
        ti = time()
        print(approximate_log_Omega(N, a, b, R, S), time()-ti)

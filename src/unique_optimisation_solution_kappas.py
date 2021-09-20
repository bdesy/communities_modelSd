#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Test kappa optimization to check whether the solution is global or not

Author: Béatrice Désy

Date : 17/05/2021
"""

import argparse
import palettable
from truc import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Gather hidden parameter filename

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-p', type=str, default='hidvar',
                        help='path to an hidden variable file')
    args = parser.parse_args()

    # Set random number generator

    rng = np.random.default_rng()

    # Retrieve coordinate and parameters

    mod = ModelSD()
    mod.load_parameters(args.filename+'_params.txt')
    mod.load_hidden_variables(args.filename+'.dat')
    N = mod.N
    mu = mod.mu
    beta = mod.beta

    tol = 10e-3
    max_iterations = 2*N

    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    ax.set_prop_cycle('color', palettable.wesanderson.Moonrise5_6.mpl_colors)

    for i in range(10):
        print('Optimization {}'.format(i+1))
        kappas = np.copy(mod.target_degrees)
        kappas_opt = optimize_kappas(N, tol, max_iterations,  
                                mod.coordinates, kappas, 
                                mod.R, beta, mu, 
                                mod.target_degrees, rng, 
                                D=mod.D, verbose=True, perturbation=0.1)

        plt.plot(kappas_opt, '^', ms=11-i)

    plt.show()






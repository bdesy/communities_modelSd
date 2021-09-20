#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Script to generate given average adjacency matrices

Author: Béatrice Désy

Date : 31/04/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *


def sample_gaussian_ring_on_sphere(phi, sigma, size):
    thetas, phis = [],[]
    thetas = np.random.random(size=size)*2*np.pi
    phis = np.random.normal(loc=phi, scale=sigma, size=size)
    return np.array(thetas), np.array(phis)

N=1000
rng = np.random.default_rng()
target_degrees = get_target_degree_sequence(10., N, rng, 'poisson', sorted=False, y=2.5) 
tol = 1e-1
max_iterations = 1000
R, beta, mu, D = 8.920620580763854, 3.0, 0.04154995094740801, 2
vertices = np.array(['v{:05d}'.format(i) for i in range(N)])
header = 'vertex       kappa       theta       phi       target degree'

#exp=''
exp='deux_hemispheres'
print(exp)

if exp=='test_sigma':
    centers = [[0, np.pi/2], [np.pi/2, 0.15], [3*np.pi/2, 0.15], [np.pi, np.pi/2]]
    for sig in [0.05, 0.1, 0.2, 0.3]:
        print('sigma is {}'.format(sig))
        sigmas = [sig, sig, sig, sig]
        filename = 'data/defi_sigma{}'.format(sig)
        thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes=[150, 350, 350, 150])
        coordinates = np.column_stack((thetas, phis))
        kappas = optimize_kappas(N, tol, max_iterations, coordinates, 
                            target_degrees+1e-3, R, beta, mu, 
                            target_degrees, rng, D=2, verbose=True, perturbation=0.1)


        data = np.column_stack((vertices, kappas, coordinates, target_degrees))
        np.savetxt(filename+'.dat', data, delimiter='       ', fmt='%s',header=header)
        params = {'mu':mu, 'beta':beta, 'dimension':D, 'radius':R}
        params_file = open(filename+'_params.txt', 'a')
        params_file.write(str(params))
        params_file.close()

elif exp=='blocs_coins':
    centers = [[0, np.pi/2], [np.pi/2, 0.15], [3*np.pi/2, 0.15], [np.pi, np.pi/2], [15*np.pi/8, np.pi/2]]
    sig = 0.15
    sigmas = [[sig, sig],[sig, sig],[sig, sig],[sig, sig],[sig, sig]]
    filename = 'data/defi_blocs_coins'
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes=[150, 250, 250, 150, 200])
    coordinates = np.column_stack((thetas, phis))
    kappas = optimize_kappas(N, tol, max_iterations, coordinates, 
                        target_degrees+1e-3, R, beta, mu, 
                        target_degrees, rng, D=2, verbose=True, perturbation=0.1)


    data = np.column_stack((vertices, kappas, coordinates, target_degrees))
    np.savetxt(filename+'.dat', data, delimiter='       ', fmt='%s',header=header)
    params = {'mu':mu, 'beta':beta, 'dimension':D, 'radius':R}
    params_file = open(filename+'_params.txt', 'a')
    params_file.write(str(params))
    params_file.close()

elif exp=='blob_anneau':
    centers = [[0, 0]]
    sig = 0.17
    sigmas = [sig]
    filename = 'data/defi_blob_anneau2'
    blob_thetas, blob_phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes=[200])
    ring_thetas, ring_phis = sample_gaussian_ring_on_sphere(np.pi/4, sig, 800)
    thetas, phis = np.concatenate((blob_thetas, ring_thetas)), np.concatenate((blob_phis, ring_phis))
    coordinates = np.column_stack((thetas, phis))

    kappas = optimize_kappas(N, tol, max_iterations, coordinates, 
                        target_degrees+1e-3, R, beta, mu, 
                        target_degrees, rng, D=2, verbose=True, perturbation=0.1)


    data = np.column_stack((vertices, kappas, coordinates, target_degrees))
    np.savetxt(filename+'.dat', data, delimiter='       ', fmt='%s',header=header)
    params = {'mu':mu, 'beta':beta, 'dimension':D, 'radius':R}
    params_file = open(filename+'_params.txt', 'a')
    params_file.write(str(params))
    params_file.close()

elif exp=='deux_hemispheres':
    centers = [[np.pi/2, np.pi/2], [3*np.pi/2, np.pi/2]]
    sigmas = [0.3, 0.3]
    filename = 'data/deux_hemispheres'
    thetas, phis = sample_gaussian_clusters_on_sphere(centers, sigmas, sizes=[500, 500])
    coordinates = np.column_stack((thetas, phis))
    kappas = optimize_kappas(N, tol, max_iterations, coordinates, 
                        target_degrees+1e-3, R, beta, mu, 
                        target_degrees, rng, D=2, verbose=True, perturbation=0.1)


    data = np.column_stack((vertices, kappas, coordinates, target_degrees))
    np.savetxt(filename+'.dat', data, delimiter='       ', fmt='%s',header=header)
    params = {'mu':mu, 'beta':beta, 'dimension':D, 'radius':R}
    params_file = open(filename+'_params.txt', 'a')
    params_file.write(str(params))
    params_file.close()


with open(filename+'_centers_sigmas.txt', 'w') as f:
    f.write(repr(centers))
    f.write(repr(sigmas))
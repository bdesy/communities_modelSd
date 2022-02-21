#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Community structure overlap experiment run in the S1 and S1 model

Author: Béatrice Désy

Date : 01/02/2022
"""


import numpy as np
import json
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
from overlap_util import *

def get_dict_key(D, dd, nc, beta, sigma):
    return 'S{}-'.format(D)+dd+'-{}coms-{}beta-{:.2f}sigma'.format(nc, beta, sigma)

def get_local_params(N, D, nc, sigma, target_degrees): 
    coordinates = get_coordinates(N, D, nc, sigma)
    local_params = {'coordinates':coordinates, 
                                'kappas': target_degrees+1e-3, 
                                'target_degrees':target_degrees, 
                                'nodes':np.arange(N)}
    return local_params

def get_coordinates(N, D, nc, sigma):
    if D==1:
        coordinates = get_communities_coordinates(nc, N, sigma, place='equator')
        coordinates = (coordinates.T[0]).reshape((N, 1))
    elif D==2:
        coordinates = get_communities_coordinates(nc, N, sigma, place='uniformly')
    return coordinates

def sample_model(global_params, local_params, opt_params, average_k, rng, sizes):
    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()
    SD.optimize_kappas(rng)
    SD.reassign_parameters()
    order = get_order_theta_within_communities(SD, sizes)
    SD.build_probability_matrix(order=order) 
    SD.communities = get_communities_array(SD.N, sizes)
    block_mat = get_community_block_matrix(SD, sizes)
    block_mat = normalize_block_matrix(block_mat, len(sizes))
    return block_mat

def measure_stuff(block_mat, r_dist, Y_dist):
    Y_dist.append(np.mean(get_disparities(block_mat)))
    r_dist.append(get_stable_rank(block_mat))

def main():
    N = 1000
    mu = 0.01
    average_k = 10.
    rng = np.random.default_rng()
    opt_params = {'tol':0.2, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}

    sample_size = 10
    nc_list = [5, 15, 25]
    dd_list = ['exp', 'pwl']
    beta_ratio_list = [1.5, 10.]
    sigma_axis = np.linspace(0.01, 0.3, 10)

    tot = 2*sample_size*len(nc_list)*len(dd_list)*len(beta_ratio_list)*len(sigma_axis)

    res = {}
    with tqdm(total=tot) as pbar:
        for nc in nc_list:
            sizes = get_equal_communities_sizes(nc, N)
            for br in beta_ratio_list:
                for dd in dd_list:
                    target_degrees = get_target_degree_sequence(average_k, N, 
                                            rng, dd, sorted=False) 
                    for sigma in sigma_axis:
                        for D in [1,2]:
                            beta = br*D
                            global_params = get_global_params_dict(N, D, beta, mu)
                            
                            key = get_dict_key(D, dd, nc, beta, sigma)
                            
                            if D==2 :
                                sigma = get_sigma_d(sigma, 2)
                            
                            r_dist = []
                            Y_dist = []
                            for i in range(sample_size):
                                local_params = get_local_params(N, D, nc, sigma, target_degrees)
                                B = sample_model(global_params, local_params, opt_params, 
                                                 average_k, rng, sizes)
                                measure_stuff(B, r_dist, Y_dist)
                                pbar.update(1)
                            res[key+'-r'] = r_dist
                            res[key+'-Y'] = Y_dist
    
    filepath = 'data/sample10_betalim'
    with open(filepath+'.json', 'w') as write_file:
        json.dump(res, write_file, indent=4)

if __name__=='__main__':
    main()

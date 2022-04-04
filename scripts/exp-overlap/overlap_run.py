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

def get_dict_key(D, dd, nc, beta, f):
    return 'S{}-'.format(D)+dd+'-{}coms-{}beta-{:.2f}sigmam'.format(nc, beta, f)

def get_local_params(N, D, nc, sigma, target_degrees): 
    coordinates = get_coordinates(N, D, nc, sigma)
    local_params = {'coordinates':coordinates, 
                                'kappas': target_degrees+1e-3, 
                                'target_degrees':target_degrees, 
                                'nodes':np.arange(N)}
    return local_params

def do_measurements(SD, n):
    block_mat = get_community_block_matrix(SD, n)
    m = np.sum(block_mat)/2
    block_mat = normalize_block_matrix(block_mat, n)
    Y_u = list(get_disparities(block_mat))
    r = get_stable_rank(block_mat)
    return m, Y_u, r

def sample_model(global_params, local_params, opt_params, average_k, rng, n):
    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()
    SD.optimize_kappas(rng)
    SD.reassign_parameters()
    return SD
 
def measure_stuff(SD, n, reassign, data):
    if reassign==False:
        sizes = get_equal_communities_sizes(n, SD.N)
        SD.communities = get_communities_array(SD.N, sizes)
    elif reassign:
        labels = np.arange(n)
        misc, centers = get_communities_coordinates(n, SD.N, 0.01, place='uniformly', 
                                                    output_centers=True)
        SD.communities = get_communities_array_closest(SD.N, SD.D, SD.coordinates, centers, labels)
    order = get_order_theta_within_communities(SD, n)
    SD.build_probability_matrix(order=order) 
    m, Y_u, r = do_measurements(SD, n)
    data[0].append(m)
    data[1].append(Y_u)
    data[2].append(r)

def main():
    N = 1000
    mu = 0.01
    average_k = 10.
    rng = np.random.default_rng()
    opt_params = {'tol':0.2, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}
    
    exp=2

    if exp==1:
        sample_size = 10
        nc_list = [5, 15, 25]
        dd_list = ['exp', 'pwl']
        beta_ratio_list = [1.5, 3.5, 10.]
        frac_sigma_axis = np.linspace(0.1, 0.9, 10)

    elif exp==2:
        sample_size = 50
        nc_list = [5, 15, 25]
        dd_list = ['exp']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.linspace(0.05, 0.95, 20)

    elif exp=='test':
        sample_size = 1
        nc_list=[15]
        dd_list = ['exp']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.linspace(0.05, 0.95, 2)

    tot = 2*sample_size*len(nc_list)*len(dd_list)*len(beta_ratio_list)*len(frac_sigma_axis)

    res = {}
    with tqdm(total=tot) as pbar:
        for n in nc_list:
            for br in beta_ratio_list:
                for dd in dd_list:
                    target_degrees = get_target_degree_sequence(average_k, N, 
                                            rng, dd, sorted=False) 
                    for f in frac_sigma_axis:
                        for D in [1,2]:
                            sigma_max = get_sigma_max(n, D)
                            sigma = f*sigma_max
                            beta = br*D
                            global_params = get_global_params_dict(N, D, beta, mu)
                            
                            key = get_dict_key(D, dd, n, beta, f)

                            data_closest = ([],[],[])
                            data = ([],[],[])

                            for i in range(sample_size):
                                local_params = get_local_params(N, D, n, sigma, target_degrees)
                                SD = sample_model(global_params, local_params, opt_params, 
                                                 average_k, rng, n)
                                for reassign in [True, False]:
                                    
                                    if reassign==False:
                                        measure_stuff(SD, n, reassign, data_closest)
                                    else:
                                        measure_stuff(SD, n, reassign, data)

                                pbar.update(1)
                            
                            res[key+'-m-closest'] = data_closest[0]
                            res[key+'-Y-closest'] = data_closest[1]
                            res[key+'-r-closest'] = data_closest[2]
                            res[key+'-m'] = data[0]
                            res[key+'-Y'] = data[1]
                            res[key+'-r'] = data[2]
    
    filepath = 'data/test'
    with open(filepath+'.json', 'w') as write_file:
        json.dump(res, write_file, indent=4)

if __name__=='__main__':
    main()

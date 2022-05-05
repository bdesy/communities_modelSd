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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, default='test',
                        help='dictate which experiment to run')
parser.add_argument('-f', '--filename', type=str, default='test',
                        help='output file name')
args = parser.parse_args()
exp = args.experiment


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
    SD.block_mat = block_mat
    binary_block_mat = np.where(block_mat>1, 1, 0)
    SD.binary_block_mat = binary_block_mat
    degrees = list(np.sum(binary_block_mat, axis=0).astype(float))
    m = np.sum(block_mat)/2
    block_mat = normalize_block_matrix(block_mat, n)
    r = get_stable_rank(block_mat)
    S = get_entropy(block_mat)
    return m, S, r, degrees

def sample_model(global_params, local_params, opt_params, average_k, rng, n):
    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()
    SD.optimize_kappas(rng)
    SD.reassign_parameters()
    return SD

def represent(SD, n, frac_sigma, entropy):
    fig = plt.figure(figsize=(5,5))
    if SD.D==1:
        ax = fig.add_subplot(221, projection='polar')
        plot_coordinates_S1(SD, ax, n)
    elif SD.D==2:
        ax = fig.add_subplot(221, projection='3d')
        plot_coordinates_S2(SD, ax, n)
    ax = fig.add_subplot(222)
    im = ax.imshow(np.log10(SD.probs+1e-7), cmap='Greys')
    ax.set_title('D={}, frac sigma {}'.format(SD.D, frac_sigma))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    ax = fig.add_subplot(223)
    im = ax.imshow(SD.block_mat, cmap='Greys')
    ax.set_title('expected number\n of inter-comm edges\n S={:.2f}'.format(entropy))
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(224)
    im = ax.imshow(SD.binary_block_mat, cmap='Greys')
    mean=np.mean(np.sum(SD.binary_block_mat, axis=0))
    ax.set_title('binarized with \nat least 1 edge\n<k>={:.2f}'.format(mean))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('figures/entropie-degree/S{}-fs0{}.png'.format(SD.D, frac_sigma))
    plt.show()

def define_communities(SD, n, reassign):
    if reassign==False:
        sizes = get_equal_communities_sizes(n, SD.N)
        SD.communities = get_communities_array(SD.N, sizes)
    elif reassign:
        labels = np.arange(n)
        if SD.D==2:
            misc, centers = get_communities_coordinates(n, SD.N, 0.01, place='uniformly', 
                                                    output_centers=True)
        elif SD.D==1:
            misc, centers = get_communities_coordinates(n, SD.N, 0.01, place='equator', 
                                                    output_centers=True)
            centers = (centers.T[0]).reshape((n, 1))
        SD.communities = get_communities_array_closest(SD.N, SD.D, SD.coordinates, centers, labels)


def measure_stuff(SD, n, data):
    order = get_order_theta_within_communities(SD, n)
    SD.build_probability_matrix(order=order) 
    m, S, r, degrees = do_measurements(SD, n)
    data[0].append(m)
    data[1].append(float(S))
    data[2].append(r)
    data[3].append(degrees)

def main():
    N = 1000
    mu = 0.01
    average_k = 10.
    rng = np.random.default_rng()
    opt_params = {'tol':0.2, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':False}

    if exp=='1':
        sample_size = 25
        nc_list = [5, 15, 25]
        dd_list = ['exp']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.linspace(0.05, 0.95, 20)

    elif exp=='test':
        sample_size = 2
        nc_list=[17]
        dd_list = ['exp']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.array([0.05])

    if exp=='2':
        sample_size = 100
        nc_list = [5, 15, 25]
        dd_list = ['exp']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.linspace(0.05, 0.95, 30)

    if exp=='pwl':
        sample_size = 25
        nc_list = [5, 15, 25]
        dd_list = ['pwl']
        beta_ratio_list = [3.5]
        frac_sigma_axis = np.linspace(0.05, 0.95, 30)

    tot = 2*sample_size*len(nc_list)*len(dd_list)*len(beta_ratio_list)*len(frac_sigma_axis)

    res = {}
    block_matrices_dict = {}

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

                            data = ([],[],[],[])
                            block_matrices = []

                            for i in range(sample_size):
                                local_params = get_local_params(N, D, n, sigma, target_degrees)
                                SD = sample_model(global_params, local_params, opt_params, 
                                                 average_k, rng, n)

                                
                                define_communities(SD, n, reassign=True)
                                measure_stuff(SD, n, data)
                                block_matrices.append(SD.block_mat.astype(float).tolist())
                                m, S, r, degrees = do_measurements(SD, n)
                                #represent(SD, n, f, S)

                                pbar.update(1)
                            res[key+'-m'] = data[0]
                            res[key+'-S'] = data[1]
                            res[key+'-r'] = data[2]
                            res[key+'-degrees'] = data[3]
                            block_matrices_dict[key] = block_matrices
    filepath = '../../../scratch/data/'+args.filename
    #filepath=args.filename
    with open(filepath+'.json', 'w') as write_file:
        json.dump(res, write_file, indent=4)
    with open(filepath+'_blockmatrices.json', 'w') as write_file:
        json.dump(block_matrices_dict, write_file, indent=4)

if __name__=='__main__':
    main()

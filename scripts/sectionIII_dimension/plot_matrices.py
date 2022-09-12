#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plots example connectivity matrices in S1 and S2 model

Author: Béatrice Désy

Date : 02/28/2022
"""


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *

beta_ratio = 100.
average_k = 10.
N=350

opt_params = {'tol':1e-1, 
            'max_iterations': 1000, 
            'perturbation': 0.1,
            'verbose':True}
rng = np.random.default_rng()

all_kappas=True

if all_kappas:
    target_degrees = get_target_degree_sequence(average_k, 
                                                N, 
                                                rng, 
                                                'exp',
                                                sorted=False) 
else:
    target_degrees = np.ones(N)*average_k

cmaps = ['Purples', 'Blues']

for D in [1,2]:
    coordinates = sample_uniformly_on_hypersphere(N, D)
    global_params = get_global_params_dict(N, D, beta_ratio*D, 0.01)
    local_params = {'coordinates':coordinates, 
                    'kappas': target_degrees+1e-3, 
                    'target_degrees':target_degrees, 
                    'nodes':np.arange(N)}
    SD = ModelSD()
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()
    
    SD.optimize_kappas(rng)
    SD.reassign_parameters()

    SD.build_probability_matrix(order='theta') 
    plt.figure(figsize=(3,3))
    plt.imshow(np.log10(SD.probs+1e-4), cmap = cmaps[D-1], vmin=np.log10(1e-5))
    #plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('mat{}.svg'.format(D), dpi=600, format='svg')
    plt.show()

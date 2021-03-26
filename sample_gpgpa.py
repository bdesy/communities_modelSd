#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Samples a graph from Sd model from given hidden variables
sequences.

Author: Béatrice Désy

Date : 17/03/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import subprocess

if __name__ == "__main__":
	# Sets variables
	path_to_hidvar = 'data/graph1000_poisson_gpa_S1_hidvar.dat'
	D=1
	beta=3.
	# Loads hidden variables
	kappa = (np.loadtxt(path_to_hidvar, dtype=str).T[1]).astype('float')
	thetas = (np.loadtxt(path_to_hidvar, dtype=str).T[2]).astype('float')
	# Computes average kappa
	average_kappa = np.mean(kappa)
	print(average_kappa)
	# Sets initial value for mu
	mu = 0.0421204717903987
	# Compiles the cpp code
	p = subprocess.Popen(['g++', '-O3', '-std=c++11', 'geometric_Sd_model/examples/generate_edgelist_from_modelSD.cpp', '-o', 'generate_edgelist_from_modelSD']) 

	pg = subprocess.Popen(['./generate_edgelist_from_modelSD', '-n', '-d', str(D), '-t', '-b', str(beta), path_to_hidvar])
	pg.wait()




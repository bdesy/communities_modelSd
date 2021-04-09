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
import argparse
import ast

if __name__ == "__main__":


# Parse input parameters

	parser = argparse.ArgumentParser()
	parser.add_argument('--path', '-p', type=str,
	                    help='path to the hidden variable file')
	args = parser.parse_args()

	# Sets variables
	path_to_hidvar = args.path+'.dat'
	D=1
	file = open(args.path+'_params.txt', 'r')
	contents = file.read()
	dictionary = ast.literal_eval(contents)
	file.close()
	beta = dictionary['beta']
	mu = dictionary['mu']

	# Loads hidden variables
	kappa = (np.loadtxt(path_to_hidvar, dtype=str).T[1]).astype('float')
	thetas = (np.loadtxt(path_to_hidvar, dtype=str).T[2]).astype('float')

	# Compiles the cpp code
	p = subprocess.Popen(['g++', '-O3', '-std=c++11', 'geometric_Sd_model/examples/generate_edgelist_from_modelSD.cpp', '-o', 'generate_edgelist_from_modelSD']) 
	p.wait()
	pg = subprocess.Popen(['./generate_edgelist_from_modelSD', '-n', '-d', str(D), '-t', '-b', str(beta), '-m', str(mu), path_to_hidvar])
	pg.wait()




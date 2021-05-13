#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Samples a graph from Sd model from given hidden variables
sequences.

Author: Béatrice Désy

Date : 17/03/2021
"""

import matplotlib.pyplot as plt
from truc import ModelSD
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

	# Sets paths

	path_to_hidvar = args.path+'.dat'
	path_to_params = args.path+'_params.txt'

	# Instanciate the model

	mod = ModelSD()
	mod.load_parameters(path_to_params)

	# Compiles the cpp code
	p = subprocess.Popen(['g++', '-O3', '-std=c++11', 'geometric_Sd_model/examples/generate_edgelist_from_modelSD.cpp', '-o', 'generate_edgelist_from_modelSD']) 
	p.wait()
	pg = subprocess.Popen(['./generate_edgelist_from_modelSD', '-n', '-d', str(mod.D), '-t', '-b', str(mod.beta), '-m', str(mod.mu), path_to_hidvar])
	pg.wait()




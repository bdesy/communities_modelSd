#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Assess my expression for the pdf of angular length of edges

Author: Béatrice Désy

Date : 13/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from geometric_functions import *
from time import time
import argparse
from comS2_util import *
import os

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nb_nodes', type=int,
                        help='number of nodes in the graph')
parser.add_argument('-nc', '--nb_communities', type=int,
                        help='number of communities to put on the sphere')
parser.add_argument('-dd', '--degree_distribution', type=str,
                        choices=['poisson', 'exp', 'pwl'],
                        help='shape of the degree distribution')
parser.add_argument('-o', '--order', type=bool, default=False,
                        help='whether or not to order the matrix with theta')
parser.add_argument('-s', '--sigma', type=float,
                        help='dispersion of points of angular clusters in d=1')
parser.add_argument('-br', '--beta_ratio', type=float,
                        help='value of beta for d=1')
args = parser.parse_args() 
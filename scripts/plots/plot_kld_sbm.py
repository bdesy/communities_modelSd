#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Script to plot KL divergence between DC-SBM and hyperbolic random graphs

Author: Béatrice Désy

Date : 20/10/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import interpolate

#ddistributions = ['poisson']
#nc = 2

filepath = '../data/kld/test_4comms_pwl_unordered'


with open(filepath+'.json', 'r') as read_file:
    data = json.load(read_file)

with open(filepath+'_params.json', 'r') as read_file:
	params = json.load(read_file)

x_axis = np.array(params['beta_list'])

colors=['red', 'darkcyan', 'coral', 'olivedrab']
formats=['.', 'o', '^', 'v']

plt.figure(figsize=(10,5))

for D in [1, 2, 3]:
	y_axis = []
	std_axis = []
	for beta in params['beta_list']:
		if beta > D:
			key = 'S{}'.format(D)+'beta{}'.format(beta)
			y_axis.append(np.mean(np.array(data[key])))
			std_axis.append(np.std(np.array(data[key])))
	plt.plot(x_axis[-len(y_axis):]/D, y_axis, linestyle='-', linewidth=1, 
				marker=formats[D], label=r'$d={}$'.format(D), c=colors[D])
	y_axis, std_axis = np.array(y_axis), 3*np.array(std_axis)
	#plt.fill_between(x_axis[-len(y_axis):]/D, y_axis-std_axis, y_axis+std_axis, alpha=0.3, color=colors[D])
plt.xlabel(r'$\beta/d$')
plt.ylabel(r'$\langle D_{KL} (p||q)\rangle$')
plt.xlim(0,10)
plt.legend()
plt.title(filepath)
plt.show()




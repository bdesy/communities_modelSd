#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Plot figure 3 from Muscoloni 2019 preprint 
« Angular separability of data clusters or network communities 
in geometrical space and its relevance to hyperbolic embedding »
with T axis rescale with dimension.

Author: Béatrice Désy

Date : 21/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'serif':['Computer Modern Roman'], 'size':14})
rc('text', usetex=True)
m=3

d1a = np.loadtxt('RA1-LE-2D.txt', delimiter=', ').T
d1b = np.loadtxt('RA1-nclSO-2D.txt', delimiter=', ').T
d2a = np.loadtxt('RA1-LE-3D-s.txt', delimiter=', ').T
d2b = np.loadtxt('RA1-nclSO-3D-s.txt', delimiter=', ').T

plt.plot(d1a[0], d1a[1], 'o', label='RA1-LE-2D', c='red', ms=m)
plt.plot(d1b[0], d1b[1], 'o', label='RA1-nclSO-2D', c='k', ms=m)
plt.plot((2*d2a[0]), d2a[1], label='RA1-LE-3D', c='red')
plt.plot((2*d2a[0]), d2a[1],':', label='RA1-nclSO-3D', c='k')

plt.ylabel('ASI')
#plt.xlabel(r'$\beta/d$')
plt.xlabel(r'$Td$')
plt.legend()
plt.savefig('fig_rescale_Td', dpi=200)
plt.show()
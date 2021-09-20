#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Studies angular distributions on the GPA.

Author: Béatrice Désy

Date : 22/03/2021
"""

import numpy as np
import matplotlib.pyplot as plt
from gpgpa import compute_angular_coordinates_gpa

N = 2000
y = 2.5
V = 0.

for i in range(20):
    phis = compute_angular_coordinates_gpa(N, y, V)

#plt.polar(phis, np.ones(N), 'o', ms=2)
#plt.show()

    plt.hist(phis, bins=120, color='darkcyan')
    plt.xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'2$\pi$'])
    plt.ylabel('Nombre de noeuds')
    plt.xlabel('Angle')
    #plt.savefig('data/angular_batch_test/test20{}'.format(i))
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 200)
    plt.close()
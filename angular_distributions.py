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

N = 100
y = 2.5
V = 0.

phis = compute_angular_coordinates_gpa(N, y, V)

plt.polar(phis, np.ones(N), 'o', ms=2)
plt.show()

plt.hist(phis, bins=120)
plt.show()
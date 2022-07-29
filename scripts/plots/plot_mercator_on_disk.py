#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot a network in hyperbolic plane whith edges following geodesics from the conformal disk model.

Author: Béatrice Désy

Date : 29/04/2021
"""

import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gaussian_kde
import sys
sys.path.insert(0, '../../src/')
from hyperbolic_random_graph import *
from hrg_functions import *
from geometric_functions import *
sys.path.insert(0, '../exp-overlap/')
from overlap_util import *


# Define sub-recipes

def compute_radius(kappa, kappa_0, R_hat):
    hyperbolic_r = (R_hat - 2*np.log(kappa/kappa_0))/R_hat
    #euclidean_r = np.tanh(hyperbolic_r/2)
    return hyperbolic_r

def get_hyperbolic_edge(t1, t2, r1, r2, R_max):
    a = r1*np.exp(1j*t1)
    b = r2*np.exp(1j*t2)
    q = a*(1+abs(b)**2) - b*(1+abs(a)**2)
    q /= a*np.conj(b) - np.conj(a)*b
    r = abs(a-q)
    return q, r

def obtain_angles_array(x, y, q, r, num=360):
    anglex = np.angle(x - q)
    angley = np.angle(y - q)
    dist = compute_angular_distance(np.array(anglex), np.array(angley), dimension=1, euclidean=False)
    arc = abs(angley - anglex)%np.pi
    if abs(arc-dist)<1e-5:
        angles = np.linspace(anglex, angley, num=360)
    else:
        arr = np.array([anglex, angley])
        angles1 = np.linspace(np.min(arr), -np.pi, num=180)
        angles2 = np.linspace(np.pi, np.max(arr), num=180)
        angles = np.hstack((angles1, angles2))
    return angles


def plot_circle_arc(ax, x, y, center, radius=1, num=360, color='k'):
    t = obtain_angles_array(x, y, center, radius, num=num)
    #t = np.linspace(0, 2*np.pi, num)
    r = np.ones(t.shape)*radius
    unitcircle = radius*np.exp(1j*t) + center
    ax.plot(unitcircle.real, unitcircle.imag, c=color, linewidth=0.5)


def change_to_lorenz_coordinates(coordinates, radiuses, zeta=1.):
    x = (1./zeta)*np.cosh(zeta*radiuses)
    y = (1./zeta)*np.sinh(zeta*radiuses)*np.cos(coordinates)
    z = (1./zeta)*np.sinh(zeta*radiuses)*np.sin(coordinates)
    return x,y,z

def project_on_disk(x,y,z):
    xb = y/(1+x)
    yb = z/(1+x)
    return xb, yb

# Parse input parameters

parser = argparse.ArgumentParser()
parser.add_argument('-ok', '--optimize_kappas', type=bool, default=False)
parser.add_argument('--mode', '-m', type=str, default='normal',
                    help='optional presentation mode for bigger stuff')
parser.add_argument('--save', '-s', type=bool, default=False,
                    help='to save or not the figure')
args = parser.parse_args()

# Load graph data and parameters
sampling=True
if sampling:
    N = 200
    nb_com = 5
    D=1
    frac_sigma_max = 0.3
    sigma1 = get_sigma_max(nb_com, 1)*frac_sigma_max
    sigma2 = get_sigma_max(nb_com, 2)*frac_sigma_max

    beta_r = 3.5
    rng = np.random.default_rng()

    #sample angular coordinates on sphere and circle
    coordinatesS1, centers = get_communities_coordinates(nb_com, N, sigma1, 
                                                            place='equator',
                                                            output_centers=True)
    coordinates = (coordinatesS1.T[0]).reshape((N, 1))
    centers = (centers.T[0]).reshape((nb_com, 1))

    #graph stuff
    mu = 0.01
    average_k = 4.
    target_degrees = get_target_degree_sequence(average_k, 
                                                N, 
                                                rng, 
                                                'pwl',
                                                sorted=False) 

    #optimization stuff
    opt_params = {'tol':1e-1, 
                'max_iterations': 1000, 
                'perturbation': 0.1,
                'verbose':True}


    SD = ModelSD()
    global_params = get_global_params_dict(N, D, beta_r*D, mu)
    local_params = {'coordinates':coordinates, 
                        'kappas': target_degrees+1e-3, 
                        'target_degrees':target_degrees, 
                        'nodes':np.arange(N)}
    SD.specify_parameters(global_params, local_params, opt_params)
    SD.set_mu_to_default_value(average_k)
    SD.reassign_parameters()

    if args.optimize_kappas:
        SD.optimize_kappas(rng)
        SD.reassign_parameters()

    labels = np.arange(nb_com)
    SD.communities = get_communities_array_closest(N, D, SD.coordinates, centers, labels)

    SD.build_probability_matrix() 

A = SD.sample_random_matrix()

#plt.imshow(A)
#plt.colorbar()
#plt.show()
G = nx.from_numpy_matrix(A)

# Compute radii 

kappa_0 = np.min(SD.kappas)
R_hat = 2*np.log(N / (mu*np.pi*kappa_0**2))

radiuses = compute_radius(SD.kappas, kappa_0, R_hat)


kappas = SD.kappas
thetas = SD.coordinates    

# Set plotting sizes
if args.mode=='normal':
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    ms=[3,2]
elif args.mode=='presentation':
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    ms=[10,8]

# Plot figure

fig = plt.figure(figsize=(5,5))
rect = [0.1, 0.1, 0.8, 0.8]
ax = fig.add_axes(rect, )

imag_coord = radiuses * np.exp(1j*thetas).flatten()

i=0
for edge in G.edges():
    n1, n2 = edge
    center, radius = get_hyperbolic_edge(thetas[n1], thetas[n2], radiuses[n1], radiuses[n2], R_hat)
    plot_circle_arc(ax, imag_coord[n1], imag_coord[n2], center, radius, color='k')
    i+=1

for node in G.nodes():
    community = SD.communities[node]
    color = 'darkcyan'#plt.cm.tab10(community%10)
    x,y = imag_coord[node].real, imag_coord[node].imag
    plt.plot(x,y, 'o', ms=ms[0], c='white')
    plt.plot(x,y, 'o', ms=ms[1], c=color)

sanity=False
if sanity:
    xl,yl,zl = change_to_lorenz_coordinates(coordinates.flatten(), np.array(radiuses), zeta=1.)
    xb, yb = project_on_disk(xl,yl,zl)
    plt.plot(xb,yb,'o', c='green', ms=1)

clean=True
if clean:
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.ylim(-1.,1.)
    plt.xlim(-1.,1.)

if args.save:
    plt.savefig('fig1_'+args.community, dpi=600)

out_circ = np.exp(1j*np.linspace(0,2*np.pi, 1000))
plt.plot(out_circ.real, out_circ.imag, c='tomato')
plt.show()
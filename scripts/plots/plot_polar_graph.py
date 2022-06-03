#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Plot a network in hyperbolic plane whilst accentuating community structure and node densities.

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
    return R_hat - 2*np.log(kappa/kappa_0)

def get_hyperbolic_edge(t1, t2, r1, r2, R_max):
    a = r1/R_max*np.exp(1j*t1)
    b = r2/R_max*np.exp(1j*t2)
    q = a*(1+abs(b)**2) - b*(1+abs(a)**2)
    q /= a*np.conj(b) - np.conj(a)*b
    r = abs(a-q)
    return q, r

def smaller_arc(arc_angle):
    result = arc_angle.copy()
    if abs(result[0]-result[1]) > np.pi:
        if result[0]<result[1]:
            result[0] += 2*np.pi
        else:
            result[1] += 2*np.pi
    return np.sort(result)
    
def arc_angle(pts, center):
    angles = np.angle(pts-center)
    return smaller_arc(angles)

def plot_circle_arc(ax, radius=1, lower_lim=0, upper_lim=2*np.pi, num=360, R_max=1, color='k'):
    t = np.linspace(lower_lim, upper_lim, num=360)
    unitcircle = R_max*radius*np.exp(1j*t)
    ax.plot(unitcircle.real, unitcircle.imag,c=color)

# Parse input parameters

parser = argparse.ArgumentParser()
parser.add_argument('-ok', '--optimize_kappas', type=bool, default=False)
parser.add_argument('--mode', '-m', type=str, default='normal',
                    help='optional presentation mode for bigger stuff')
parser.add_argument('--density', '-d', type=bool, default=False,
                    help='to turn off angular density plot on the circumference')
parser.add_argument('--save', '-s', type=bool, default=False,
                    help='to save or not the figure')
args = parser.parse_args()

# Load graph data and parameters
sampling=True
if sampling:
    N = 10
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
    coordinatesS1 = (coordinatesS1.T[0]).reshape((N, 1))
    centersS1 = (centers.T[0]).reshape((nb_com, 1))

    coordinates = [coordinatesS1]
    centers = [centersS1]

    #graph stuff
    mu = 0.01
    average_k = 3.
    target_degrees = get_target_degree_sequence(average_k, 
                                                N, 
                                                rng, 
                                                'exp',
                                                sorted=False) 

    #optimization stuff
    opt_params = {'tol':1e-1, 
                'max_iterations': 1000, 
                'perturbation': 0.1,
                'verbose':True}


    SD = ModelSD()
    global_params = get_global_params_dict(N, D, beta_r*D, mu)
    local_params = {'coordinates':coordinates[D-1], 
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
    SD.communities = get_communities_array_closest(N, D, SD.coordinates, centers[D-1], labels)

    SD.build_probability_matrix() 

A = SD.sample_random_matrix()

plt.imshow(A)
plt.colorbar()
plt.show()
G = nx.from_numpy_matrix(A)
print(len(G.edges()))

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
    ms=[5,3]
elif args.mode=='presentation':
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    ms=[10,8]

# Plot figure

fig = plt.figure(figsize=(3.375,3))
rect = [0.1, 0.1, 0.8, 0.8]
ax = fig.add_axes(rect, projection='polar')
for edge in G.edges():
    n1, n2 = edge
    center, radius = get_hyperbolic_edge(thetas[n1], thetas[n2], radiuses[n1], radiuses[n2], R_hat)
    #plot_circle_arc(ax, center, radius, color='k', R_max=R_hat)
    ax.plot([thetas[n1], thetas[n2]], [radiuses[n1], radiuses[n2]], c='k', linewidth=1.)

for node in G.nodes():
    community = SD.communities[node]
    color = 'darkcyan'#plt.cm.tab10(community%10)
    ax.plot(thetas[node], radiuses[node], 'o', ms=ms[0], c='white')
    ax.plot(thetas[node], radiuses[node], 'o', ms=ms[1], c=color)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)

if args.density:
    tt = np.linspace(0.0,2*np.pi, 1000)
    kde = gaussian_kde(thetas.T[0], bw_method=0.02)
    upper = kde(tt)
    '''
    upper /= np.max(upper)
    upper *= (R_hat*1.1 - R_hat)
    upper += R_hat
    lower = np.ones(1000)*R_hat
    #ax.fill_between(tt, lower, upper, alpha=0.3, color='darkcyan')
    #ax.plot(tt, upper, c='darkcyan', linewidth=1)

    xx = np.linspace(0.01, R_hat, 1000)
    kde_r = gaussian_kde(radiuses, bw_method=0.07)
    yy = np.array(kde_r(xx))
    yy = yy/np.max(yy)*R_hat/3

    r_den = np.sqrt(xx**2 + yy**2)
    th_den = np.arccos(xx / r_den)'''

    
    used_theta=np.linspace(0, 2*np.pi, 1000)
    used_rad = np.linspace(R_hat*1.05, R_hat*1.15, 1000)
    X,Y = np.meshgrid(used_theta, used_rad) #rectangular plot of polar data
    truc = upper.reshape((1,1000))
    density = np.repeat(truc, repeats=1000, axis=0)
    #print(density.shape)
    ax.pcolormesh(X, Y, density, cmap='Purples')

if args.save:
    plt.savefig('fig1_'+args.community, dpi=600)

plt.ylim(0., R_hat)
plt.plot(np.linspace(0, 2*np.pi, 1000), np.ones(1000)*R_hat, c='k')
plt.show()




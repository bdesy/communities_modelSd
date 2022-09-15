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

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)

cray = '#484747'

def S1_angular_distance(coord_i, coord_j):
    return np.pi - abs(np.pi - abs(coord_i - coord_j))

def get_hyperbolic_edge(t1, t2, r1, r2): # from here https://github.com/brsr/math/blob/b55c538137ae8855b150c990ef54ef4b2e9ba542/Lines%20on%20the%20Hyperbolic%20Poincare%20Disk.ipynb
    a = r1*np.exp(1j*t1)
    b = r2*np.exp(1j*t2)
    q = a*(1+abs(b)**2) - b*(1+abs(a)**2)
    if abs(a*np.conj(b) - np.conj(a)*b)>1e-5:
        q /= a*np.conj(b) - np.conj(a)*b
    r = abs(a-q)
    return q, r

def obtain_angles_array(x, y, q, r, num=360):
    anglex = np.angle(x - q)
    angley = np.angle(y - q)
    dist = S1_angular_distance(anglex, angley)
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
    ax.plot(unitcircle.real, unitcircle.imag, c=color, linewidth=0.1, zorder=1, alpha=0.1)

def change_to_lorenz_coordinates(coordinates, radiuses, zeta=1.):
    x = (1./zeta)*np.cosh(zeta*radiuses)
    y = (1./zeta)*np.sinh(zeta*radiuses)*np.cos(coordinates)
    z = (1./zeta)*np.sinh(zeta*radiuses)*np.sin(coordinates)
    return x,y,z

def project_on_disk(x,y,z):
    xb = y/(1+x)
    yb = z/(1+x)
    return xb, yb

def obtain_ball_radius(theta, r_hyp, zeta):
    x,y,z = change_to_lorenz_coordinates(theta, r_hyp, zeta=zeta)
    xb, yb = project_on_disk(x,y,z)
    return np.sqrt(xb**2 + yb**2)

# Load graph data and coordinates

geometric_data =  np.loadtxt('data/WorldAirportsOFGC_00.inf_coord',comments="#",dtype='str')
graph_data = np.loadtxt('data/WorldAirportsOFGC.edge',comments="#",dtype='str')
meta_data = np.loadtxt('data/WorldAirportsOFGC.meta', comments='#', dtype='str')
R_hat = 35.6456

G = nx.from_edgelist(graph_data)

thetas_dict = {}
radiuses_dict = {}
imag_coord = {}
for line in geometric_data:
    key = line[0]
    theta = float(line[2]) 
    r_h = float(line[3])
    r = r_h/R_hat
    #r = obtain_ball_radius(theta, r_h, zeta=1.)
    #print(r)
    thetas_dict[key] = theta
    radiuses_dict[key] = r
    imag_coord[key] = r * np.exp(1j*theta)

continents = {}
for line in meta_data:
    key = line[0]
    continents[key] = line[2]

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
ms = [2,1.8]
red = '#960C0C'
green = '#97CC04'
orange = '#F45D01'
violet = '#B50388'
yellow = '#FDD235'
blue = '#2D7DD2'
colors = {'NA':red, 'SA':green, 'EU':orange, 
            'OC':violet, 'AS':yellow, 'AF':blue}

# Plot figure

fig = plt.figure(figsize=(3.4,3.4))
rect = [0.05, 0.05, 0.9, 0.9]
ax = fig.add_axes(rect, )

i=0
for edge in G.edges():
    n1, n2 = edge
    center, radius = get_hyperbolic_edge(thetas_dict[n1], thetas_dict[n2], radiuses_dict[n1], radiuses_dict[n2])
    plot_circle_arc(ax, imag_coord[n1], imag_coord[n2], center, radius, color='k')
    i+=1

labels= {red:'North America', green:'South America', orange:'Europe',
        violet:'Oceania', yellow:'Asia', blue:'Africa'}

for c in list(colors.values()):
    plt.scatter(100,100, marker='o', s=10, c=c, edgecolor=cray, linewidths=0.4, label=labels[c])

for node in G.nodes():
    color = colors[continents[node]]
    x,y = imag_coord[node].real, imag_coord[node].imag
    #plt.plot(x,y, 'o', ms=ms[0], c=cray)
    plt.scatter(x,y, marker='o', s=5, c=color, edgecolor=cray, linewidths=0.2, zorder=10)

sanity=False
if sanity:
    xl,yl,zl = change_to_lorenz_coordinates(geometric_data.T[2].astype(float), geometric_data.T[3].astype(float), zeta=1./15.)
    xb, yb = project_on_disk(xl,yl,zl)
    plt.plot(xb,yb,'o', c='green', ms=1)

clean=True
if clean:
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.ylim(-1.5,1.5)
    plt.xlim(-1.5,1.5)


out_circ = 1.05*np.exp(1j*np.linspace(0,2*np.pi, 1000))
plt.plot(out_circ.real, out_circ.imag, c=cray, linewidth=0.5)
plt.legend(ncol=3, frameon=False, loc='lower center', borderpad=0., columnspacing=1., handletextpad=0.1)
plt.savefig('my_airports.pdf', dpi=600, format='pdf')
plt.show()
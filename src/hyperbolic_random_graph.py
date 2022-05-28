#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Hyperbolic random graph model with heterogeneous angular positions

Author: Béatrice Désy

Date : 03/01/2022
"""

import numpy as np
from tqdm import tqdm
from optimization_kappas import optimize_kappas
from model_parameters import *
from hrg_functions import *

class ModelSD():
    def __init__(self):
        self.gp = GlobalParameters()
        self.lp = LocalParameters()
        self.op = OptimizationParameters()
        self.reassign_parameters()

    def specify_parameters(self, global_params, local_params, opt_params):
        self.gp.specify(global_params)
        self.lp.specify(local_params)
        self.op.specify(opt_params)
        self.reassign_parameters()

    def reassign_parameters(self):
        self.N, self.D = self.gp.N, self.gp.dimension
        self.R, self.euclidean = self.gp.radius, self.gp.euclidean
        self.mu, self.beta = self.gp.mu, self.gp.beta
        self.nodes = self.lp.nodes
        self.coordinates = self.lp.coordinates
        self.kappas = self.lp.kappas
        self.target_degrees = self.lp.target_degrees

    def get_H2_stuff(self):
        self.kappa_0 = 1.0 #POULET
        self.R_max = 2*np.log(self.N / (self.mu*np.pi*self.kappa_0**2)) #POULET

    def load_all_parameters_from_file(self, path):
        self.gp.load_from_file(path+'gp.pkl')
        self.lp.load_from_file(path+'lp.pkl')
        self.op.load_from_file(path+'op.pkl')
        self.reassign_parameters()

    def save_all_parameters_to_file(self, path):
        self.gp.save_to_file(path+'gp.pkl')
        self.lp.save_to_file(path+'lp.pkl')
        self.op.save_to_file(path+'op.pkl')

    def set_mu_to_default_value(self, average_k):
        self.mu = compute_default_mu(self.D, self.beta, average_k)
        self.gp.mu = self.mu

    def optimize_kappas(self, rng):
        if self.op.verbose==True:
            print('Optimizing latent degrees kappas')
        kappas, success = optimize_kappas(rng, self.gp.get_njitable(), self.lp, self.op)
        self.kappas = kappas
        if self.op.verbose==True:
            print('Optimization has succeeded : {}'.format(success))

    def compute_all_expected_degrees(self):
        self.expected_degrees = compute_all_expected_degrees(self.N, 
                                                            self.coordinates, 
                                                            self.kappas, 
                                                            self.gp)

    def build_probability_matrix(self, order=None):
        if (type(order) is str) and (order=='theta'):
            order = np.argsort(self.coordinates.T[0])
        self.probs = build_probability_matrix(self.N, 
                                            self.coordinates, 
                                            self.kappas, 
                                            self.gp.get_njitable(),
                                            order=order)

    def build_angular_distance_matrix(self, order=None):
        if type(order) is str:
            if order=='theta':
                order = np.argsort(self.coordinates.T[0])
        self.angular_distance_matrix = build_angular_distance_matrix(self.N, 
                                                    self.coordinates, 
                                                    self.D, 
                                                    euclidean=self.euclidean,  
                                                    order=order)

    def build_hyperbolic_distance_matrix(self, order=None):
        self.get_H2_stuff()
        if type(order) is str:
            if order=='theta':
                order = np.argsort(self.coordinates.T[0])
        self.hyperbolic_distance_matrix = build_hyperbolic_distance_matrix(self.N, 
                                                    self.coordinates, 
                                                    self.kappas,
                                                    self.D, 
                                                    euclidean=self.euclidean,  
                                                    kappa_0=self.kappa_0, 
                                                    R_max=self.R_max,
                                                    order=order)

    def sample_random_matrix(self):
        rand = np.triu(np.random.random(size=(self.N, self.N)), k=1)
        rand += rand.T
        return np.where(self.probs>rand, 1, 0)

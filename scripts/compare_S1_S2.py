import argparse
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../src/')
from hyperbolic_random_graphs import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-p', type=str, default='hidvar',
                        help='path to the S2 hidden variable file')
    args = parser.parse_args()

    # Set random number generator

    rng = np.random.default_rng()

    # Sets S2 model and parameters

    modelS2 = ModelSD()
    modelS2.load_parameters(args.filename+'_params.txt')
    modelS2.load_hidden_variables(args.filename+'.dat')
    N = modelS2.N
    mu = modelS2.mu
    beta = modelS2.beta

    # Set parameters for S1 kappas optimization

    tol = 10e-2
    max_iterations = 2*N

    thetas = np.copy(modelS2.coordinates.T[0].reshape((modelS2.N, 1)))

    plt.hist(thetas, bins=120, color='darkcyan')
    plt.xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'2$\pi$'])
    plt.ylabel('Nombre de noeuds', fontsize=20)
    plt.xlabel('Angle', fontsize=20)
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 100)
    plt.show()


    R = compute_radius(N, D=1)

    kappas = np.copy(modelS2.target_degrees)
    print(thetas.shape)
    kappas_opt_1d = optimize_kappas(N, tol, max_iterations,  
                                    thetas, kappas, 
                                    R, beta, mu, 
                                    modelS2.target_degrees, rng, 
                                    D=1, verbose=True, perturbation=0.1)
    expected_degrees_1d = compute_all_expected_degrees(N, thetas, kappas_opt_1d, R, beta, mu, D=1)


    # Plots target degrees and kappas for S1

    plt.plot(modelS2.target_degrees, 'o', label='target degrees', c='purple', ms=7)
    plt.plot(expected_degrees_1d, 'o', label='expected degrees in ensemble', c='darkcyan', ms=3)
    plt.plot(kappas_opt_1d, '^', label='kappas', c='coral', ms=2)
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


    # Saves the output in a format that the C++ code eats

    save=True

    if save:
        vertices = np.array(['v{:05d}'.format(i) for i in range(N)])
        data1d = np.column_stack((vertices, kappas_opt_1d, thetas, modelS2.target_degrees))
        filename1d = args.filename.replace('S2', 'S1')

        header1d = 'vertex       kappa       theta      target degree'

        np.savetxt(filename1d+'.dat', data1d, delimiter='       ', fmt='%s',
                    header=header1d)

        params = {'mu':mu, 'beta':beta, 'dimension':1, 'radius':R}
        params_file = open(filename1d+'_params.txt', 'a')
        params_file.write(str(params))
        params_file.close()
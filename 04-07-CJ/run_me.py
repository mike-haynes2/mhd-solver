import numpy as np
import math as m
from scipy import constants
import matplotlib.pyplot as plt
import os
import mhd as Balbas
from datetime import datetime
from joblib import Parallel, delayed
from functools import partial



def initialize(name, num_vars, nx, gamma, sigmoid_value=0.0):  # meshOBJ, alpha, Tmax, t, nt, tL, num_vars, Bx, gamma, nx, lam, n_plots
    """returns the initial conditions based on the type of problem specified by the 'name' variable"""
    # create object to store most recent iterations for all variables
    meshOBJ = np.empty((2, num_vars, nx))

    # choose the appropriate initial conditions based on the problem type
    match name:
        case 'sigmoid':
            def sig(x): return 1 / (1 + np.exp(-sigmoid_value * x))
            p0 = sig(np.linspace(-1, 1, nx))
            rho = p0.copy()
            en = p0 / (gamma - 1) + 1./2. # I think the 1/2 is the additional terms
            plt.plot(np.linspace(-1, 1, nx), en)
            meshOBJ[0, 0, :] = rho ; meshOBJ[0, -1, :] = en
            meshOBJ[0, 1:-1, 0:(nx // 2)] = np.array([[ 0.0, 0.0, 0.0, 1.0, 0.0]]).T
            meshOBJ[0, 1:-1, (nx // 2):nx] = np.array([[0.0, 0.0, 0.0, -1.0, 0.0]]).T  # WHY DIVIDE BY 10?
        case 'Dai & Woodward':
            pass
            ######### THIS CASE REQUIRES BX TO BE CHANGING SO MAYBE NOT ##############
            # ro0_neg = 1.08; pr0_neg = 0.95
            # ux0 = 1.2; uy0 = 0.01; uz0 = 0.5
            # bx0 = 2. / np.sqrt(16 * np.arctan(1.)); by0 = 3.6/ np.sqrt(16 * np.arctan(1.)); bz0 = 2. / np.sqrt(16 * np.arctan(1.))
            # b_vec_neg = np.array([bx0, by0, bz0])
            # u_vec_neg = np.array([ux0, uy0, uz0])

            # ro0_pos = 1.; pr0_pos = 1
            # ux0 = 0.; uy0 = 0.; uz0 = 0.
            # bx0 = 2. / np.sqrt(16 * np.arctan(1.)); by0 = 4. / np.sqrt(16 * np.arctan(1.)); bz0 = 2. / np.sqrt(16 * np.arctan(1.))
            # b_vec_pos = np.array([bx0, by0, bz0])
            # u_vec_pos = np.array([ux0, uy0, uz0])
            # meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.08, 1.2 / 1.08, 0.01 / 1.08, 0.5 / 1.08, 1.0, 0.0, e0]]).T
            # meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0 / 10.]]).T

        case 'Brio & Wu':

            e0 = (1. / (gamma - 1.)) - (1. / 2.) # ASK ABOUT THIS. WHERE DO WE INCORPERATE PRESSURE TO IC?

            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0 / 10.]]).T # WHY DIVIDE BY 10?

            # ro0_neg = 1.; pr0_neg = 1.
            # ux0 = 0.; uy0 = 0.; uz0 = 0.
            # bx0 = .75; by0 = 1.; bz0 = 0.
            # b_vec_neg = np.array([bx0, by0, bz0])
            # u_vec_neg = np.array([ux0, uy0, uz0])

            # ro0_pos = .125; pr0_pos = .1
            # ux0 = 0.; uy0 = 0.; uz0 = 0.
            # bx0 = .75; by0 = -1.; bz0 = 0
            # b_vec_pos = np.array([bx0, by0, bz0])
            # u_vec_pos = np.array([ux0, uy0, uz0])

        case 'slow shock':
            pass
            # ro0_neg = 1.368; pr0_neg = 1.769
            # ux0 = 0.269; uy0 = 1.0; uz0 = 0.
            # bx0 = 1.; by0 = 0.; bz0 = 0.
            # b_vec_neg = np.array([bx0, by0, bz0])
            # u_vec_neg = np.array([ux0, uy0, uz0])

            # ro0_pos = 1.; pr0_pos = 1.
            # ux0 = 0.; uy0 = 0.; uz0 = 0.
            # bx0 = 1.; by0 = 1.; bz0 = 0
            # b_vec_pos = np.array([bx0, by0, bz0])
            # u_vec_pos = np.array([ux0, uy0, uz0])
            e0 = (1. / (gamma - 1.)) - (1. / 2.)
            # B_X must also be 1.0 not .75
            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.368, 0.269 * 1.368, 1.0 * 1.368, 0.0 , 0.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0 / 10.]]).T

        case 'rarefaction':
            pass
            # ro0_neg = 1.; pr0_neg = 2.
            # ux0 = 0.; uy0 = 0.; uz0 = 0.
            # bx0 = 1.; by0 = 0.; bz0 = 0.
            # b_vec_neg = np.array([bx0, by0, bz0])
            # u_vec_neg = np.array([ux0, uy0, uz0])

            # ro0_pos = .2; pr0_pos = 0.1368
            # ux0 = 1.186; uy0 = 2.967; uz0 = 0.
            # bx0 = 1.; by0 = 1.6405; bz0 = 0
            # b_vec_pos = np.array([bx0, by0, bz0])
            # u_vec_pos = np.array([ux0, uy0, uz0])
            e0 = (1. / (gamma - 1.)) - (1. / 2.)
            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.2, 1.186 * 0.2, 2.967 * 0.2, 0.0, 1.6405, 0.0, e0 / 10.]]).T
        case _:
            raise ValueError(
                'invalid testing: use ""Dai & Woodward"" or  ""Brio & Wu"" or ""slow shock"" or ""rarefaction""')

    return meshOBJ

# name='Brio & Wu'
# nx = 200
#################################### Individual Alpha Run ####################################
# meshOBJ= initialize(name=name, num_vars=7, nx=nx, gamma=2)
# Balbas.balbas_one_dimension(meshOBJ, alpha=1.4, Tmax=.2, num_vars=7, Bx=.75, gamma=2, nx=nx, n_plots=20, CFL_safety=40, length=1, name=name, alpha_test=True)
#################################### Variational Alpha Runs ####################################
# fixed inputs for alpha variation runs
# os.mkdir(f'alpha_test_dir_case_{name}')
# meshOBJ= initialize(name=name, num_vars=7, nx=nx, gamma=2)
# fixed_work_alpha_run = partial(Balbas.balbas_one_dimension,
#      meshOBJ=meshOBJ, Tmax=.2, num_vars=7, Bx=.75, gamma=2, nx=nx, n_plots=5, CFL_safety=40, length=2, name=name, alpha_test=True)
# # alpha_vals = np.arange(.8, 4, .4)
# alpha_vals = [1, 1.4]
# # this is the parrallel run for all of the alpha values // -1 means it runs on all available cores set to whatever you actually want it to be
# Parallel(n_jobs=2)(delayed(fixed_work_alpha_run)(alpha=a) for a in alpha_vals)

# plotting alpha graphs (working in the testing.ipynb file)
name = 'sigmoid'
nx = 200
#################################### Individual Sigmoid Run ####################################
# meshOBJ= initialize(name=name, num_vars=7, nx=nx, gamma=2, sigmoid_value=10)
# Balbas.balbas_one_dimension(meshOBJ, alpha=1.4, Tmax=.2, num_vars=7, Bx=.75, gamma=2, nx=nx, n_plots=10, CFL_safety=40, length=2, name=name, alpha_test=False, sigmoid_test=True)
#################################### Sigmoid IC Runs ####################################
sigmoid_vals = np.arange(1, 50, 10); mesh_inputs = []
directory = f'sigmoid_test' # directory = 'sigmoid_test'
for sig in sigmoid_vals: mesh_inputs.append(initialize(name=name, num_vars=7, nx=nx, gamma=2, sigmoid_value=sig))
os.mkdir(f'sigmoid_test')
fixed_work_sigmoid_run = partial(Balbas.balbas_one_dimension,
     alpha=1.4, Tmax=.2, num_vars=7, Bx=.75, gamma=2, nx=nx, n_plots=5, CFL_safety=40, length=2, name=name, alpha_test=True)

args = list(zip(mesh_inputs, sigmoid_vals))
Parallel(n_jobs=-1)(delayed(fixed_work_sigmoid_run)(meshOBJ=mesh, sig=sig) for mesh, sig in args)


#################################### getting data after runs ####################################
# these are for all alpha/sigma and are organized by <alpha/sigma>_<time> for plotting or other purposes
B_y = {}; B_z = {}
rho = {}; en = {}
u_x = {}; u_y = {}; u_z = {}
for file in os.listdir(directory):
    full_path = os.path.join(directory, file)
    print(full_path)
    split_name = file.split('_')
    case = split_name[1]; var = split_name[3]; t = split_name[5] # val will be either alpha or sigma depending on what directory is commented
    data = np.load(full_path)
    B_y[f'{var}_{t}'] = data[f'B_y_{var}_{t}']
    B_z[f'{var}_{t}'] = data[f'B_z_{var}_{t}']
    u_x[f'{var}_{t}'] = data[f'u_x_{var}_{t}']
    u_y[f'{var}_{t}'] = data[f'u_y_{var}_{t}']
    u_z[f'{var}_{t}'] = data[f'u_z_{var}_{t}']
    en[f'{var}_{t}'] = data[f'en_{var}_{t}']
    rho[f'{var}_{t}'] = data[f'rho_{var}_{t}']





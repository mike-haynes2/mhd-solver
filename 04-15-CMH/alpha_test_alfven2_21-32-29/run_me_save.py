import numpy as np
import math as m
from scipy import constants
import matplotlib.pyplot as plt
import os
import mhd as Balbas
from datetime import datetime
from joblib import Parallel, delayed
from functools import partial


# added 04-16 for the Alfven wave case
from Library import calc_f


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
            meshOBJ[0, 1:-1, (nx // 2):nx] = np.array([[0.0, 0.0, 0.0, -1.0, 0.0]]).T  

        case 'Dai&Woodward':
            pass
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
            e0 = (1. / (gamma - 1.)) - (1. / 2.)
            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.08, 1.2 / 1.08, 0.01 / 1.08, 0.5 / 1.08, 1.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0 / 10.]]).T

        case 'Brio&Wu':

            e0 = (1. / (gamma - 1.)) - (1. / 2.) 

            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0 / 10.]]).T 

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

        case 'slow-shock':
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

        case 'alfven':
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
            meshOBJ[0, :, :] = np.array([[1., 0., 1., 1., 1., 0., e0]]).T
            # meshOBJ[0, :, (nx // 2):nx] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0 / 10.]]).T
        case 'alfven2':
            pass

            # initialize alfven wave according to this test scenario:
            # https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html
            # (also gives the expression to calculate error)

            amp = 1.e-05
            length = 20.
            # expression for the perturbation:
            # d U = amp * flux[i] * sin(2 pi x)
            Xs = np.linspace(start=-length/2.,stop=length/2.,num=nx)
            sins = np.sin(2.*np.pi*Xs)

            e0 = (1. / (gamma - 1.)) - (1. / 2.)
            # B_X must also be 1.0 not .75
            meshOBJ[0, :, :] = np.array([[1., 0., 1., 1., 1., 0., e0]]).T
            f_vals_A = calc_f(meshOBJ[0, :, :], 1., gamma)
            delta_vals = amp * f_vals_A * sins
            for i in range(1,num_vars):
                meshOBJ[0, i, :] += meshOBJ[0, i, :]  * delta_vals[i]
            



        case 'sod-shock':
            e0 = (1. / (gamma - 1.)) - (1. / 2.) 

            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, 0.0, 0.0, e0 / 10.]]).T 
        case _:
            raise ValueError(
                'invalid testing: use ""Dai&Woodward"" or  ""Brio&Wu"" or ""slow-shock"" or ""rarefaction""')

    return meshOBJ

def run(name='Brio&Wu', test=False, alpha=1.4, Tmax=.2,
                      num_vars=7, Bx=.75, gamma=2, nx=200, n_plots=20,
                        CFL_safety=40, length=2, alpha_test=False, sigmoid_value=0):

    formatted_time = datetime.now().strftime("%H-%M-%S")
    os.mkdir(f'sigmoid_test_{formatted_time}') # there is probably a better way to do this call specifically without rewriting it, but it works for now
    # os.mkdir(f'sigmoid_test_bruh')
    os.mkdir(f'alpha_test_{name}_{formatted_time}')
    if name == 'sigmoid' and  ~test:
        sigmoid_vals = np.arange(1, 50, 10); mesh_inputs = []
        for sig in sigmoid_vals: mesh_inputs.append(
            initialize(name=name, num_vars=num_vars, nx=nx, gamma=gamma, sigmoid_value=sig))

        fixed_work_sigmoid_run = partial(Balbas.balbas_one_dimension,
                                         alpha=alpha, Tmax=Tmax, num_vars=num_vars, Bx=Bx,
                                         gamma=gamma, nx=nx, n_plots=n_plots, CFL_safety=CFL_safety,
                                         length=length, name=name, alpha_test=False, start_time=formatted_time)

        args = list(zip(mesh_inputs, sigmoid_vals))
        Parallel(n_jobs=-1)(delayed(fixed_work_sigmoid_run)(meshOBJ=mesh, sig=sig) for mesh, sig in args)
    elif alpha_test:
        alpha_vals = np.arange(.8, 4, .4)
        meshOBJ = initialize(name=name, num_vars=num_vars, nx=nx, gamma=gamma, sigmoid_value=sigmoid_value)

        fixed_work_alpha_run = partial(Balbas.balbas_one_dimension,
                                       meshOBJ=meshOBJ, Tmax=Tmax, num_vars=num_vars, Bx=Bx, gamma=gamma, nx=nx, n_plots=n_plots,
                                       CFL_safety=CFL_safety, length=length, name=name, alpha_test=alpha_test, start_time=formatted_time)

        Parallel(n_jobs=-1)(delayed(fixed_work_alpha_run)(alpha=a) for a in alpha_vals)
    else:
        # this is going to be a singlular test
        meshOBJ = initialize(name=name, num_vars=num_vars, nx=nx, gamma=gamma, sigmoid_value=sigmoid_value)
        Balbas.balbas_one_dimension(meshOBJ, alpha=alpha, Tmax=Tmax, num_vars=num_vars, Bx=Bx,
                                    gamma=gamma, nx=nx, n_plots=n_plots, CFL_safety=CFL_safety,
                                    length=length, name=name, alpha_test=False, start_time=formatted_time)

### MODIFY HERE ###
input_dict_base = {'name':'bruh', 'alpha':1.4, 'test':True, 'Tmax':2., 'num_vars':7, 'Bx':0.75,
     'gamma':2, 'nx':2000, 'n_plots':100, 'CFL_safety':80.,
     'length':20, 'alpha_test':False, 'sigmoid_value':0}

### TAKES INPUT_DICT_BASE AND MODIFIES ONLY THE VARIABLES SPECIFIED ###
input_dict_sigmoid = {**input_dict_base, 'name':'sigmoid'}
# input_dict_sigmoid_test = {**input_dict_base, 'name':'sigmoid', 'sigmoid_value':5, 'test':True}
# run(**input_dict_sigmoid) # sigmoid case

# input_dict_alpha_test = {**input_dict_base, 'name':'Brio&Wu', 'alpha_test':True, 'test':True} # test=true runs one rather than a collection of tests
input_dict_alpha = {**input_dict_base, 'name':'Brio&Wu','alpha_test':True, 'alpha':1}
# run(**input_dict_alpha)

input_single_other = {**input_dict_base, 'name':'alfven2', 'test':True, 'Bx':1.0}
run(**input_single_other)

#################################### getting data after runs ####################################
# %%
# import os
# import numpy as np
# from datavis import visualize_mhd_data


# B_y = {};B_z = {}
# rho = {};en = {}
# u_x = {};u_y = {};u_z = {}
# # directory = 'sigmoid_test_bruh'# YOU WILL NEED TO CHANGE THIS BASED ON WHAT YOU NEED
# directory = 'alpha_test_Brio&Wu_14:57:23'
# # SEE LINE 103 AND 107 FOR FORMAT OF THE NAMES IF YOU WANT TO FIND A CLEVER WAY TO DO IT
# t_vals =[];var_vals = []
# for file in os.listdir(directory):
#     full_path = os.path.join(directory, file)
#     split_name = file.split('_')
#     if split_name[0] == 'sigmoid':
#         case = split_name[0]; var = split_name[1]; t = split_name[3]  # val will be either alpha or sigma depending on what directory is commented
#     else:
#         case = split_name[1]; var = split_name[3]; t = split_name[5]

#     t_vals.append(t); var_vals.append(var)
#     data = np.load(full_path)

#     B_y[f'{var}_{t}'] = data[f'B_y_{var}_{t}']
#     B_z[f'{var}_{t}'] = data[f'B_z_{var}_{t}']
#     u_x[f'{var}_{t}'] = data[f'u_x_{var}_{t}']
#     u_y[f'{var}_{t}'] = data[f'u_y_{var}_{t}']
#     u_z[f'{var}_{t}'] = data[f'u_z_{var}_{t}']
#     en[f'{var}_{t}'] = data[f'en_{var}_{t}']
#     rho[f'{var}_{t}'] = data[f'rho_{var}_{t}']


# t_vals = np.unique(t_vals); var_vals = np.unique(var_vals)
# for t_val in t_vals:
#     for count, (key, value) in enumerate(B_y.items()):
#         var, t = key.split('_')
#         #### time plotting ####
#         print(t)
#         print(var)
#         if t == t_val:
#             plt.plot(value, label=f'{var}', alpha=.3, lw=2)
#     plt.title(f'Time: {t_val}')
#     plt.xlabel('Position')
#     plt.ylabel('B_y') # NEEDTO CHANGE THIS TO WHATEVER YOU ARE CALLING ASBFKJASDBFKJASDBFKASBFKBAF
#     plt.legend()
#     plt.show()




import numpy as np
import math as m

from scipy import constants
import matplotlib.pyplot as plt
import os
import pandas as pd

from Library import minmod, calc_f, calc_f_cell, animate
from datetime import datetime
from tqdm import tqdm

########## Variable Indices ###########
# 0: rho 
# 1: rho u_x 
# 2: rho u_y 
# 3: rho u_z 
# 4: By 
# 5: Bz 
# 6: energy 

########## Initial Parameters ##########
# get the plotting parameters


# construct initial grid

def balbas_one_dimension(meshOBJ, alpha, Tmax, num_vars, Bx, gamma, nx, n_plots, CFL_safety, length, name, alpha_test=False, sig=0):

    dx = length / nx
    dt = dx / CFL_safety
    nt = int(Tmax / dt)
    lam = dt / dx

    now = datetime.now()

    t = 0; tL = 0

    # initialize stagger_switch
    stagger_switch = 0

    meshOBJ[1,:,:] = meshOBJ[0,:,:].copy()

    # to force the (data)typing with numpy to work
    update_arr = np.empty(num_vars)
    wjnhalf_arr = np.empty_like(update_arr)
    wjp1nhalf_arr = np.empty_like(update_arr)

    while t <= Tmax:

        # iterate over entire array storing each variable component on every grid node: calculates and returns the derivative values at each cell fpr all vars
        f_vals = calc_f(meshOBJ[stagger_switch, :, :], Bx, gamma)

        # loop over all cells, with a margin of 2 inside the boundaries
        # leaving these two cells untouched should act as ghost cells, ensuring our boundaries stay "pinned" to the initial values (i.e., dirichlet condition)
        for i in range(2,nx-2):
            # loop calculations over each quantity in the supervector [rho, rho u_x, rho u_y, rho u_z, By, Bz, energy] since we vectorized the derivative funcs
            for equation_idx in range(num_vars):
                # get 1D array of each scalar quantity (or component, i.e., By individually) on the domain
                w = meshOBJ[stagger_switch, equation_idx, :]
                # grab the argument of the divergence term in each MHD eqn by slicing the array returned from calc_f() (i.e., equation 4.2 in Balbas)
                f_ = f_vals[equation_idx,:]
                # use minmod to calculate spatial derivatives, following equation depending on whether offset mesh
                wp = minmod(alpha,w[i-1],w[i],w[i+1])
                wpnp1 = minmod(alpha, w[i-(2*stagger_switch)], w[i-(2*stagger_switch)+1], w[i-(2*stagger_switch)+2])
                # equation 2.19 in Balbas: i.e., the "predictor" step of the leapfrog (still switching between staggered mesh)
                wjnhalf_arr[equation_idx] = w[i] - (lam/2.) * minmod(alpha, f_[i-1], f_[i], f_[i+1])
                wjp1nhalf_arr[equation_idx] = w[i-(2*stagger_switch) + 1] - (lam/2.) * minmod(alpha, f_[i-(2*stagger_switch)], f_[i-(2*stagger_switch)+1], f_[i-(2*stagger_switch)+2])
                # equation 2.17 (RHS) in Balbas: use the staggered cells to integrate exactly, in terms of our polynomial reconstruction
                update_arr[equation_idx] = (1./2.) * (w[i] + w[i-(2*stagger_switch)+1]) + (-1 * stagger_switch)*(wp - wpnp1)/8.

            # handle the temporal integral now (equation written on bottom of pg 266)
            # evaluate the derivatives of the quantities along the midcells using the predictor step, wjnhalf & wjp1nhalf
            fjnhalf = calc_f_cell(wjnhalf_arr, Bx, gamma)
            fjnp1half = calc_f_cell(wjp1nhalf_arr, Bx, gamma)

            # handle the staggered mesh by using the plus/minus
            if stagger_switch == 0:
                meshOBJ[1,:,i] = update_arr + lam*(fjnp1half - fjnhalf)
            elif stagger_switch == 1:
                meshOBJ[0,:,i] = update_arr - lam*(fjnp1half - fjnhalf)
            else:
                raise ValueError('The variable "stagger_switch" must be either zero or unity!')

        # benchmarking output
        # should largely incorporate the animation and plotting routines we have already made
        if tL >= 0:
            if (tL % (nt // n_plots)) == 0:
                percent = round((t / Tmax)*100.,2)
                print('time t=',t)
                print(r'progress is ', percent, r'% done')
                # output data
                # might need to call at first index: 1 - stagger_switch*1 to get the values that we JUST UPDATED IN THIS TIMESTEP,
                # i.e., the values on the offset mesh (since we switch up and down between the j+1/2 and j each dt)

                rho_arr, u_x_arr, u_y_arr, u_z_arr, B_y_arr, B_z_arr, en_arr = animate(meshOBJ, stagger_switch, tL, now, alpha=alpha, Tmax=Tmax, nx=nx, dt=dt, dx=dx, num_vars=num_vars, name=name, sig=sig)
                if alpha_test: # this is all to save the data so we can do the plotting easier later on
                    data = {f'rho_{alpha}_{t}': rho_arr, f'u_x_{alpha}_{t}': u_x_arr, f'u_y_{alpha}_{t}': u_y_arr, f'u_z_{alpha}_{t}': u_z_arr,
                            f'B_y_{alpha}_{t}': B_y_arr, f'B_z_{alpha}_{t}': B_z_arr, f'en_{alpha}_{t}': en_arr}
                    # df = pd.DataFrame(data)
                    # df = df.astype(float)
                    # df.to_csv(f'alpha_test_dir_case_{name}/case_{name}_alpha_{alpha}_time_{t}_timestep_{tL}.csv')
                    np.savez(f'alpha_test_dir_case_{name}/case_{name}_alpha_{alpha}_time_{t}_timestep_{tL}.npz', **data)
                elif sig != 0:
                    data = {f'rho_{sig}_{t}': rho_arr, f'u_x_{sig}_{t}': u_x_arr, f'u_y_{sig}_{t}': u_y_arr, f'u_z_{sig}_{t}': u_z_arr,
                            f'B_y_{sig}_{t}': B_y_arr, f'B_z_{sig}_{t}': B_z_arr, f'en_{sig}_{t}': en_arr}
                    np.savez(f'sigmoid_test/sigmoid_{sig}_time_{t}_timestep_{tL}.npz', **data)
                # also calculate CFL number
                if tL != 0:
                    CFL = meshOBJ[stagger_switch, 1, :].max() * dt / dx
                    print('CFL is',CFL)
                    # might be good to put a CFL=0 check to break the code ie
                    # if CFL == 0 or CFL == np.nan: break
                    # might not do well if we are attempting to graph all together for all time values


        if stagger_switch == 1:
            stagger_switch = 0
        elif stagger_switch == 0:
            stagger_switch = 1
        else:
            raise ValueError('The variable "stagger_switch" must be either zero or unity!')

        # advance time
        # goofy ahh 2 AM eigenvalue calculation
        # this might depend on what layer we are using
        # need to reconstruct pressure from
        # B_square = np.dot( np.dot(meshOBJ[stagger_switch, 4, :], meshOBJ[stagger_switch, 5, :]),  np.ones_like(meshOBJ[stagger_switch, 5, :]) * Bx)
        # pressure = (gamma - 1) * (meshOBJ[stagger_switch, 6, :] - np.dot( np.dot(meshOBJ[stagger_switch, 1, :], meshOBJ[stagger_switch, 2, :]),  meshOBJ[stagger_switch, 3, :]) / 2.
        #                     - B_square / 2)
        # a_square = gamma * np.divide(pressure , meshOBJ[stagger_switch, 0, :])
        # c_ax = Bx / np.sqrt(meshOBJ[stagger_switch, 0, :])
        # c_A_squared = np.divide(B_square, meshOBJ[stagger_switch, 0, :])
        # c_f = .5 * ( a_square + c_A_squared + np.sqrt( np.power(a_square + c_A_squared, 2) - 4 * a_square * np.power(c_ax,2) ) )
        # wave_speed = np.max(np.max(c_f), np.max(c_ax) )  # maybe only need one max around. too tired
        # v_max = meshOBJ[stagger_switch, 1, :].max() + wave_speed
        # dt = new_dt =  dx / v_max / CFL_safety if v_max != 0 else 1e-4 # if you wanted to do an adaptive time step with only the velocity
        t += dt
        tL += 1


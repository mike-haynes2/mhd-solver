import numpy as np # type: ignore
import math as m 

from scipy import constants # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os

from Library import minmod, calc_f, calc_f_cell, animate
import configuration as c
from datetime import datetime
from tqdm import tqdm # type: ignore

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
now =  datetime.now()

# initialize stagger_switch
stagger_switch = 0

# construct initial grid
meshOBJ = c.initialize(c.name)
meshOBJ[1,:,:] = meshOBJ[0,:,:].copy()

# to force the (data)typing with numpy to work
update_arr = np.empty(c.num_vars)
wjnhalf_arr = np.empty_like(update_arr)
wjp1nhalf_arr = np.empty_like(update_arr)

while c.t <= c.Tmax:

    # iterate over entire array storing each variable component on every grid node: calculates and returns the derivative values at each cell fpr all vars
    f_vals = calc_f(meshOBJ[stagger_switch, :, :], c.Bx, c.gamma)
    
    # loop over all cells, with a margin of 2 inside the boundaries
    # leaving these two cells untouched should act as ghost cells, ensuring our boundaries stay "pinned" to the initial values (i.e., dirichlet condition)
    for i in range(2,c.nx-2):
        # loop calculations over each quantity in the supervector [rho, rho u_x, rho u_y, rho u_z, By, Bz, energy] since we vectorized the derivative funcs 
        for equation_idx in range(c.num_vars):
            # get 1D array of each scalar quantity (or component, i.e., By individually) on the domain
            w = meshOBJ[stagger_switch, equation_idx, :]
            # grab the argument of the divergence term in each MHD eqn by slicing the array returned from calc_f() (i.e., equation 4.2 in Balbas)
            f_ = f_vals[equation_idx,:]
            # use minmod to calculate spatial derivatives, following equation depending on whether offset mesh 
            wp = minmod(c.alpha,w[i-1],w[i],w[i+1])
            wpnp1 = minmod(c.alpha, w[i-(2*stagger_switch)], w[i-(2*stagger_switch)+1], w[i-(2*stagger_switch)+2])
            # equation 2.19 in Balbas: i.e., the "predictor" step of the leapfrog (still switching between staggered mesh)
            wjnhalf_arr[equation_idx] = w[i] - (c.lam/2.) * minmod(c.alpha, f_[i-1], f_[i], f_[i+1])
            wjp1nhalf_arr[equation_idx] = w[i-(2*stagger_switch) + 1] - (c.lam/2.) * minmod(c.alpha, f_[i-(2*stagger_switch)], f_[i-(2*stagger_switch)+1], f_[i-(2*stagger_switch)+2])
            # equation 2.17 (RHS) in Balbas: use the staggered cells to integrate exactly, in terms of our polynomial reconstruction
            update_arr[equation_idx] = (1./2.) * (w[i] + w[i-(2*stagger_switch)+1]) + (-1 * stagger_switch)*(wp - wpnp1)/8.

        # handle the temporal integral now (equation written on bottom of pg 266) 
        # evaluate the derivatives of the quantities along the midcells using the predictor step, wjnhalf & wjp1nhalf
        fjnhalf = calc_f_cell(wjnhalf_arr, c.Bx, c.gamma)
        fjnp1half = calc_f_cell(wjp1nhalf_arr, c.Bx, c.gamma)

        # handle the staggered mesh by using the plus/minus 
        if stagger_switch == 0:
            meshOBJ[1,:,i] = update_arr + c.lam*(fjnp1half - fjnhalf)
        elif stagger_switch == 1:
            meshOBJ[0,:,i] = update_arr - c.lam*(fjnp1half - fjnhalf)
        else:
            raise ValueError('The variable "stagger_switch" must be either zero or unity!')

    # benchmarking output 
    # should largely incorporate the animation and plotting routines we have already made
    if c.tL == 0: 
        percent = round((c.t / c.Tmax)*100.,2)
        print('time t=',c.t)
        print(r'progress is ', percent, r'% done')
        # output data
        # might need to call at first index: 1 - stagger_switch*1 to get the values that we JUST UPDATED IN THIS TIMESTEP,
        # i.e., the values on the offset mesh (since we switch up and down between the j+1/2 and j each dt)

        animate(meshOBJ, stagger_switch, c.tL, now)  
    elif c.tL > 0:
        if (c.tL % (c.nt // c.n_plots)) == 0:
            percent = round((c.t / c.Tmax)*100.,2)
            print('time t=',c.t)
            print(r'progress is ', percent, r'% done')
            # output data
            # might need to call at first index: 1 - stagger_switch*1 to get the values that we JUST UPDATED IN THIS TIMESTEP,
            # i.e., the values on the offset mesh (since we switch up and down between the j+1/2 and j each dt)

            animate(meshOBJ, stagger_switch, c.tL, now)

            # also calculate CFL number
            CFL = meshOBJ[stagger_switch, 1, :].max() * c.dt / c.dx
            print('CFL is',CFL)

    if stagger_switch == 1:
        stagger_switch = 0
    elif stagger_switch == 0:
        stagger_switch = 1
    else:
        raise ValueError('The variable "stagger_switch" must be either zero or unity!')

    # advance time
    c.t += c.dt
    c.tL += 1

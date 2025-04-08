import numpy as np
import math as m

from scipy import constants
import matplotlib.pyplot as plt
import os

from Library import minmod, calc_f, calc_f_cell

######### Variable indices ###########
# 0: rho #
# 1: rho u_x #
# 2: rho u_y #
# 3: rho u_z #
# 4: By #
# 5: Bz #
# 6: energy #
### keep track as we loop over equations ###






# initialize parameters (in this 1D model, Bx is a mere parameter)
## Constants
# currently haven't worried about reconstruction of dimensionality in results. Would be necessary for that
mu0 = constants.mu_0
c = constants.c

## Physical parameters
# derivative factor in minmod, still need to implement
alpha = 1.4
# constant, uniform x component of B field
Bx = 0.75
# adiabatic index
gamma = 2

## Spatial domain
# initialize number of gridpoints
nx = 200
# spatial extent:
length = 1. # m
# spatial step:
dx = length / nx

## time domain
# set dt according to rough courant (CFL) estimate. Velocities in their runs are order 1
dt = dx / 100.
print('dt=',dt, ' s')
# max time
Tmax = 0.15 # s
# number of timesteps
nt = int(Tmax / dt)
# lambda from balbas
lam = dt / dx

## for vectorized loop over all quantities
num_vars = 7




## define objects to store everything / initialize
meshOBJ = np.empty((2, num_vars, nx))

stagger_switch = 0

## initialize the shock tube problem
e0 = (1. / (gamma-1.)) - (1./2.)

meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0]]).T
meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0/10.]]).T

# init time
t = 0.
# timestep
tL = 0



### ^^^^ ##### all above here should be in a configuration file (for initialization)
# would be nice to include the match case thing with the other shock configurations (and we can futher validate the code with the hydrodynamic test case)




# initialize staggered 'offset' mesh
meshOBJ[1,:,:] = meshOBJ[0,:,:].copy()


# to force the (data)typing with numpy to work
# and to avoid the asserts y'all hate lol
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

        # PROPERLY handle the staggered mesh by using the plus/minus 
        # SWITCHING BACK AND FORTH
        # why didn't this click before ? I (CMH) am illiterate 
        if stagger_switch == 0:
            meshOBJ[1,:,i] = update_arr + lam*(fjnp1half - fjnhalf)
        elif stagger_switch == 1:
            meshOBJ[0,:,i] = update_arr - lam*(fjnp1half - fjnhalf)
        else:
            raise ValueError('The variable "stagger_switch" must be either zero or unity!')
        

    # benchmarking output 
    # should largely incorporate the animation and plotting routines we have already made
    if tL > 0:
        if (tL % (nt // 10)) == 0:
            print('time t=',t)
            print(r'progress is ',round((t / Tmax)*100.,2),r'% done')
            # output data
            # might need to call at first index: 1 - stagger_switch*1 to get the values that we JUST UPDATED IN THIS TIMESTEP,
            # i.e., the values on the offset mesh (since we switch up and down between the j+1/2 and j each dt)
            rho_arr = meshOBJ[stagger_switch, 0, :]
            u_x_arr = meshOBJ[stagger_switch, 1, :] / rho_arr
            u_y_arr = meshOBJ[stagger_switch, 2, :] / rho_arr
            u_z_arr = meshOBJ[stagger_switch, 3, :] / rho_arr
            B_y_arr = meshOBJ[stagger_switch, 4, :] 
            B_z_arr = meshOBJ[stagger_switch, 5, :]
            en__arr = meshOBJ[stagger_switch, 6, :] 

            plt.plot(rho_arr)
            plt.show()
            plt.plot(u_y_arr)
            plt.show()
            plt.plot(B_y_arr)
            plt.show()



    if stagger_switch == 1:
        stagger_switch = 0
    elif stagger_switch == 0:
        stagger_switch = 1
    else:
        raise ValueError('The variable "stagger_switch" must be either zero or unity!')


    # advance time
    t += dt
    tL += 1

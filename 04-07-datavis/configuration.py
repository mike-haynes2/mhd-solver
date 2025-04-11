import numpy as np # type: ignore
import math as m 

from scipy import constants  # type: ignore
import matplotlib.pyplot as plt # type: ignore

# would be nice to include the match case thing with the other shock configurations (and we can futher validate the code with the hydrodynamic test case)


# initialize parameters (in this 1D model, Bx is a mere parameter)
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
CFL_safety = 40.
dt = dx / CFL_safety
print('dt=',dt, ' s')
# max time
Tmax = 0.2 # s
# number of timesteps
nt = int(Tmax / dt)
# lambda from balbas
lam = dt / dx

n_plots = 20

# for vectorized loop over all quantities
num_vars = 7

# init time and time level
t = 0.
tL = 0

## define objects to store everything / initialize
meshOBJ = np.empty((2, num_vars, nx))

# problem to solve
name = 'Brio & Wu'

def initialize(name):
    """returns the initial conditions based on the type of problem specified by the 'name' variable"""

    # create object to store most recent iterations for all variables
    meshOBJ = np.empty((2, num_vars, nx))

    # choose the appropriate initial conditions based on the problem type
    match name:
        case 'Dai & Woodward':
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

        case 'Brio & Wu':
            
            e0 = (1. / (gamma-1.)) - (1./2.)

            meshOBJ[0, :, 0:(nx // 2)] = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, e0]]).T
            meshOBJ[0, :, (nx // 2):nx] = np.array([[0.125, 0.0, 0.0, 0.0, -1.0, 0.0, e0/10.]]).T

            return meshOBJ

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
        case _:
            raise ValueError('invalid testing: use ""Dai & Woodward"" or  ""Brio & Wu"" or ""slow shock"" or ""rarefaction""')



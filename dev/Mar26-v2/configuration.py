import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bokeh.models.widgets import inputs
from scipy import constants

def initialize_1D(name='', nx=400, xmin=-1., xmax=1.):
    Xs = np.linspace(start=xmin, stop=xmax, num=nx)
    dx = (xmax-xmin) / (nx - 2) # minus 2 here for the two ghost cells
    left_mask = (Xs < 0.0); right_mask = ~left_mask
    num_neg = left_mask.sum()

    base = np.ones(nx)

    match name:
        case 'Dai & Woodward':
            ro0_neg = 1.08; pr0_neg = 0.95
            ux0 = 1.2; uy0 = 0.01; uz0 = 0.5
            bx0 = 2. / np.sqrt(16 * np.arctan(1.)); by0 = 3.6/ np.sqrt(16 * np.arctan(1.)); bz0 = 2. / np.sqrt(16 * np.arctan(1.))
            b_vec_neg = np.array([bx0, by0, bz0])
            u_vec_neg = np.array([ux0, uy0, uz0])

            ro0_pos = 1.; pr0_pos = 1
            ux0 = 0.; uy0 = 0.; uz0 = 0.
            bx0 = 2. / np.sqrt(16 * np.arctan(1.)); by0 = 4. / np.sqrt(16 * np.arctan(1.)); bz0 = 2. / np.sqrt(16 * np.arctan(1.))
            b_vec_pos = np.array([bx0, by0, bz0])
            u_vec_pos = np.array([ux0, uy0, uz0])

        case 'Brio & Wu':
            ro0_neg = 1.; pr0_neg = 1.
            ux0 = 0.; uy0 = 0.; uz0 = 0.
            bx0 = .75; by0 = 1.; bz0 = 0.
            b_vec_neg = np.array([bx0, by0, bz0])
            u_vec_neg = np.array([ux0, uy0, uz0])

            ro0_pos = .125; pr0_pos = .1
            ux0 = 0.; uy0 = 0.; uz0 = 0.
            bx0 = .75; by0 = -1.; bz0 = 0
            b_vec_pos = np.array([bx0, by0, bz0])
            u_vec_pos = np.array([ux0, uy0, uz0])
        case 'slow shock':
            ro0_neg = 1.368; pr0_neg = 1.769
            ux0 = 0.269; uy0 = 1.0; uz0 = 0.
            bx0 = 1.; by0 = 0.; bz0 = 0.
            b_vec_neg = np.array([bx0, by0, bz0])
            u_vec_neg = np.array([ux0, uy0, uz0])

            ro0_pos = 1.; pr0_pos = 1.
            ux0 = 0.; uy0 = 0.; uz0 = 0.
            bx0 = 1.; by0 = 1.; bz0 = 0
            b_vec_pos = np.array([bx0, by0, bz0])
            u_vec_pos = np.array([ux0, uy0, uz0])
        case 'rarefaction':
            ro0_neg = 1.; pr0_neg = 2.
            ux0 = 0.; uy0 = 0.; uz0 = 0.
            bx0 = 1.; by0 = 0.; bz0 = 0.
            b_vec_neg = np.array([bx0, by0, bz0])
            u_vec_neg = np.array([ux0, uy0, uz0])

            ro0_pos = .2; pr0_pos = 0.1368
            ux0 = 1.186; uy0 = 2.967; uz0 = 0.
            bx0 = 1.; by0 = 1.6405; bz0 = 0
            b_vec_pos = np.array([bx0, by0, bz0])
            u_vec_pos = np.array([ux0, uy0, uz0])
        case _:
            raise ValueError('invalid testing: use ""Dai & Woodward"" or  ""Brio & Wu"" or ""slow shock"" or ""rarefaction""')
            return np.zeros(nx), np.zeros((nx, 3)), np.zeros((nx, 3)), np.zeros(nx), np.zeros(nx), dx, Xs

    rho0 = base.copy()
    u0 = np.concatenate((np.outer(base[:num_neg], u_vec_neg), np.outer(base[num_neg:], u_vec_pos)))
    B0 = np.concatenate((np.outer(base[:num_neg], b_vec_neg), np.outer(base[num_neg:], b_vec_pos)))
    rho0[left_mask] = ro0_neg; rho0[right_mask] = ro0_pos


    B_square_neg = np.dot(b_vec_neg, b_vec_neg) / mu0 / 2 # this might be wrong
    B_square_pos = np.dot(b_vec_pos, b_vec_pos) / mu0 / 2
    u_square_neg = np.dot(u_vec_neg, u_vec_neg)
    u_square_pos = np.dot(u_vec_pos, u_vec_pos)
    p0 = np.concatenate((base[:num_neg] * pr0_neg + B_square_neg, base[num_neg:] * pr0_pos + B_square_pos))
    energy0 = p0 / (adiabatic_index - 1) + np.concatenate(( base[:num_neg] * B_square_neg , base[:num_neg] * B_square_pos)) \
            + np.concatenate(( rho0[:num_neg] * u_square_neg / 2, rho0[num_neg:] * u_square_pos / 2 ))

    return rho0, u0, B0, energy0, p0, dx, Xs

# from reconstruct import reconstruct_pressure

# ALL parameters, state equations, and other system / solver configuration info 


## phyiscal constants
################################################################################
################################################################################
# imported constants
mp = constants.m_p
me = constants.m_e
c = constants.c
mu0 = constants.mu_0
eps0 = constants.epsilon_0
################################################################################
################################################################################

## physical parameters
################################################################################
################################################################################
# thermal parameters 
adiabatic_index = 2.     # gamma (1 + 2 / DOF)
# (FOR pickup process:)
# distribution properties for the injected population

## numerical parameters
################################################################################
################################################################################

# spatial extent (1D case)

# time evolution (max time)

################################################################################
################################################################################

## Initial conditions
################################################################################
################################################################################
# initialize spatial grid
X_lower = -1.; X_upper = 1.;nx = 800
rho0, u0, B0, energy0, p0, dx, Xs = initialize_1D(name='Brio & Wu', nx=nx, xmin=X_lower, xmax=X_upper) # might have to change the name of some things
CFL_velocity = 1.e+03 # 1 km / s (i.e., 'max' velocity we would expect)
CFL_safety_factor = .4  # how close to CFL condition is our timestep (4.4 Balbas)
dt = (dx * CFL_safety_factor)/CFL_velocity
N_Tmax = 100
# in seconds:
Tmax = N_Tmax*dt # as a note the paper uses a hard cutoff of .2 or for high mach number is 0.012

alpha = 1.4 # paper uses 1.4 in caption of figure 4 # weighting threshold for MinMod derivative approximator

left_mask = (Xs < 0.0)
right_mask = ~left_mask
num_neg = left_mask.sum()
############################
# shock tube test case
############################

# placeholder for each quantity (u, B, p, etc)
w_t0 = np.ones_like(Xs)
# define boundary conditions to be enforced:
# CONSTANT boundary values:
# value at lower endpoint
w_t0_x0 = 0.
# value at upper endpoint
w_t0_xM = 0.
# constant value for flow-aligned magnetic field B:
B_x_perscription = 3./4.    # nT    (see Equation 4.4 in Balbas et al)
# initial value for density


## MHD System
################################################################################
################################################################################
# 1-D equations (see eqns 4.1-4.2 in Balbas et al)
# continuity:
def f_continuity_1D(rho, u_vector):
    u_x = np.array(u_vector)[0]
    return (rho*u_x)
# EOM / Navier Stokes:
def f_NS_1D(u_vector, rho, B_vector, p):
    pstar = p + (np.dot(B_vector,B_vector)/mu0)
    xhat = np.array([1.,0.,0.])
    u_x = np.dot(np.array(u_vector),xhat)
    B_x = np.dot(np.array(B_vector),xhat)
    return (rho * u_x * u_vector - B_x * B_vector + pstar * xhat)
# Faraday's Law for MHD (dB/dt = - curl E = curl(u x B) = div(B tensor u - u tensor B))
def f_faraday_1D(B_vector, u_vector):
    By = u_vector[0] * B_vector[1] - B_vector[0] * u_vector[1]
    Bz = u_vector[0] * B_vector[2] - B_vector[0] * u_vector[2]
    return np.array([0.,By,Bz])
# energy equation
def f_energy_1D(Energy, u_vector, B_vector, p):
    pstar = p + (np.dot(B_vector,B_vector)/mu0)
    f1 = (Energy + pstar) * u_vector[0]
    f2 = B_vector[0] * np.dot(u_vector,B_vector)
    return (f1 - f2)
############### If we wanted to use only one time integration the files###############
# def f_continuity_1D(rho, inputs):
#     u_vector = inputs[1]
#     u_x = np.array(u_vector)[0]
#     return (rho*u_x)
# # EOM / Navier Stokes:
# def f_NS_1D(u_vector, inputs):
#     rho = inputs[0]; B_vector = inputs[1]; p = inputs[2]
#     pstar = p + (np.dot(B_vector,B_vector)/mu0)
#     xhat = np.array([1.,0.,0.])
#     u_x = np.dot(np.array(u_vector),xhat)
#     B_x = np.dot(np.array(B_vector),xhat)
#     return (rho * u_x * u_vector - B_x * B_vector + pstar * xhat)
# # Faraday's Law for MHD (dB/dt = - curl E = curl(u x B) = div(B tensor u - u tensor B))
# def f_faraday_1D(B_vector, inputs):
#     u_vector = inputs[0]
#     By = u_vector[0] * B_vector[1] - B_vector[0] * u_vector[1]
#     Bz = u_vector[0] * B_vector[2] - B_vector[0] * u_vector[2]
#     return np.array([0.,By,Bz])
# # energy equation
# def f_energy_1D(Energy, inputs):
#     u_vector = inputs[0]; B_vector=inputs[1]; p=inputs[2]
#     pstar = p + (np.dot(B_vector,B_vector)/mu0) # missing 1/2?
#     f1 = (Energy + pstar) * u_vector[0]
#     f2 = B_vector[0] * np.dot(u_vector,B_vector)
#     return (f1 - f2)


# adiabatic law
def calculate_p_adiabatic(rho,gamma,reference_rho=1.,reference_p=1.):
    c = (rho/reference_rho) ** gamma
    return (reference_p * c)

def animation(time_data, name):
    fig, ax = plt.subplots()
    line, = ax.plot(time_data[0], alpha=0.7)
    ax.set_ylim(np.max(time_data) - 1, np.max(time_data) + 1)  # if it crashes on this line just comment it out
    ax.set_xlabel("Position")
    ax.set_ylabel(name)

    def update(frame):
        line.set_ydata(time_data[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=np.size(time_data, 0), interval=100, blit=True)
    ani.save(f"{name}_animation.mp4", writer="ffmpeg")
    plt.show()


################################################################################
################################################################################

## define (untouched) quantities for solver based on above definitions:
################################################################################
################################################################################
all_quantities_t0_with_adiabatic = [rho0, u0, B0, energy0, p0]
all_functions_with_adiabatic = [f_continuity_1D, f_NS_1D, f_faraday_1D, f_energy_1D, calculate_p_adiabatic]

all_quantities_t0 = all_quantities_t0_with_adiabatic[0:4]
all_functions = all_functions_with_adiabatic[0:4]
################################################################################
################################################################################

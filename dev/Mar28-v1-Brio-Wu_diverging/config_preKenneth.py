import os
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from bokeh.models.widgets import inputs
from scipy import constants

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
# weighting threshold for MinMod derivative approximator
alpha = 1. # paper uses 1.4 in caption of figure 4

# spatial / time increment
# (CFL parameters)
CFL_velocity = 10 # 1 km / s (i.e., 'max' velocity we would expect)
CFL_safety_factor = .4  # how close to CFL condition is our timestep (4.4 Balbas)
# perscribe spatial step
dx = 1.e-02              # 1 cm grid spacing
# calculate stable timestep dt
dt = (dx * CFL_safety_factor)/CFL_velocity
print('timestep is: '+str(dt))
# spatial extent (1D case)
X_lower = -1.
X_upper = 1.
# time evolution (max time, s)
Tmax = 0.002 # as a note the paper uses a hard cutoff of .2 or for high mach number is 0.012
# in timesteps
N_Tmax = Tmax / dt

################################################################################
################################################################################



## Initial conditions
################################################################################
################################################################################
# initialize spatial grid
Xs = np.arange(start=X_lower, stop=X_upper,step=dx)
set_num_X = False
if set_num_X:# If we want to set the number of spatial steps instead of dx
    nx = 400
    Xs = np.linspace(start=X_lower,stop=X_upper,num=nx)



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

rho_const_negative = 1.
rho_const_positive = 0.125

u_vector_const = np.zeros(3)

By_const_negative = 1.
Bz_const_negative = 0.
B_vector_const_negative = np.array([B_x_perscription,By_const_negative, Bz_const_negative])
By_const_positive = -1.
Bz_const_positive = 0
B_vector_const_positive = np.array([B_x_perscription,By_const_positive, Bz_const_positive])

rho0 = w_t0.copy()
rho0[left_mask] = rho_const_negative; rho0[right_mask] = rho_const_positive

u0 = np.outer(w_t0,u_vector_const)
B0 = np.concatenate((np.outer(w_t0[:num_neg],B_vector_const_negative), np.outer(w_t0[num_neg:],B_vector_const_positive)))

#p0 = reconstruct_pressure(rho0,u0,B0,energy0)
# we are given initial pressure term, do we need to reconstruct it? (eq. 4.4)
p0_negative = 1.; p0_positive = 0.1 # for high mach number only difference is p0_negative is 1000.
B_square = np.dot(B_vector_const_negative, B_vector_const_negative)
p0 = np.concatenate((w_t0[:num_neg] * p0_negative, w_t0[num_neg:] * p0_positive)) + .5 * B_square / mu0

energy0 = p0 / (adiabatic_index - 1) + .5 * B_square / mu0 # + .5 * rho * u but it is zero so does not matter


## MHD System
################################################################################
################################################################################
# 1-D equations (see eqns 4.1-4.2 in Balbas et al)
# continuity:
# def f_continuity_1D(rho, u_vector):
#     u_x = np.array(u_vector)[0]
#     return (rho*u_x)
# # EOM / Navier Stokes:
# def f_NS_1D(u_vector, rho, B_vector, p):
#     pstar = p + (np.dot(B_vector,B_vector)/mu0)
#     xhat = np.array([1.,0.,0.])
#     u_x = np.dot(np.array(u_vector),xhat)
#     B_x = np.dot(np.array(B_vector),xhat)
#     return (rho * u_x * u_vector - B_x * B_vector + pstar * xhat)
# # Faraday's Law for MHD (dB/dt = - curl E = curl(u x B) = div(B tensor u - u tensor B))
# def f_faraday_1D(B_vector, u_vector):
#     By = u_vector[0] * B_vector[1] - B_vector[0] * u_vector[1]
#     Bz = u_vector[0] * B_vector[2] - B_vector[0] * u_vector[2]
#     return np.array([0.,By,Bz])
# # energy equation
# def f_energy_1D(Energy, u_vector, B_vector, p):
#     pstar = p + (np.dot(B_vector,B_vector)/(2.*mu0))
#     f1 = (Energy + pstar) * u_vector[0]
#     f2 = B_vector[0] * np.dot(u_vector,B_vector)
#     return (f1 - f2)

############### If we wanted to use only one time integration the files###############
def f_continuity_1D(rho, inputs):
    u_vector = inputs[0]
    u_x = np.array(u_vector)[:,0]
    return (rho*u_x)
# EOM / Navier Stokes:
def f_NS_1D(u_vector, inputs):
    rho = inputs[0]; B_vector = inputs[1]; p = inputs[2]
    pstar = p + ((np.linalg.norm(B_vector,axis=1) ** 2.)/(2.*mu0)) 
    xhat = np.array([1.,0.,0.])
    # u_x = np.dot(np.array(u_vector),xhat)
    # B_x = np.dot(np.array(B_vector),xhat)
    # print(np.shape(u_vector))
    u_x = u_vector[:,0]
    B_x = B_vector[:,0]
    return (np.multiply(np.multiply(u_vector,u_x[:,np.newaxis]),rho[:,np.newaxis])- np.multiply(np.multiply(B_vector,B_x[:,np.newaxis]), pstar[:,np.newaxis]) )
# Faraday's Law for MHD (dB/dt = - curl E = curl(u x B) = div(B tensor u - u tensor B))
def f_faraday_1D(B_vector, inputs):
    B = np.zeros_like(B_vector)
    for i, bvec in enumerate(B_vector):
        # print(bvec)
        u_vector = inputs[0][i]
        B[i,1] = u_vector[0] * bvec[1] - bvec[0] * u_vector[1]
        B[i,2] = u_vector[0] * bvec[2] - bvec[0] * u_vector[2]
    return B
# energy equation
def f_energy_1D(Energy, inputs):
    u_vector = inputs[0]; B_vector=inputs[1]; p=inputs[2]
    pstar = p + ((np.linalg.norm(B_vector,axis=1) ** 2.)/(2.*mu0)) 
    f1 = (Energy + pstar) * u_vector[:,0]
    f2 = B_vector[:,0] * np.einsum('ij,ij->i', u_vector, B_vector)
    return (f1 - f2)

# adiabatic law
def calculate_p_adiabatic(rho,gamma,reference_rho=1.,reference_p=1.):
    c = (rho/reference_rho) ** gamma
    return (reference_p * c)

### Movie Animation Function ###
# def animate(time_data, name):
#     fig, ax = plt.subplots()
#     line, = ax.plot(time_data[0,:], alpha=0.7)
#     # ax.set_ylim(np.max(time_data) - 1, np.max(time_data) + 1)  # if it crashes on this line just comment it out
#     ax.set_xlabel("Position")
#     ax.set_ylabel(name)

#     def update(frame):
#         line.set_ydata(time_data[frame,:])
#         return line,

#     ani = animation.FuncAnimation(fig, update, frames=np.size(time_data, 0), interval=200, blit=True)
#     ani.save(f"{name}_animation.mp4", writer="ffmpeg")
#     plt.show()
#     plt.savefig('/home/nbaker47/Desktop/shock_animation')

### Plot-Saving Function ###
def animate(time_data, name):
    """saves individual plots for each variable"""

    # check if directories exist
    if os.path.exists("Plots"):
        print(f"Plotting directories exist.")
    else:
        os.mkdir("Plots")
        os.mkdir("Plots/density")
        os.mkdir("Plots/velocity_x")
        os.mkdir("Plots/velocity_y")
        os.mkdir("Plots/magnetic_field_y")
        os.mkdir("Plots/energy")
        os.mkdir("Plots/pressure")
        print(f"Plotting directories created.")


    for t, data in enumerate(time_data):

        img_path = 'Plots/' + name + '/time ' + str(round(t * dt, 4)) + '.png'
        plt.plot(data[:])
        plt.xlabel("Position")
        plt.ylabel(name)
        print(img_path)
        plt.savefig(img_path)
        plt.close()


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


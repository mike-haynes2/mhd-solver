import numpy as np
import math as m
from scipy import constants


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
# constant value for flow-aligned magnetic field B:
B_x_perscription = 3./4.    # nT    (see Equation 4.4 in Balbas et al)




## numerical parameters
################################################################################
################################################################################
# weighting threshold for MinMod derivative approximator
alpha = 1.

# spatial / time increment
# (CFL parameters)
CFL_velocity = 1.e+03 # 1 km / s (i.e., 'max' velocity we would expect)
CFL_safety_factor = 4.  # how close to CFL condition is our timestep
# perscribe spatial step
dx = 1.e-02              # 1 cm grid spacing
# calculate stable timestep dt
dt = (dx * CFL_safety_factor)/CFL_velocity
# spatial extent (1D case)
X_lower = -1.
X_upper = 1.
# time evolution (max time)
# in timesteps:
N_Tmax = 1000
# in seconds:
Tmax = N_Tmax*dt

################################################################################
################################################################################




## Initial conditions
################################################################################
################################################################################
# initialize spatial grid 

Xs = np.arange(start=X_lower, stop=X_upper,step=dx)
w_t0 = np.ones_like(Xs)



## MHD System
################################################################################
################################################################################
# 1-D equations (see eqns 4.1-4.2 in Balbas et al)
# continuity:
def f_continuity_1D(rho, u_vector):
    u_x = np.array(u_vector)[0]
    return (rho*u_x)
# EOM / Navier Stokes:
def f_NS_1D(rho, u_vector, B_vector, p):
    pstar = p + (np.dot(B_vector,B_vector)/mu0)
    xhat = np.array([1.,0.,0.])
    u_x = np.dot(np.array(u_vector),xhat)
    B_x = np.dot(np.array(B_vector),xhat)
    return (rho * u_x * u_vector - B_x * B_vector + pstar * xhat)
# Faraday's Law for MHD (dB/dt = - curl E = curl(u x B) = div(B tensor u - u tensor B))
def f_faraday_1D(u_vector, B_vector):
    By = u_vector[0] * B_vector[1] - B_vector[0] * u_vector[1]
    Bz = u_vector[0] * B_vector[2] - B_vector[0] * u_vector[2]
    return np.array([0.,By,Bz])
# energy equation
def f_energy_1D(Energy, u_vector, B_vector, p):
    pstar = p + (np.dot(B_vector,B_vector)/mu0)
    f1 = (Energy + pstar) * u_vector[0]
    f2 = B_vector[0] * np.dot(u_vector,B_vector)
    return (f1 - f2)
# adiabatic law
def calculate_p_adiabatic(rho,gamma,reference_rho=1.,reference_p=1.):
    c = (rho/reference_rho) ** gamma
    return (reference_p * c)

################################################################################
################################################################################



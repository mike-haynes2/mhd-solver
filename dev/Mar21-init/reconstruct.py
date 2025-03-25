import numpy as np
import configuration as config
from minmod import MinMod
from deltas import Deltas



## Function to construct the polynomial approximation from cell averages w_bar
def construct_poly_approx(wbar):
    for j in range(len(wbar)-1):
        if j != len(wbar)-2:
            Dplus, Dminus, D0 = Deltas(wbar[j],wbar[j-1],wbar[j+1])
            poly_approx = wbar[j] + (1./2.) * MinMod(config.alpha * Dplus, D0, config.alpha * Dminus)
        else:
            poly_approx = wbar[j] + (1./2.) * config.alpha * (wbar[j] - wbar[j-1])
    return poly_approx

### 3D RECONSTRUCT POLYNOMIAL


## Functions to be called after constructing cell averages into polynomial approximations (which are sampled at the grid points)

def reconstruct_rho(rho):
    return rho

def reconstruct_u_vector(u_vector, rho):
    return (u_vector / rho)

def reconstruct_B_vector(B_vector):
    return B_vector

def reconstruct_energy(energy):
    return energy

def reconstruct_pressure(rho, u_vector, B_vector, energy):
    p = np.zeros_like(rho)
    for i in range(len(p)):
        fact = (config.adiabatic_index - 1.)
        p[i] = fact * (energy[i] - ( ((1./2.)*rho[i]*np.dot(u_vector[i],u_vector[i])) +  (np.dot(B_vector[i],B_vector[i])/(2.*config.mu0))) )
    return p


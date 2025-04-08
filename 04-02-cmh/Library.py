import numpy as np
import math as m

from scipy import constants
import matplotlib.pyplot as plt
import os


# minmod (see equation 2.8 and following discussion in Balbas et al)
def minmod(alpha, wjm, wj, wjp):
    if np.sign(wjm) == np.sign(wj) and np.sign(wj) == np.sign(wjp):
        delminus = alpha * (wj - wjm)
        delplus = alpha * (wjp - wj)
        del0 = (1./2.) * (delminus + delplus)
        # calculate minimums in the least efficient and most verbose way possible (hey, it works)
        minl = min(delminus,del0)
        minp = min(delplus,del0)
        return_min = min(minl,minp)
        
        return np.sign(wj) * return_min
    
    else:
    
        return 0.




# calculate f() for all quantities in a single vector (following Balbas explicitly)
def calc_f(meshOBJ, Bx, gam):
    fs = np.zeros_like(meshOBJ)
    for i in range(meshOBJ.shape[1]):
        fs[:,i] = calc_f_cell(meshOBJ[:,i], Bx, gam)
    return fs



# function to actually encode the MHD equations: calculating the argument of the divergence at each cell
def calc_f_cell(cell, Bx, gam):
    # unpack the values of u vector (see equation 4.1 in balbas)
    rho, rho_ux, rho_uy, rho_uz, By, Bz, e = cell
    # equivalent of our old "reconstruct" routine
    ux = rho_ux / rho
    uy = rho_uy / rho
    uz = rho_uz / rho
    B2 = Bx**2 + By**2 + Bz**2
    v2 = ux**2 + uy**2 + uz**2
    
    # Energy equation
    p = (gam-1.) * (e - ((1./2.) * rho * v2) - (1./2.) * B2)
    pstar = p + (1./2.) * B2
    
    # calculate the equations of MHD in a vectorized manner
    # following the expression blindly from Balbas itself, instead of worrying about units and physics
    # see equation (4.2, 4.3)
    f0 = rho_ux
    f1 = (rho_ux**2 / rho) + pstar - Bx**2
    f2 = (rho_ux * uy) - Bx * By
    f3 = (rho_ux * uz) - Bx * Bz
    f4 = (By * ux) - Bx * uy
    f5 = (Bz * ux) - Bx * uz
    f6 = (e + pstar) * ux - Bx * np.dot(np.array([Bx,By,Bz]),np.array([ux,uy,uz]))
    
    return np.array([f0, f1, f2, f3, f4, f5, f6])
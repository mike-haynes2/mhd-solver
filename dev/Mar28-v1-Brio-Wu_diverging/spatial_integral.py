import numpy as np
import configuration as config
from minmod import MinMod, MinMod3D
from deltas import Deltas, Deltas3D

def Spatial_Integral(w_n, variable=''): # time is given by variable n
    """
    returns the value of the spatial integral at the next timestep n+1
    w_n: the cell averages from the previous time step
    """

    match variable:
        case 'rho':
            deltas = Deltas
            minmod = MinMod
        case 'B':
            deltas = Deltas3D
            minmod = MinMod3D
        case 'u':
            deltas = Deltas3D
            minmod = MinMod3D
        case 'energy':
            deltas = Deltas
            minmod = MinMod
        case _:
            raise ValueError('Bruh. must be "rho", "u", "energy"')

    # create array to store new values
    w_np1 = np.zeros_like(w_n)
    # apply boundary conditions (assumes time-independency of boundary conditions)
    w_np1[0] = w_n[0]
    w_np1[-1] = w_n[-1]

    # fill in w_np1 array
    for j in range(1, len(w_n)-1):

        ### w_prime_j ###
        # calculate deltas 
        delta_plus_j, delta_minus_j, delta_0_j = deltas(w_n[j], w_n[j-1], w_n[j+1])
        # calculate new step options
        a_j = delta_plus_j * config.alpha
        b_j = delta_0_j
        c_j = delta_minus_j * config.alpha
        # choose the minimum for each w_prime_j using MinMod
        w_prime_j = minmod(a_j, b_j, c_j)

        ### w_prime_jp1 ###
        if j == len(w_n)-2: # account for problematic j + 2 index
            # it might be possible to just make an additional ghost cell
            #w_prime_jp1 = (w_n[j+1] - w_n[j]) * config.alpha # use Euler backward ???still multiplying by alpha???
            if np.shape(w_n[j])==():
                w_prime_jp1 = 0.
            elif np.shape(w_n[j])==(3,):
                w_prime_jp1 = np.array([0.,0.,0.])
            else:
                raise TypeError('wrong shapes in spatial_integral')
        else:
            delta_plus_jp1, delta_minus_jp1, delta_0_jp1 = deltas(w_n[j+1], w_n[j], w_n[j+2])
            # calculate new step options
            a_jp1 = delta_plus_jp1 * config.alpha
            b_jp1 = delta_0_jp1
            c_jp1 = delta_minus_jp1 * config.alpha
            # choose the minimum for each w_prime_jp1 using MinMod
            w_prime_jp1 = minmod(a_jp1, b_jp1, c_jp1)

        ### calculate the integral ###
        # vector case
        if variable=='u' or variable=='B':
            I_j = np.array([0, 0, 0])
            for i in range(3):
                I_j[i] =  (1./2. * (w_n[j][i] + w_n[j+1][i]) + 1./8. * (w_prime_j[i] - w_prime_jp1[i] )) # 1. /config.dx
        # scalar case
        else:
            I_j =  (1./2. * (w_n[j] + w_n[j+1]) + 1./8. * (w_prime_j - w_prime_jp1  )) # 1. /config.dx *

        w_np1[j] = I_j

    return w_np1
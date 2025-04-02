import numpy as np
import configuration as config
from minmod import MinMod
from deltas import Deltas3D

def Temporal_Integral_B(w_n, u): # time is given by variable n
    """
    returns the value of the temporal integral at the next timestep n+1
    w_n: the cell averages from the previous time step
    f: the function
    """

    # calculate global parameter
    lam = config.dt / config.dx

    # create array in the shape of w_n to store new values
    w_np1 = np.zeros_like(w_n)
    # apply boundary conditions (assumes time-independency)
    w_np1[0] = w_n[0]
    w_np1[-1] = w_n[-1] ### THINK ABOUT BCS

    for j in range(1, len(w_n)-1):
        
        ### f_j ###
        # calculate deltas
        f_j = config.f_faraday_1D(w_n[j], u)
        f_jm1 = config.f_faraday_1D(w_n[j-1], u)
        f_jp1 = config.f_faraday_1D(w_n[j+1], u)
        delta_plus_j, delta_minus_j, delta_0_j = Deltas3D(f_j, f_jm1, f_jp1)

        # calculate new step options
        a_j = delta_plus_j * config.alpha
        b_j = delta_0_j
        c_j = delta_minus_j * config.alpha
        # choose the minimum for each f_prime_j using MinMod
        f_prime_j = MinMod(a_j, b_j, c_j)

        ### f_jp1 ###
        if j == len(w_n)-2: # account for problematic j + 2 index
            f_prime_jp1 = (f_jp1 - f_j) * config.alpha # use Euler backward ???still multiplying by alpha???
        else: 
            f_jp1 = config.f_faraday_1D(w_n[j+1], u)
            f_j = config.f_faraday_1D(w_n[j], u)
            f_jp2 = config.f_faraday_1D(w_n[j+2], u)
            delta_plus_jp1, delta_minus_jp1, delta_0_jp1 = Deltas3D(f_jp1, f_j, f_jp2)

            # calculate new step options
            a_jp1 = delta_plus_jp1 * config.alpha
            b_jp1 = delta_0_jp1
            c_jp1 = delta_minus_jp1 * config.alpha
            # choose the minimum for each f_prime_j using MinMod
            f_prime_jp1 = MinMod(a_jp1, b_jp1, c_jp1)

        ### calculate the integral ###
        I_j = lam * (config.f_faraday_1D(w_n[j] - lam/2 * f_prime_j, u) + config.f_faraday_1D(w_n[j+1] - lam/2 * f_prime_jp1, u))
        w_np1[j] = I_j

    return w_np1

# returning these two w_np1 grids will double the bcs, since we applied them both and then we will be adding them together. Make sure to halve them in sovler
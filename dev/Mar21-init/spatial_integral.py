import numpy as np
import configuration as config
from minmod import MinMod
from deltas import Deltas

def Spatial_Integral(w_n): # time is given by variable n
    """
    returns the value of the spatial integral at the next timestep n+1
    w_n: the cell averages from the previous time step
    """

    # create array to store new values
    w_np1 = np.array(len(w_n))
    # apply boundary conditions (assumes time-independency)
    w_np1[0] = w_n[0]
    w_np1[-1] = w_n[-1]

    # fill in w_np1 array
    for j in range(1, len(w_n)-1):

        ### w_prime_j ###
        # calculate deltas 
        delta_plus_j, delta_minus_j, delta_0_j = Deltas(w_n[j], w_n[j-1], w_n[j+1])
        # calculate new step options
        a_j = delta_plus_j * config.alpha
        b_j = delta_0_j
        c_j = delta_minus_j * config.alpha
        # choose the minimum for each w_prime_j using MinMod
        w_prime_j = MinMod(a_j, b_j, c_j)

        ### w_prime_jp1 ###
        if j == len(w_n)-2: # account for problematic j + 2 index
            w_prime_jp1 = (w_n[j+1] - w_n[j]) * config.alpha # use Euler backward ???still multiplying by alpha???
        else:
            delta_plus_jp1, delta_minus_jp1, delta_0_jp1 = Deltas(w_n[j+1], w_n[j], w_n[j+2])
            # calculate new step options
            a_jp1 = delta_plus_jp1 * config.alpha
            b_jp1 = delta_0_jp1
            c_jp1 = delta_minus_jp1 * config.alpha
            # choose the minimum for each w_prime_jp1 using MinMod
            w_prime_jp1 = MinMod(a_jp1, b_jp1, c_jp1)

        ### calculate the integral ###
        I_j = 1/config.dx * (1/2 * (w_n[j] + w_n[j+1]) + 1/8 * (w_prime_j + w_prime_jp1)) # ???keep or remove dx???
        w_np1[j] = I_j

    return w_np1
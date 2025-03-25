import numpy as np
import configuration as config
from minmod import MinMod
from deltas import Deltas3D, Deltas


def Temporal_Integral(w_n, rho=0, u=0, B=0, p=0, variable=''):  # time is given by variable n
    """
    returns the value of the temporal integral at the next timestep n+1
    w_n: the cell averages from the previous time step
    f: the function
    """
    match variable:
        case 'rho':
            f = config.f_continuity_1D
            inputs = [u]
            deltas = Deltas
        case 'B':
            f = config.f_faraday_1D
            inputs = [u]
            deltas = Deltas3D
        case 'u':
            f = config.f_NS_1D
            inputs = [rho, B, p]
            deltas = Deltas3D
        case 'energy':
            f = config.f_energy_1D
            inputs = [u, B, p]
            deltas = Deltas
        case _:
            raise ValueError('Bruh. must be "rho", "u", "energy"')

    # calculate global parameter
    lam = config.dt / config.dx

    # create array in the shape of w_n to store new values
    w_np1 = np.zeros_like(w_n)
    # apply boundary conditions (assumes time-independency)
    w_np1[0] = w_n[0]
    w_np1[-1] = w_n[-1]  ### THINK ABOUT BCS

    for j in range(1, len(w_n) - 1):

        ### f_j ###
        # calculate deltas
        f_j = f(w_n[j], inputs)
        f_jm1 = f(w_n[j - 1], inputs)
        f_jp1 = f(w_n[j + 1], inputs)
        delta_plus_j, delta_minus_j, delta_0_j = deltas(f_j, f_jm1, f_jp1)

        # calculate new step options
        a_j = delta_plus_j * config.alpha
        b_j = delta_0_j
        c_j = delta_minus_j * config.alpha
        # choose the minimum for each f_prime_j using MinMod
        f_prime_j = MinMod(a_j, b_j, c_j)

        ### f_jp1 ###
        if j == len(w_n) - 2:  # account for problematic j + 2 index
            f_prime_jp1 = (f_jp1 - f_j) * config.alpha  # use Euler backward ???still multiplying by alpha???
        else:
            f_jp1 = f(w_n[j + 1], inputs)
            f_j = f(w_n[j], inputs)
            f_jp2 = f(w_n[j + 2], inputs)
            delta_plus_jp1, delta_minus_jp1, delta_0_jp1 = deltas(f_jp1, f_j, f_jp2)

            # calculate new step options
            a_jp1 = delta_plus_jp1 * config.alpha
            b_jp1 = delta_0_jp1
            c_jp1 = delta_minus_jp1 * config.alpha
            # choose the minimum for each f_prime_j using MinMod
            f_prime_jp1 = MinMod(a_jp1, b_jp1, c_jp1)

        ### calculate the integral ###
        I_j = lam * (f(w_n[j] - lam / 2 * f_prime_j, inputs) + f(w_n[j + 1] - lam / 2 * f_prime_jp1, inputs))
        w_np1[j] = I_j

    return w_np1

# returning these two w_np1 grids will double the bcs, since we applied them both and then we will be adding them together. Make sure to halve them in sovler
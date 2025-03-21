import numpy as np
import matplotlib as plt
import configuration as config
import reconstruct as recon
from spatial_integral import Spatial_Integral
from temporal_integral_rho import Temporal_Integral_Rho
from temporal_integral_u import Temporal_Integral_U
from temporal_integral_b import Temporal_Integral_B
from temporal_integral_energy import Temporal_Integral_Energy

########################
### HELPER FUNCTIONS ###
########################
def next_step_rho(w_t, u):
    """
    calculates the next time step for some quantity q
    rho requires (vector) u
    """
    w_tp1 = Spatial_Integral(w_t) - Temporal_Integral_Rho(w_t, u) # update to take more variables
    w_tp1[0] = config.w_t0_x0
    w_tp1[-1] = config.w_t0_xM
    return w_tp1

def next_step_u(w_t, rho, B, p): # update to take more variables
    """
    calculates the next time step for some quantity q
    u requires rho, (vector) B and p
    """
    w_tp1 = Spatial_Integral(w_t) - Temporal_Integral_U(w_t, rho, B, p) # update to take more variables
    w_tp1[0] = config.w_t0_x0
    w_tp1[-1] = config.w_t0_xM
    return w_tp1

def next_step_B(w_t, u): # update to take more variables
    """
    calculates the next time step for some quantity q
    B requires (vector) u
    """
    w_tp1 = Spatial_Integral(w_t) - Temporal_Integral_B(w_t, u) # update to take more variables
    w_tp1[0] = config.w_t0_x0
    w_tp1[-1] = config.w_t0_xM
    return w_tp1

def next_step_energy(w_t, rho, u, B, p): # update to take more variables
    """
    calculates the next time step for some quantity q
    energy requires rho, u, B, p
    """
    w_tp1 = Spatial_Integral(w_t) - Temporal_Integral_Energy(w_t, rho, u, B, p) # update to take more variables
    w_tp1[0] = config.w_t0_x0
    w_tp1[-1] = config.w_t0_xM
    return w_tp1

############
### MAIN ###
############
def main():
    """runs the solver"""

    # quantities in the order they appear in all_quantities_t
    rho = 0
    u = 1
    B = 2
    energy = 3

    # initialize all lists for storing variable history
    all_quantities_t = config.all_quantities_t0     # does not store variable history
    rho_t = [all_quantities_t[rho]]                 # stores variable history
    u_t = [all_quantities_t[u]]                     # stores variable history
    B_t = [all_quantities_t[B]]                     # stores variable history
    energy_t = [all_quantities_t[energy]]           # stores variable history
    p_t = config.p0                                 # pressure

    # run loop to step through time in steps dt
    t = 0
    while t < config.Tmax:    

        # run loop to step through all quantities at time t
        for i, quantity in enumerate(all_quantities_t):

            if i == rho:
                # collect quantities (at time n) we will need to evaluate rho 
                u_n = u_t[-1]
                # iterate to the next time step, including polynomial construction
                w_tp1 = recon.construct_poly_approx(next_step_rho(quantity, u_n))
                # update all_quantities_t list with the most recent grid (overwrites previous time step)
                all_quantities_t[rho] = w_tp1.deepcopy()
                # append new grid to appropriate variable history list
                rho_t.append(w_tp1.deepcopy())
            elif i == u:
                # collect quantities at time n
                rho_n = rho_t[-2]
                B_n = B_t[-1]
                p_n = p_t
                # iterate
                w_tp1 = recon.construct_poly_approx(next_step_u(quantity, rho_n, B_n, p_n))
                # reconstruct u using updated t+1 (n+1) rho value
                rho_tp1 = all_quantities_t[rho]
                w_tp1 = recon.reconstruct_u_vector(w_tp1, rho_tp1)
                all_quantities_t[u] = w_tp1.deepcopy()
                u_t.append(w_tp1.deepcopy())
            elif i == B:
                # collect quantities at time n
                u_n = u_t[-2]
                # iterate, etc.
                w_tp1 = recon.construct_poly_approx(next_step_B(quantity, u_n))
                all_quantities_t[B] = w_tp1.deepcopy()
                B_t.append(w_tp1.deepcopy())
            elif i == energy:
                # collect quantities at time n
                rho_n = rho_t[-2]
                u_n = u_t[-2]
                B_n = B_t[-2]
                p_n = p_t
                # iterate, etc.
                w_tp1 = recon.construct_poly_approx(next_step_energy(quantity, rho_n, u_n, B_n, p_n))
                all_quantities_t[energy] = w_tp1.deepcopy()
                energy_t.append(w_tp1.deepcopy())
        
        # handle the pressure update
        rho_np1 = all_quantities_t[rho]
        u_np1 = all_quantities_t[u]
        B_np1 = all_quantities_t[B]
        energy_np1 = all_quantities_t[energy]
        p_t = recon.reconstruct_pressure(rho_np1, u_np1, B_np1, energy_np1)

        # iterate time
        t += config.dt

################
### RUN MAIN ###
################
if __name__ == '__main__':
    main()


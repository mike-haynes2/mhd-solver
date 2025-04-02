import numpy as np
import matplotlib as plt
import matplotlib.animation as animation
import configuration as config
import reconstruct as recon
from spatial_integral import Spatial_Integral
from temporal_integral import Temporal_Integral
from datetime import datetime

########################
### HELPER FUNCTIONS ###
########################
def next_step(w_t, rho=0, u=0, B=0, p=0, variable=''):
    """
    calculates the next time step for some quantity q
    rho requires (vector) u
    """

    w_tp1 = Spatial_Integral(w_t, variable=variable) - Temporal_Integral(w_t, rho=rho, u=u, B=B, p=p, variable=variable) # update to take more variables
    # spat_int = Spatial_Integral(w_t, variable=variable)
    # temp_int = Temporal_Integral(w_t, rho=rho, u=u, B=B, p=p, variable=variable)
    # w_tp1 = spat_int - temp_int
    # if variable=='u':
    #     print('spat_int: ',spat_int)
    #     print('temp_int: ',temp_int)
    # w_tp1 = Spatial_Integral(w_t) - Temporal_Integral(w_t, u=u, variable='rho') # general case
    # w_tp1[0] = config.w_t0_x0
    # w_tp1[-1] = config.w_t0_xM
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
    p_t = [config.p0]                               # stores variable history

    # run loop to step through time in steps dt
    t = 0
#    while t < config.Tmax:    
    while t < 2 * config.dt :    
        print(t)
        # run loop to step through all quantities at time t
        for i, quantity in enumerate(all_quantities_t):

            if i == rho:
                # collect quantities (at time n) we will need to evaluate rho 
                u_n = u_t[-1]
                # iterate to the next time step, including polynomial construction
                NextStep_rho = next_step(quantity, u=u_n, variable='rho')
                # print('next step rho [-1]: ',NextStep_rho[-1])
                # print('next step rho [0]: ', NextStep_rho[0])
                w_tp1 = recon.construct_poly_approx(NextStep_rho)

                # print('w_tp1[0] rho:',w_tp1[0])
                # print('w_tp1[-1] rho:',w_tp1[-1])
                # print('minimum rho: ',np.min(w_tp1))
                # update all_quantities_t list with the most recent grid (overwrites previous time step)
                all_quantities_t[rho] = w_tp1.copy()
                # append new grid to appropriate variable history list
                rho_t.append(w_tp1.copy())
            elif i == u:
                # collect quantities at time n
                rho_n = rho_t[-2]
                B_n = B_t[-1]
                p_n = p_t[-1]
                # iterate
                NextStep_u = next_step(quantity, rho=rho_n, B=B_n, p=p_n, variable='u')
                # print('next step u [-1]: ',NextStep_u[-1])
                # print('next step u [0]: ', NextStep_u[0])
                w_tp1 = recon.construct_poly_approx_3D(NextStep_u)
                # print('w_tp1[-1] u: ',w_tp1[-1])
                # print('w_tp1[0] u: ', w_tp1[0])

                # reconstruct u using updated t+1 (n+1) rho value
                rho_tp1 = all_quantities_t[rho]
                w_tp1 = recon.reconstruct_u_vector(w_tp1, rho_tp1)
                all_quantities_t[u] = w_tp1.copy()
                u_t.append(w_tp1.copy())
            elif i == B:
                # collect quantities at time n
                u_n = u_t[-2]
                # iterate, etc.
                w_tp1 = recon.construct_poly_approx_3D(next_step(quantity, u=u_n, variable='B'))
                all_quantities_t[B] = w_tp1.copy()
                B_t.append(w_tp1.copy())
            elif i == energy:
                # collect quantities at time n
                rho_n = rho_t[-2]
                u_n = u_t[-2]
                B_n = B_t[-2]
                p_n = p_t[-1]
                # iterate, etc.
                w_tp1 = recon.construct_poly_approx(next_step(quantity, rho=rho_n, u=u_n, B=B_n, p=p_n, variable='energy'))
                all_quantities_t[energy] = w_tp1.copy()
                energy_t.append(w_tp1.copy())
        
        # handle the pressure update
        rho_np1 = all_quantities_t[rho]
        u_np1 = all_quantities_t[u]
        B_np1 = all_quantities_t[B]
        energy_np1 = all_quantities_t[energy]
        p_t.append(recon.reconstruct_pressure(rho_np1, u_np1, B_np1, energy_np1))

        # iterate time
        t += config.dt

    # convert all lists to arrays
    rho_t = np.array(rho_t)
    u_t = np.array(u_t)
    B_t = np.array(B_t)
    energy_t = np.array(energy_t)
    p_t = np.array(p_t)  
    
    # After a single timestep, Bx and ux, uy are getting set to zero. check how the updated values propagate back through the method, see if any wrong expressions


    # print('shape rho, u(t)', np.shape(rho_t), np.shape(u_t))
    now_plotting = datetime.now()
    ###########plotting animation stuff############
    config.animate(rho_t, 'density', now_plotting)
    config.animate(u_t[:,:,0], 'velocity_x', now_plotting)        # need to make sure it is the right value I am grabbing
    config.animate(u_t[:,:,1], 'velocity_y', now_plotting)        # need to make sure it is the right value I am grabbing
    config.animate(B_t[:,:,1], 'magnetic_field_y', now_plotting) # need to make sure it is the right value I am grabbing
    config.animate(energy_t, 'energy', now_plotting)
    config.animate(p_t, 'pressure', now_plotting)

################
### RUN MAIN ###
################
if __name__ == '__main__':
    main()




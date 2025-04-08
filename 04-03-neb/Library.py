import numpy as np
import math as m

from scipy import constants
import matplotlib.pyplot as plt
import configuration as c
import os


# minmod (see equation 2.8 and following discussion in Balbas et al)
def minmod(alpha, wjm, wj, wjp):
    if np.sign(wjm) == np.sign(wj) and np.sign(wj) == np.sign(wjp):
        delminus = alpha * (wj - wjm)
        delplus = alpha * (wjp - wj)
        del0 = (1./2.) * (delminus + delplus)
        # calculate minimums in the least efficient and most verbose way possible
        dels = np.array([delminus, delplus, del0])
        
        return np.sign(wj) * np.nanmin(dels)
    
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

# plotting function
def animate(mesh, stagger_switch, tL, now):
    """saves individual plots for each variable"""

    meshOBJ = mesh.copy()

    plot_dir = 'Plots-- a=' + str(c.alpha) + ', Tmax=' + str(c.Tmax) + ', nx=' + str(c.nx) + ', dt,dx=' + str(c.dt) + ',' + str(c.dx)  
    # check if directories exist
    if os.path.exists(plot_dir):
        print(f"Saving new plots")
    else:
        os.mkdir(plot_dir)
        os.mkdir(plot_dir+"/rho")
        os.mkdir(plot_dir+"/u_x")
        os.mkdir(plot_dir+"/u_y")
        os.mkdir(plot_dir+"/u_z")
        os.mkdir(plot_dir+"/B_x")
        os.mkdir(plot_dir+"/B_y")
        os.mkdir(plot_dir+"/energy")
        os.mkdir(plot_dir+"/pressure")
        print(f"Plotting directories created. Saving new plots")

    ### plotting ###
    for i in range(c.num_vars):
        if i == 0:
            name = 'rho'
        elif i == 1:
            name = 'u_x'
            meshOBJ[stagger_switch, 1, :] /= meshOBJ[stagger_switch, 0, :] # divide by rho
        elif i == 2:
            name = 'u_y'
            meshOBJ[stagger_switch, 2, :] /= meshOBJ[stagger_switch, 0, :] # divide by rho
        elif i == 3:
            name = 'u_z'
            meshOBJ[stagger_switch, 3, :] /= meshOBJ[stagger_switch, 0, :] # divide by rho
        elif i == 4:
            name = 'B_x'
        elif i == 5:
            name = 'B_y'
        elif i == 6:
            name = 'energy'
        elif i == 7:
            name = 'pressure'
        
        # isolate variable array
        arr = meshOBJ[stagger_switch, i, :]
        # full individualized path
        img_path = plot_dir +'/' + name + '/tL' + str(tL) + '.png'
        # plot
        plt.plot(arr)
        plt.xlabel("position")
        plt.ylabel(name)
        plt.title('t = ' + str(tL))
        plt.grid()
        print(img_path)
        plt.savefig(img_path)
        plt.close()

    # rho_arr = meshOBJ[stagger_switch, 0, :]
    # u_x_arr = meshOBJ[stagger_switch, 1, :] / rho_arr
    # u_y_arr = meshOBJ[stagger_switch, 2, :] / rho_arr
    # u_z_arr = meshOBJ[stagger_switch, 3, :] / rho_arr
    # B_y_arr = meshOBJ[stagger_switch, 4, :] 
    # B_z_arr = meshOBJ[stagger_switch, 5, :]
    # en__arr = meshOBJ[stagger_switch, 6, :]
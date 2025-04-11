# mhd-solver: Computational Physics Term Project
# A Multi-Dimensional, Distributed, Non-Oscillatory Ideal Magnetohydrodynamics Central Differencing Framework
## Created 03/21
### C. Michael Haynes, Neil Baker, Kenneth Thompson, Corinne Hill, Eden Schapera

Please contact mhaynes@eas.gatech.edu for inquiries.

## Code Overview
The following is a brief description of the code and a mapping between code files and equations in the Balbas paper off which the model is based. For the 2-page progress report, see the template provided by Michael in Overleaf. Below is a brief description of each code file, which will serve as a guide to the code and the paper as the section of the report that describes the code is being written.
#### solver.py
The main() function, which is the solver method, handles the time stepping of the primary function, eq. (2.5), and to that end offloads tasks to other functions. The "next_step" function is simply a helper function that further delagates to the integral solvers. The main() function loops through time first, and then loops through all quantities (rho, u, B, energy, p). in each time step. This is because to reconstruct the value of each variable at each time step, the values of the other variables at the previous time step must be known.
#### spatial_integral.py
This file handles the space-integration (first) term in eq. (2.5). The argument of the integral is given by eq. (2.28).
#### temporal_integral.py
This file handles the time-integration (second, third) terms in eq. (2.5). Expressions for the integrals and arguments of the integrals are given by eqs. (2.19-2.20, 2.26). The functions f are the functions defined in configuration.py.
#### reconstruct.py
This function computes the polynomial approximations for each quantity based on cell averages. We used a second order reconstruction method eq. (2.6). It also contains four functions used to reconstruct the quantities we are interested in after the general conservation equation has been solved. For example, the solver for velocity u is actually finding the time evolution of the quantity (rho u), as per the continuity equation, eq. (12) of the project proposal. To account for this, the function f_continuity_1D divides the result by rho.
#### configuration.py
The configuration file is the file that handles all the initial conditions and simulation parameters. Ideally, at the start of a run, one must only modify this file. It also contains four functions which are the arguments of the conservation laws of the MHD equations. See eqs. (12-15) in the project proposal for these arguments. The current initial conditions correspond to the Brio-Wu shock tube problem, section 4.1 of Balbas' paper.
There are other initial test cases run that are under the names of 'slow shock', 'rarefaction', and 'Dai & Woodward' for other cases. To change to these cases simply put their name in the initialization function call.
#### deltas.py
The deltas function is used to calculate both the spatial and temporal integrals of Eq. (2.5). The function is defined in the footnote of p.264. We have also written a 3D version of the function, which handles vector (u, B) input. 
#### minmod.py
The minmod function is used to calculate both the spatial and temporal integrals of Eq. (2.5). The function is defined at the top of p.265. We have also written a 3D version of the function, which handles vector (u, B) input. 

### INSTRUCTIONS FOR RUNNING AND PLOTTING AS OF 04/11 ###
1. Decide on the initial states. The states and their initial conditions are listed in run_me.py lines 20-117
2. The initial state will have a name. Input this name (as a string) into run_me.py line 174 if running a single test, or into lines 166 or 171 if running a collection of sigmoid or alpha tests, respectively. If either sigmoid or alpha tests are being run, comment out line 175 and uncomment line 168 or 172, respectively. 
3. At the terminal, run run_me.py. This will generate an output folder and a collection of zip files containing output data at each time step. The name of this new folder should go into bruh.py line 12.
4. Once the program run_me.py has run, run bruh.py at the terminal. This will generate output plots based on the plotting routing contained within bruh.


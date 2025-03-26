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
This function computes the polynomial approximations for each quantity based on cell averages. We used a second order reconstruction method eq. (2.6). It also contains four functions used to reconstruct the quantities we are interested in after the general conservation equation has been solved. For example, the continuity equation states that $ \partial_t(\rho) + \nabla \cdot (\rho \vec{u}) = 0 $, so the solver for $\vec{u}$ is actually finding the time evolution of the quantity $ \rho \vec{u} $. To account for this, the function f_continuity_1D divides the result by $\rho$.
#### configuration.py
The configuration file is the file that handles all of the initial conditions and simulation parameters. Ideally, at the start of a run, one must only modify this file. It also contains four functions which are the arguments of the conservation laws of the MHD equations. See eqs. (12-15) in the project proposal for these arguments. The current initial conditions correspond to the Brio-Wu shock tube problem, section 4.1 of Balbas' paper
#### deltas.py
The deltas function is used to calculate both the spatial and temporal integrals of Eq. (2.5). The function is defined in the foornote of p.264. We have also written a 3D version of the function, which handles vector (u, B) input. 
#### minmod.py
The minmod function is used to calculate both the spatial and temporal integrals of Eq. (2.5). The function is defined at the top of p.265. We have also written a 3D version of the function, which handles vector (u, B) input. 

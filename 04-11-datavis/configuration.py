import numpy as np
import math as m

from scipy import constants
import matplotlib.pyplot as plt

# would be nice to include the match case thing with the other shock configurations (and we can futher validate the code with the hydrodynamic test case)

# values to pass in alpha, nx, T_max, initial_CFL safety, initial_Condition_name,


# initialize parameters (in this 1D model, Bx is a mere parameter)
## Physical parameters
# derivative factor in minmod, still need to implement

## time domain
# set dt according to rough courant (CFL) estimate. Velocities in their runs are order


## define objects to store everything / initialize


# problem to solve
name = 'Brio & Wu'


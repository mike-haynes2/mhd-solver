import numpy as np
from minmod import MinMod
from deltas import Deltas

def spatial_integral(dx, alpha, w_n):
    """returns the value of the spatial integral at the next timestep n+1
    dx: the spatial step
    alpha: a parameter in the range [1, 4)
    w_n: the cell averages for the previous time step"""

    # create array to store new values
    w_n_new = np.array(len(w_n))
    # apply boundary conditions
    w_n_new[0] = w_n[0]
    w_n_new[-1] = w_n[-1]

    for j in range(1, len(w_n)-1):

        delta_plus, delta_minus, delta_0 = Deltas(w_n[j], w_n[j-1], w_n[j+1])

        a = alpha * 
        b = delta_0 * w[j]
        c = alpha * delta_minus * w[j]
        w_primej = MinMod()
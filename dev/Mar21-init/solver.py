import numpy as np
import matplotlib as plt
import configuration as config
from spatial_integral import Spatial_Integral
from temporal_integral import Temporal_Integral

def main():
    """runs the solver"""

    # initialize time and grids
    t = 0
    w_t = config.w_t0
    w_T = [w_t] # store history

    # run loop to step through time in steps dt
    while t < config.Tmax:
        w_tp1 = Spatial_Integral(w_t) - Temporal_Integral(w_t, f)
        copy = w_tp1.deepcopy()
        w_T += [copy]
        w_t = copy
        t += config.dt
    
if __name__ == '__main__':
    pass


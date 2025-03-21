import numpy as np

def Deltas(w_j, w_jm1, w_jp1):
    """calculates the ###"""

    delta_plus = w_jp1 - w_j
    delta_minus = w_j - w_jm1
    delta_0 = 1/2 * (delta_plus + delta_minus)

    return delta_plus, delta_minus, delta_0
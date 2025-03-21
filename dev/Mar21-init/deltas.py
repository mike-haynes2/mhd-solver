import numpy as np

def Deltas(q_j, q_jm1, q_jp1):
    """
    calculates delta_plus/minus/0 for some quantity q evaluated at timesteps j, j-1 and j+1
    """

    delta_plus = q_jp1 - q_j
    delta_minus = q_j - q_jm1
    delta_0 = 1/2 * (delta_plus + delta_minus)

    return delta_plus, delta_minus, delta_0
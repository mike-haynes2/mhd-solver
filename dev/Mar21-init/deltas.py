import numpy as np

def Deltas(w_j, w_jm1, w_jp1):
    """
    calculates delta_plus/minus/0 for some quantity q evaluated at timesteps j, j-1 and j+1
    """
    print('w_j is', w_j)
    delta_plus = w_jp1 - w_j
    delta_minus = w_j - w_jm1
    delta_0 = 1/2 * (delta_plus + delta_minus)

    return delta_plus, delta_minus, delta_0

def Deltas3D(w_j, w_jm1, w_jp1):

    wx_delta_plus, wx_delta_minus, wx_delta_0 = Deltas(w_j[0], w_jm1[0], w_jp1[0])
    wy_delta_plus, wy_delta_minus, wy_delta_0 = Deltas(w_j[1], w_jm1[1], w_jp1[1])
    wz_delta_plus, wz_delta_minus, wz_delta_0 = Deltas(w_j[2], w_jm1[2], w_jp1[2])

    delta_plus = np.array([wx_delta_plus, wy_delta_plus, wz_delta_plus])
    delta_minus = np.array([wx_delta_minus, wy_delta_minus, wz_delta_minus])
    delta_0 = np.array([wx_delta_0, wy_delta_0, wz_delta_0])

    return delta_plus, delta_minus, delta_0
        
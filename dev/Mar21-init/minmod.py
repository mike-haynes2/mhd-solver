import numpy as np

def MinMod(a, b, c):
    """returns the minimum value of three stepper methods a, b and c"""
    if np.sign(a) == np.sign(b) and np.sign(b) == np.sign(c):
        minimum = min(abs(a), abs(b), abs(c))
        return np.sign(a) * minimum
    else: 
        return 0

if __name__ == '__main__':
    print(MinMod(-1.44, -8.5, -3.6)) # returns -1.44

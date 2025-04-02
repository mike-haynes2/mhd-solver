import numpy as np

def MinMod(a, b, c):
    """returns the minimum value of three stepper methods a, b and c"""
    # print(a, b, c)
    if np.sign(a) == np.sign(b) and np.sign(b) == np.sign(c):
        minimum = min(abs(a), abs(b), abs(c))
        return np.sign(a) * minimum
    else: 
        return 0
    
def MinMod3D(a, b, c):

    vector = [0, 0, 0]
    for j in range(len(a)):
        value = MinMod(a[j], b[j], c[j])
        vector[j] = value

    return vector

# if __name__ == '__main__':
#     print(MinMod(-1.44, -8.5, -3.6)) # returns -1.44

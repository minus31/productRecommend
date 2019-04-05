import numpy as np 


def l2_distance(p, q):
    return np.linalg.norm(p - q)

def l2_normalize(v):

    norm = np.linalg.norm(v, axis=1, keepdims=True)

    return np.divide(v, norm, where=norm!=0)
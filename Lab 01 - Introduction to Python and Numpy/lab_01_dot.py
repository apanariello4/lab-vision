import numpy as np

a, b = np.ndarray()

def dot(a, b):
    out = np.zeros((a.shape[0],b.shape[1]))
    if len(a.shape) == 2 and len(b.shape) == 2 and a.shape[1] == b.shape[0]:
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                out[i,j] = np.sum(a[i,:] * b[:,j])
        return out
    else:
        print("Wrong shapes")
        return out

out = dot(a, b)
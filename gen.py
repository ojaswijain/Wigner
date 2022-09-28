from sympy.physics.wigner import wigner_3j
import numpy as np
import pipeline
from mpmath import *
from sympy import *
import pickle as pkl
import time

mp.dps=500

def gen(l):
    #l can be a list or an np array
    l=np.array(l)
    wigner0 = np.array([l,[0, 0, 0]])
    v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
    v=mpf(v)
    pipeline.store_val_ana(wigner0[0],wigner0[1], v)
    for i in [0,1]:
        for j in range(0,-i,-1):
            k=-i-j
            wigner0 = np.array([l,[i, j, k]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([l,[-i,-j,-k]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([l,[k, i, j]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([l,[-k,-i,-j]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([l,[j, k, i]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([l,[-j,-k,-i]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),50)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
        with open('ana.pkl', 'wb') as handle:
            pkl.dump(pipeline.wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    start=time.time()
    #specify l values here
    for l in [2]:
    # for l in range(1,1001):
        gen([l,l+1,l+3])
    end=time.time()
    print("Time taken: ", end-start)

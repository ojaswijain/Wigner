from sympy.physics.wigner import wigner_3j
import numpy as np
import pipeline
from mpmath import *
from sympy import *
import pickle as pkl
import time

def gen(l):
    for i in [0,1]:
        for j in range(0,-i,-1):
            k=-i-j
            wigner0 = np.array([[l,l,l],[i, j, k]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([[l,l,l],[-i,-j,-k]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([[l,l,l],[k, i, j]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([[l,l,l],[-k,-i,-j]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([[l,l,l],[j, k, i]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
            wigner0 = np.array([[l,l,l],[-j,-k,-i]])
            v=N(wigner_3j(*wigner0[0],*wigner0[1]),30)
            v=mpf(v)
            pipeline.store_val_ana(wigner0[0],wigner0[1], v)
        with open('ana.pkl', 'wb') as handle:
            pkl.dump(pipeline.wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    start=time.time()
    #specify l values here
    for l in [10, 20, 30, 50, 80, 100, 200, 300, 500, 800, 1000]:
        gen(l)
    end=time.time()
    print("Time taken: ", end-start)

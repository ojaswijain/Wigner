from gen import gen
import simple_wigner as sw
from gaunt import gaunt_rec 
import numpy as np
import pipeline
import sys
import time

def gaunt_gen(l):
    f = open("gaunt.pkl", "w")
    f.close()
    f = open("ana.pkl", "w")
    f.close()
    gen(l)
    sw.wigner3j(l)
    return gaunt_rec(l)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <l1 l2 l3>")
        exit(1)
    f = open("gaunt.pkl", "w")
    f.close()
    f = open("ana.pkl", "w")
    f.close()
    
    start = time.time()
    l = np.array([int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])])
    print(gaunt_gen(l))
    print(time.time()-start)
    



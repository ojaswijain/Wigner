import pickle as pkl
import numpy as np

def strip_key(s):
    lis=s.split(".")
    l1=[int(lis[0]), int(lis[1]), int(lis[2])]
    m1=[int(lis[3]), int(lis[4]), int(lis[5])]
    l=np.array(l1)
    m=np.array(m1)
    return l,m

file = open("ana.pkl", "rb")
dic = pkl.load(file)
print(dic)


import pickle as pkl
import numpy as np

def strip_key(s):
    lis=s.split(".")
    l1=[int(lis[0]), int(lis[1]), int(lis[2])]
    m1=[int(lis[3]), int(lis[4]), int(lis[5])]
    l=np.array(l1)
    m=np.array(m1)
    return l,m

def to_key(l1,l2,l3,m1,m2,m3):
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))
    return idx1+idx2

def load_all_keys():
    ftr=open("ana.pkl", 'rb')
    dic=pkl.load(ftr)
    all_keys=dic.keys()
    known_3js=[]
    for x in all_keys:
        l,m=strip_key(x)
        known_3js.append([list(l),list(m)])
    return known_3js


file = open("ana.pkl", "rb")
dic = pkl.load(file)
# print(dic)


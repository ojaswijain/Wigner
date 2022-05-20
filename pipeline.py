# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:30:16 2022

@author: ojasw
"""

import numpy as np
import pickle as pkl
import sys
from multipledispatch import dispatch

wigner_dict_ana={
    
}

wigner_dict_rec={
    
}

@dispatch(int, int, int, int, int, int, float)
def store_val_ana(l1,l2,l3,m1,m2,m3, val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))
    
    wigner_dict_ana[idx1+idx2]=val
    with open('ana.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return

@dispatch(np.ndarray, float)
def store_val_ana(arr1, val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    arr=arr1.astype(dtype=int)
    idx1=str(arr[0])+"."+str(arr[1])+"."+str(arr[2])+"."
    idx2=str(arr[3])+"."+str(arr[4])+"."+str(arr[5])
    
    wigner_dict_ana[idx1+idx2]=val
    with open('ana.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return

@dispatch(np.ndarray, np.ndarray, float)
def store_val_ana(l1, m1, val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])
    
    wigner_dict_ana[idx1+idx2]=val
    with open('ana.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return  

@dispatch(int, int, int, int, int, int)
def give_val_ana(l1,l2,l3,m1,m2,m3):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))

    file = open("ana.pkl", "rb")
    wigner_dict_ana = pkl.load(file)

    return wigner_dict_ana[idx1+idx2]

@dispatch(np.ndarray)
def give_val_ana(arr1):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    arr=arr1.astype(dtype=int)
    idx1=str(arr[0])+"."+str(arr[1])+"."+str(arr[2])+"."
    idx2=str(arr[3])+"."+str(arr[4])+"."+str(arr[5])

    file = open("ana.pkl", "rb")
    wigner_dict_ana = pkl.load(file)

    return wigner_dict_ana[idx1+idx2]

@dispatch(np.ndarray, np.ndarray)
def give_val_ana(l1, m1):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])

    file = open("ana.pkl", "rb")
    wigner_dict_ana = pkl.load(file)

    return wigner_dict_ana[idx1+idx2]

@dispatch(int, int, int, int, int, int, float)
def store_val_rec(l1,l2,l3,m1,m2,m3,val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))
    
    wigner_dict_rec[idx1+idx2]=val
    with open('rec.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_rec, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return 

@dispatch(np.ndarray, float)
def store_val_rec(arr1,val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    arr=arr1.astype(dtype=int)
    idx1=str(arr[0])+"."+str(arr[1])+"."+str(arr[2])+"."
    idx2=str(arr[3])+"."+str(arr[4])+"."+str(arr[5])
    
    wigner_dict_rec[idx1+idx2]=val
    with open('rec.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_rec, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return

@dispatch(np.ndarray, np.ndarray, float)
def store_val_rec(l1, m1,val):
    '''
    Calculate the key of the 
    given wigner 3-j and store
    the value in the dictionary
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])
    
    wigner_dict_rec[idx1+idx2]=val
    with open('rec.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_rec, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return 

@dispatch(int, int, int, int, int, int, int)
def give_val_rec(l1,l2,l3,m1,m2,m3):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))

    file = open("rec.pkl", "rb")
    wigner_dict_rec = pkl.load(file)

    return wigner_dict_rec[idx1+idx2]

@dispatch(np.ndarray)
def give_val_rec(arr1):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    arr=arr1.astype(dtype=int)
    idx1=str(arr[0])+"."+str(arr[1])+"."+str(arr[2])+"."
    idx2=str(arr[3])+"."+str(arr[4])+"."+str(arr[5])

    file = open("rec.pkl", "rb")
    wigner_dict_rec = pkl.load(file)

    return wigner_dict_rec[idx1+idx2]

@dispatch(np.ndarray, np.ndarray)
def give_val_rec(l1, m1):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])

    file = open("rec.pkl", "rb")
    wigner_dict_rec = pkl.load(file)

    return wigner_dict_rec[idx1+idx2]

def main():
    args = sys.argv[:]
    with open('ana.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open('rec.pkl', 'wb') as handle:
        pkl.dump(wigner_dict_rec, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if args[0]==0:
        store_val_ana(args[1],args[2],args[3],args[4],args[5],args[6],args[7])

    if args[0]==1:
        store_val_rec(args[1],args[2],args[3],args[4],args[5],args[6],args[7])

    if args[0]==2:
        give_val_ana(args[1],args[2],args[3],args[4],args[5],args[6])

    else:
        return

    if args[0]==3:
        give_val_rec(args[1],args[2],args[3],args[4],args[5],args[6])

if __name__ == "__main__":
    main()
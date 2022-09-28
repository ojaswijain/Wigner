# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:30:16 2022

@author: ojasw
"""

import numpy as np
import pickle as pkl
import sys
from multipledispatch import dispatch
from mpmath import *

wigner_dict_ana={
    
}

gaunt_dic={

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
    # with open('ana.pkl', 'wb') as handle:
    #     pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # return

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
    # with open('ana.pkl', 'wb') as handle:
    #     pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # return

# @dispatch(np.ndarray, np.ndarray, float)
@dispatch(np.ndarray, np.ndarray, mpf)
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
    # with open('ana.pkl', 'wb') as handle:
    #     pkl.dump(wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # return  

@dispatch(int, int, int, int, int, int)
def give_val_ana(l1,l2,l3,m1,m2,m3):
    '''
    Calculate the key of the 
    required wigner 3-j and return
    the value from the dictionary
    '''
    idx1=str(int(l1))+"."+str(int(l2))+"."+str(int(l3))+"."
    idx2=str(int(m1))+"."+str(int(m2))+"."+str(int(m3))

    # file = open("ana.pkl", "rb")
    # wigner_dict_ana = pkl.load(file)

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

    # file = open("ana.pkl", "rb")
    # wigner_dict_ana = pkl.load(file)

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

    # file = open("ana.pkl", "rb")
    # wigner_dict_ana = pkl.load(file)

    return wigner_dict_ana[idx1+idx2]


def wigner_3j(l1, m1):
    '''
    return wigner_3j value
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])

    return wigner_dict_ana[idx1+idx2]


def gaunt(l1, m1):
    '''
    Return Gaunt Value
    '''
    l=l1.astype(dtype=int)
    m=m1.astype(dtype=int)
    idx1=str(l[0])+"."+str(l[1])+"."+str(l[2])+"."
    idx2=str(m[0])+"."+str(m[1])+"."+str(m[2])

    return gaunt_dic[idx1+idx2]


if __name__ == "__main__":

    file = open("ana.pkl", "rb")
    wigner_dict_ana = pkl.load(file)

    file = open("gaunt.pkl", "rb")
    gaunt_dic = pkl.load(file)

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:30:16 2022

@author: ojasw
"""

import pipeline
import numpy as np

file = open('val.dat', 'r')

lines = file.readlines()

for line in lines:
    arr = np.fromstring(line, sep = ",")
    val=arr[-1]
    print(val)
    pipeline.store_val_ana(arr, val)
    x=pipeline.give_val_ana(arr)
    print(x)

file.close()
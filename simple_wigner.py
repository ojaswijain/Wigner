# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:30:16 2022

@authors: joao alberto, ojasw
"""

import numpy as np
from sympy.physics.wigner import wigner_3j
import time
import pipeline
from plotter import plotter, plot_time, heatmap, plot_wig
import pandas as pd
import utils
from gen import gen
import pickle as pkl
from mpmath import *
from sympy import *


mp.dps=30

def print_lll(wigner3j):
	l1,l2,l3 = wigner3j[0][0], wigner3j[0][1], wigner3j[0][2]
	m1,m2,m3 = wigner3j[1][0], wigner3j[1][1], wigner3j[1][2]
	print("| {}\t{}\t{}\t|\n| {}\t{}\t{}\t|\n".format(l1,l2,l3,m1,m2,m3))

def recursion_coeffs(wigner3j, m_idx, sign):
	'''
	m_idx is the index of the m chosen to be lowered down
	or summed up, and sign indicates which operation
	has been performed over it (+1 or -1).
	'''
	coeffs = [0,0,0]
	if m_idx==0:
		
		coeffs[1] = mp.sqrt((wigner3j[0][0]+sign*(wigner3j[1][0]-sign)+1)*
							(wigner3j[0][0]-sign*(wigner3j[1][0]-sign)))
		coeffs[0] = mp.sqrt((wigner3j[0][1]+sign*wigner3j[1][1]+1)*
							(wigner3j[0][1]-sign*wigner3j[1][1]))
		coeffs[2] = mp.sqrt((wigner3j[0][2]+sign*wigner3j[1][2]+1)*
							(wigner3j[0][2]-sign*wigner3j[1][2]))
	
	elif m_idx==1:
	
		coeffs[2] = mp.sqrt((wigner3j[0][0]+sign*wigner3j[1][0]+1)*
							(wigner3j[0][0]-sign*wigner3j[1][0]))
		coeffs[1] = mp.sqrt((wigner3j[0][1]+sign*(wigner3j[1][1]-sign)+1)*
							(wigner3j[0][1]-sign*(wigner3j[1][1]-sign)))
		coeffs[0] = mp.sqrt((wigner3j[0][2]+sign*wigner3j[1][2]+1)*
							(wigner3j[0][2]-sign*wigner3j[1][2]))
	
	elif m_idx==2:
	
		coeffs[0] = mp.sqrt((wigner3j[0][0]+sign*wigner3j[1][0]+1)*
							(wigner3j[0][0]-sign*wigner3j[1][0]))
		coeffs[2] = mp.sqrt((wigner3j[0][1]+sign*wigner3j[1][1]+1)*
							(wigner3j[0][1]-sign*wigner3j[1][1]))
		coeffs[1] = mp.sqrt((wigner3j[0][2]+sign*(wigner3j[1][2]-sign)+1)*
							(wigner3j[0][2]-sign*(wigner3j[1][2]-sign)))

	return coeffs

def recursion_brute(wigner3j):
	
	sign1 = np.sign(wigner3j[1][0])
	sign2 =	np.sign(wigner3j[1][1])
	sign3 =	np.sign(wigner3j[1][2])

	a = abs(wigner3j[1][0])
	b = abs(wigner3j[1][1])
	c = abs(wigner3j[1][2])	

	if a>b and a>c:
		sign = sign1
		wigner1 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0]-sign,
				    wigner3j[1][1]+sign,
				    wigner3j[1][2]]]
				  )
						    
		wigner3 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0]-sign,
				    wigner3j[1][1],
				    wigner3j[1][2]+sign]]
				  )
		
		coeffs = recursion_coeffs(wigner3j, 0, sign)

	elif b>a and b>c:
		sign = sign2
		wigner1 = np.array(
				[wigner3j[0],
				[wigner3j[1][0],			   
				wigner3j[1][1]-sign,
				wigner3j[1][2]+sign]]
				)

		wigner3 = np.array(
				[wigner3j[0],
				[wigner3j[1][0]+sign,
				wigner3j[1][1]-sign,
				wigner3j[1][2]]]
			)
		
		coeffs = recursion_coeffs(wigner3j, 1, sign)

	elif c>a and c>b:
		sign = sign3
		wigner1 = np.array(
			[wigner3j[0],
			[wigner3j[1][0]+sign,
			wigner3j[1][1],
			wigner3j[1][2]-sign]]
		)

		wigner3 = np.array(
			[wigner3j[0],
			[wigner3j[1][0],
			wigner3j[1][1]+sign,
			wigner3j[1][2]-sign]]
		)

		coeffs = recursion_coeffs(wigner3j, 2, sign)	

	elif a==b:
		sign = sign1
		wigner1 = np.array(
			[wigner3j[0],
			[wigner3j[1][0]-sign,
			wigner3j[1][1]+sign,
			wigner3j[1][2]]]
		)
		wigner3 = np.array(
			[wigner3j[0],
			[wigner3j[1][0]-sign,
			wigner3j[1][1],
			wigner3j[1][2]+sign]]
		)

		coeffs = recursion_coeffs(wigner3j, 0, sign)
	
	elif b==c:
		sign = sign2
		wigner1 = np.array(
			[wigner3j[0],
			[wigner3j[1][0],
			wigner3j[1][1]-sign,
			wigner3j[1][2]+sign,]]
		)
		wigner3 = np.array(
			[wigner3j[0],
			[wigner3j[1][0]+sign,
			wigner3j[1][1]-sign,
			wigner3j[1][2]]]
		)

		coeffs = recursion_coeffs(wigner3j, 1, sign)

	elif c==a:
		sign = sign3
		wigner1 = np.array(
			[wigner3j[0],
			[wigner3j[1][0]+sign,
			wigner3j[1][1],
			wigner3j[1][2]-sign]]
		)
		wigner3 = np.array(
			[wigner3j[0],
			[wigner3j[1][0],
			wigner3j[1][1]+sign,
			wigner3j[1][2]-sign]]
		)

		coeffs = recursion_coeffs(wigner3j, 2, sign)

	return wigner1, wigner3, coeffs
			
x=[]
y=[]
a=[]
b=[]		

def wigner_sol(wigner3j):

	global known_3js, x, y

	wig2 = np.array(wigner3j)

	# if wigner3j.tolist() in known_3js:
	# 	# print_lll(wigner3j)
	# 	val = pipeline.give_val_ana(wig2[0], wig2[1])
	# 	# w = N(wigner_3j(*wigner3j[0],*wigner3j[1]),30)
	# 	# w = mpf(w)
	# 	# e = float((w-val)/w)
	# 	# print("Calculated: " + str(val) +
	# 	# 		"\nCorrect: " + str(w) +
	# 	# 		"\nRel. Error: " + str(e) +
	# 	# 		"\n-----------------------------------------------------")

	# 	# x.append(np.abs(wigner3j[1]).max())
	# 	# a.append(wigner3j[1][0])
	# 	# b.append(wigner3j[1][1])
	# 	# y.append(e)
	# 	return val

	# known_3js.append(wigner3j.tolist())
	wigner1, wigner3, coeff= recursion_brute(wigner3j)
	wig1 = np.array(wigner1)
	wig3 = np.array(wigner3)
	val1 = mpf(pipeline.give_val_ana(wig1[0], wig1[1]))
	val3 = mpf(pipeline.give_val_ana(wig3[0], wig3[1]))

	val = mpf(-1*(val1*coeff[0]+val3*coeff[2])/coeff[1])

	pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]), val)

	#Comment out everything except line 227 to measure true speed of algorithm
	# print_lll(wigner3j)
	# w = N(wigner_3j(*wigner3j[0],*wigner3j[1]),30)
	# e = float((w-val)/w)
	# print("Calculated: " + str(val) +
    #         "\nCorrect: " + str(w) +
	# 		"\nRel. Error: " + str(e) +
	# 		"\n-----------------------------------------------------")

	x.append(np.abs(wigner3j[1]).max())
	# a.append(wigner3j[1][0])
	# b.append(wigner3j[1][1])
	# y.append(e)

	return val

if __name__ == "__main__":

	start = time.time()
	t=[]
	v=[]
	L=[]
	T=[]

	file = open("ana.pkl", "rb")
	pipeline.wigner_dict_ana = pkl.load(file)
	known_3js=utils.load_all_keys()	
	#specify l values here
	for l in [10, 20, 30]:#, 50, 80, 100, 200, 300, 500, 800, 1000]:
		s=time.time()
		# gen(l)
		# with open('ana.pkl', 'wb') as handle:
		# 	pkl.dump(pipeline.wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
		for m in range(2,l+1,1):
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([[l,l,l],[i,j,m]])
				wigner_sol(wigner0)
				wigner0 = np.array([[l,l,l],[-i,-j,-m]])
				wigner_sol(wigner0)
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([[l,l,l],[m,i,j]])
				wigner_sol(wigner0)
				wigner0 = np.array([[l,l,l],[-m,-i,-j]])
				wigner_sol(wigner0)
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([[l,l,l],[i,m,j]])
				wigner_sol(wigner0)
				wigner0 = np.array([[l,l,l],[-i,-m,-j]])
				wigner_sol(wigner0)
			
			wigner0=np.array([[l,l,l],[0,-m,m]])
			wigner_sol(wigner0)
			wigner0=np.array([[l,l,l],[0,m,-m]])
			wigner_sol(wigner0)
			wigner0=np.array([[l,l,l],[m,0,-m]])
			wigner_sol(wigner0)
			wigner0=np.array([[l,l,l],[-m,0,m]])
			wigner_sol(wigner0)
			wigner0=np.array([[l,l,l],[m,-m,0]])
			wigner_sol(wigner0)
			wigner0=np.array([[l,l,l],[-m,m,0]])
			wigner_sol(wigner0)
			end = time.time()
			t.append(end-start)
			v.append(len(x))
		L.append(l)
		T.append(end-s)
		
	# df = pd.DataFrame({"mprove50_x":x, "mprove50_y":y})
	# df.to_csv("errors.csv", index=False)
	# df = pd.DataFrame({"mprove50_x":v, "mprove50_y":t})
	# df.to_csv("time.csv", index=False)

	#create a heatmap of the errors. Ignore scale
    # heatmap(a,b,y,"simple100")
	#Scatter plot of error
    # plotter(x,y,"simple","Red")
	with open('ana.pkl', 'wb') as handle:
	   pkl.dump(pipeline.wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
	#Plot time against number of Wigners

plot_time(v,t,"Time_vs_Wigners","Blue")
plot_wig(L,T,"Time_vs_L","Blue")

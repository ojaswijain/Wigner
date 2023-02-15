# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:30:16 2022

@authors: joao alberto, ojasw
"""

import numpy as np
from sympy.physics.wigner import wigner_3j
import time
import pipeline
# from plotter import plotter, plot_time, heatmap, plot_wig
import utils
from gen import gen
import pickle as pkl
from mpmath import *
from sympy import *


mp.dps=500

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

	#catching exceptions
	l=np.array(wigner3j)[0]
	m=np.array(wigner3j)[1]
	if l[0]<abs(m[0]) or l[1]<abs(m[1]) or l[2]<abs(m[2]):
		pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]), mpf(0.0))
		return
	# if l[0]<0 or l[1]<0 or l[2]<0:
		# pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]), mpf(0.0))
		# return
	if l[0]>l[1]+l[2] or l[1]>l[0]+l[2] or l[2]>l[0]+l[1]:
		pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]),mpf(0.0))
		return
	if l[0]==0 and l[1]==0 and l[2]==0:
		pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]), mpf(1.0))
		return 

	wig2 = np.array(wigner3j)
	for i in range(0,3):
		if wig2[0][i]<abs(wig2[0][i]):
			return

	wigner1, wigner3, coeff= recursion_brute(wigner3j)
	wig1 = np.array(wigner1)
	wig3 = np.array(wigner3)
	val1 = mpf(pipeline.give_val_ana(wig1[0], wig1[1]))
	val3 = mpf(pipeline.give_val_ana(wig3[0], wig3[1]))

	val = mpf(-1*(val1*coeff[0]+val3*coeff[2])/coeff[1])

	pipeline.store_val_ana(np.array(wigner3j[0]), np.array(wigner3j[1]), val)

	#Comment out everything except marked line to measure true speed of algorithm
	# print_lll(wigner3j)
	# w = N(wigner_3j(*wigner3j[0],*wigner3j[1]),50)
	# e = float((w-val)/w)
	# print("Calculated: " + str(val) +
    #         "\nCorrect: " + str(w) +
	# 		"\nRel. Error: " + str(e) +
			# "\n-----------------------------------------------------")


	#don't comment out
	x.append(np.abs(wigner3j[1]).max())

	# a.append(wigner3j[1][0])
	# b.append(wigner3j[1][1])
	# y.append(e)

	return

def wigner3j(l):

	file = open("ana.pkl", "rb")
	pipeline.wigner_dict_ana = pkl.load(file)

	l=np.array(l)

	gen(l)
	lmax = l.max()

	for m in range(1,lmax+1,1):
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([l,[i,j,m]])
				wigner_sol(wigner0)
				wigner0 = np.array([l,[-i,-j,-m]])
				wigner_sol(wigner0)
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([l,[m,i,j]])
				wigner_sol(wigner0)
				wigner0 = np.array([l,[-m,-i,-j]])
				wigner_sol(wigner0)
			for i in range(-1,-m,-1):
				j=-i-m
				wigner0 = np.array([l,[i,m,j]])
				wigner_sol(wigner0)
				wigner0 = np.array([l,[-i,-m,-j]])
				wigner_sol(wigner0)
			
			wigner0=np.array([l,[0,-m,m]])
			wigner_sol(wigner0)
			wigner0=np.array([l,[0,m,-m]])
			wigner_sol(wigner0)
			wigner0=np.array([l,[m,0,-m]])
			wigner_sol(wigner0)
			wigner0=np.array([l,[-m,0,m]])
			wigner_sol(wigner0)
			wigner0=np.array([l,[m,-m,0]])
			wigner_sol(wigner0)
			wigner0=np.array([l,[-m,m,0]])
			wigner_sol(wigner0)

	with open('ana.pkl', 'wb') as handle:
		pkl.dump(pipeline.wigner_dict_ana, handle, protocol=pkl.HIGHEST_PROTOCOL)
	

if __name__ == "__main__":

	start = time.time()
	t=[]
	v=[]
	L=[]
	T=[]

	known_3js=utils.load_all_keys()	
	#specify l values here
	for l in [3]:
	# for l in range(2,101):
		s=time.time()
		wigner3j([0,15,15])
		
	end = time.time()
	print(len(x))
	print("Time taken: ", end-start)
	# print(pipeline.wigner_dict_ana)
	
#create a heatmap of the errors. Ignore scale
	# heatmap(a,b,y,"simple100")
#Scatter plot of error
	# plotter(x,y,"simple","Red")

	
#Plot time against number of Wigners
# plot_time(v,t,"Time_vs_100Wigners","Blue")
# plot_wig(L,T,"Time_vs_100L","Blue")

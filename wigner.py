# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:30:16 2022

@author: ojasw
"""

import numpy as np
import pipeline


# Variables used as Globals:
recursion_count = 0	
iteration_coeffs = []
iteration_3js = []


def print_lll(wigner3j):
	l1,l2,l3 = wigner3j[0][0], wigner3j[0][1], wigner3j[0][2]
	m1,m2,m3 = wigner3j[1][0], wigner3j[1][1], wigner3j[1][2]
	print("| {}\t{}\t{}\t|\n| {}\t{}\t{}\t|\n".format(l1,l2,l3,m1,m2,m3))


###################
# RECURSION ORDERS


def recursion_coeffs(wigner3j, m_idx, sign):
	'''
	m_idx is the index of the m chosen to be lowered down
	or summed up, and sign indicates which operation
	has been performed over it (+1 or -1).
	'''
	coeffs = [0,0,0]
	k = 0
	print(m_idx, sign)
	for i in range(3):
		if i!=m_idx:
			coeffs[k] = np.sqrt((wigner3j[0][i]-sign*wigner3j[1][i]+1)*
								(wigner3j[0][i]+sign*wigner3j[1][i]))
			print("sqrt({}-{}*{}+1)({}+{}*{})".format(wigner3j[0][i],sign,wigner3j[1][i],
											       wigner3j[0][i],sign,wigner3j[1][i]))
			k=2				
		else:
			coeffs[1] = np.sqrt((wigner3j[0][i]+sign*wigner3j[1][i]+1)*
								(wigner3j[0][i]-sign*wigner3j[1][i]))
			print("sqrt({}+{}*{}+1)({}-{}*{})".format(wigner3j[0][i],sign,wigner3j[1][i],
												wigner3j[0][i],sign,wigner3j[1][i]))	
	return coeffs
	



def recursion_2(wigner3j, previous_sign=0):
	'''
	This recursion method selects one of the three m's to sum up or
	lower down one unit from. This ensures to keep constant the total
	sum of the m's. 
	It is preferred that, if in the last operation, the m was lowered
	down, the next one is summed up, alternating the sign of the added
	unit (this is being studied yet).
	
	e.g. choosing the first m to +-1:
	# (m-+1, m+-1, m), (m, m, m), (m-+1, m, m+-1)
	'''
	
	# First, it's going to try to lower the last m to 0
	if wigner3j[1][2]>0 and previous_sign>=0:
	
		# (m+1, m, m-1), (m, m, m), (m, m+1, m-1)
		
		wigner1 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0]+1,
				    wigner3j[1][1],
				    wigner3j[1][2]-1]]
				  )
						   
		wigner3 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0],
				    wigner3j[1][1]+1,
				    wigner3j[1][2]-1]]
				  )
		
		sign = +1 
		coeffs = recursion_coeffs(wigner3j, 2, sign)
		
	elif wigner3j[1][2]<0 and previous_sign<=0:
	
		# (m-1, m, m+1), (m, m, m), (m, m-1, m+1)
		
		wigner1 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0]-1,
				    wigner3j[1][1],
				    wigner3j[1][2]+1]]
				  )
						   
		wigner3 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0],
				    wigner3j[1][1]-1,
				    wigner3j[1][2]+1]]
				  )
						   
		sign = -1
		coeffs = recursion_coeffs(wigner3j, 2, sign)
		
		
	# Then it will lower the highest-valued m
	else:
	
		sign1 = np.sign(wigner3j[1][0])
		sign2 =	np.sign(wigner3j[1][1])
		
		if (abs(wigner3j[1][0]) >= abs(wigner3j[1][1]) or 
			sign1*previous_sign>=0):
			
			# (m+-1, m-+1, m), (m, m, m), (m+-1, m, m-+1)
			
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
							   
			
		elif (abs(wigner3j[1][0]) < abs(wigner3j[1][1]) or
			  sign2*previous_sign>=0):
			  
			# (m-+1, m+-1, m), (m, m, m), (m, m+-1, m-+1)
			
			sign = sign2
			
			wigner1 = np.array(
					  [wigner3j[0],
					   [wigner3j[1][0]+sign,
					    wigner3j[1][1]-sign,
					    wigner3j[1][2]]]
					  )
								
			wigner3 = np.array(
					  [wigner3j[0],
					   [wigner3j[1][0],
					    wigner3j[1][1]-sign,
					    wigner3j[1][2]+sign]]
					  )
					  
			coeffs = recursion_coeffs(wigner3j, 1, sign)
	
	return wigner1, wigner3, coeffs, -sign
	
	
	
def recursion_3(wigner3j, previous_sign=0):
	'''
	This recursion method selects one of the first two m's to sum up or
	lower down one unit from. This ensures to keep constant the total
	sum of the m's. 
	It is preferred that, if in the last operation, the m was lowered
	down, the next one is summed up, alternating the sign of the added
	unit (this is being studied yet).
	
	e.g. choosing the first m to +-1:
	# (m-+1, m+-1, m), (m, m, m), (m-+1, m, m+-1)
	'''
	
	# This method is just recursion_2 without the "last m" lowering.
	
	sign1 = np.sign(wigner3j[1][0])
	sign2 =	np.sign(wigner3j[1][1])
	sign3 =	np.sign(wigner3j[1][2])
	
	# we want to lower abs(m1), and for this to happen, we must subtract sign(m1) from it.
	# but we must recall that we are focusing on alternating operations (summing, 
	# subtracting, and so on), and for this to happen sign(m1)==previous_sign
	print((abs(wigner3j[1][0]) >= abs(wigner3j[1][1]),sign1,previous_sign))
	if (abs(wigner3j[1][0]) >= abs(wigner3j[1][1]) and
		sign1==previous_sign):
		
		# (m+-1, m-+1, m), (m, m, m), (m+-1, m, m-+1)
		
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
						   
	# if sign(m1)!=previous_sign, 
	elif (abs(wigner3j[1][0]) < abs(wigner3j[1][1]) and
		  sign2==previous_sign):
		  
		# (m-+1, m+-1, m), (m, m, m), (m, m+-1, m-+1)
		
		sign = sign2
		
		wigner1 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0]+sign,
				    wigner3j[1][1]-sign,
				    wigner3j[1][2]]]
				  )
							
		wigner3 = np.array(
				  [wigner3j[0],
				   [wigner3j[1][0],
				    wigner3j[1][1]-sign,
				    wigner3j[1][2]+sign]]
				  )
				  
		coeffs = recursion_coeffs(wigner3j, 1, sign)
		
		
	else:		
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
	
	return wigner1, wigner3, coeffs, -sign
	
	
#####################
# APPLYING RECURSION


def wigner_recursion(wigner3j, sign=0, node=0, print_recursion=True):
	'''
	print("\n")
	print_lll(l1,l2,l3,m1,m2-1,m3+1)
	print("")
	print_lll(l1,l2,l3,m1,m2,m3)
	print("")
	print_lll(l1,l2,l3,m1,m2+1,m3-1)
	'''
	
	global recursion_count, known_3js, iteration_coeffs, iteration_3js
	recursion_count+=1

	wigner1, wigner3, coeffs, sign = recursion_3(wigner3j, previous_sign=sign)
	wigners = [wigner1.tolist(),
			   wigner3j.tolist(),
			   wigner3.tolist()]
	
	if print_recursion:
		print("*=============================*")
		print("Node {} (From {})\n".format(recursion_count,node))
		print("   ({})".format(coeffs[0]))
		print_lll(wigner1)
		print(" + ({})".format(coeffs[1]))
		print_lll(wigner3j)
		print(" + ({})".format(coeffs[2]))
		print_lll(wigner3)
		print(" = 0\n")
	
	
	node = recursion_count
	
	if (wigner1.tolist() in known_3js or 
		wigner3.tolist() in known_3js or 
		wigners in iteration_3js):
		iteration_coeffs.append(coeffs)
		iteration_3js.append(wigners)	
		
	else:
		iteration_coeffs.append(coeffs)
		iteration_3js.append(wigners)
		next_wigner11, next_wigner12 = wigner_recursion(wigner1,
														sign,
														node,
														print_recursion)
		next_wigner31, next_wigner32 = wigner_recursion(wigner3,
														sign,
														node,
														print_recursion)
   
	return wigner1, wigner3


########################
# SOLVING LINEAR SYSTEMS


def forward_subs(L,b):
    y=[]
    print(L)
    print(b)
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i,j]*y[j])
        y[i]=y[i]/L[i,i]
    return y

def back_subs(U,y):
    x=np.zeros_like(y)
    for i in range(len(x),0,-1):
      x[i-1]=(y[i-1]-np.dot(U[i-1,i:],x[i:]))/U[i-1,i-1]
    return x

def solve_system_LU(L,U,b):
    y=forward_subs(L,b)
    x=back_subs(U,y)
    return x

	
	
def solve_wigner_system(iteration_3js, iteration_coeffs, print_matrices=False):
	
	global known_3js
	
	# First, we need to arrange the matrices in such a manner that the
	# system becomes C x W = V, in which C contains the coefficients,
	# W contains the wigners we need to discover, and V contains the
	# wigners whose values we already know (acting as constants).
	# Therefore, solving the system is reduced to W = C^(-1) x V.
		
	# Let's create the arrays C, W and V.
	
	C = []
	W = []
	V = []
	
	for i, wigs in enumerate(iteration_3js):
	
		any_known = False
		C.append(np.zeros(len(W)).tolist())
		
		for j, wig in enumerate(wigs):
			
			
			# ===========TO-OJASWI-2=========
			# since we are not using the sympy.physics.wigner_3j
			# anymore, the line below must be modified to get the 
			# value from the known_3js
			if wig in known_3js[0:-1]:
				any_known = True
				wig_value = pipeline.give_val_ana(*wig[0],*wig[1]) # TO-OJASWI: this line
				V.append(float(-wig_value*iteration_coeffs[i][j]))
				
			elif wig in W:
				idx = W.index(wig)
				C[i][idx] = iteration_coeffs[i][j]
				
			else:
				W.append(wig)
				new_column = np.zeros((len(C),1))
				C = np.hstack((C, new_column)).tolist()
				C[i][-1] = iteration_coeffs[i][j]
			
		if not any_known: V.append(0.)
	
	C = np.array(C)
	V = np.array(V)
	sol = np.linalg.lstsq(C,V, rcond=None)

	if print_matrices:
		print("\nCoefficients matrix:")	
		print(C)
		print("\nWigners matrix:")
		print(np.array(W))
		print("\nValues matrix:")
		print(V)
			
		rank = np.linalg.matrix_rank(C)
		n_vars = C.shape[1]
		print("\nNo. of independent equations: {} / No. of variables: {}"
			  .format(rank,n_vars))
		solving_str = {True: "Yes", False: "No"}
		print("Is the system solvable? {}".format(solving_str[rank>=n_vars]))
	
	
	# Simple Inversion Method
	if True:
		# C @ W = V
		# C.T @ C @ W = C.T @ V
		# W = inv(C.T @ C) @ C.T @ V
		sol = np.linalg.inv(C.T @ C) @ C.T @ V
		sol0 = sol

	# QR Decomposition
	elif False:
		n_eqs = len(C)
		n_3js = len(C[0])
		for i in range(n_3js,n_eqs):
			new_column = np.zeros((n_eqs,1))
			new_column[i] = 1
			C = np.hstack((C, new_column)) # making C square
			
		indep_columns = np.linalg.qr(C.T)[1].tolist()
		C_new = []
		V_new = []
		empty = np.zeros(n_eqs).tolist()
		for i in range(n_eqs):
			if indep_columns[i]!=empty:
				pass#print(i)
	
	# Least Squares Method
	if False:
		sol = np.linalg.lstsq(C,V)[0]
		sol2 = sol
	
	# SVD Method
	if False:
		b = V
		U_svd,S_svd,V_svd = np.linalg.svd(C)
		
		d = np.linalg.inv(U_svd) @ b
		i_max = len(S_svd) # S is in descending order
		
		print(np.linalg.inv(U_svd))
		print(d)
		print(S_svd)
		
		y = d[0:i_max]/S_svd[0:i_max]
		
		sol = np.linalg.inv(V_svd) @ y
		sol3 = sol
		
	
	if False:
		from scipy.linalg import lu
		P,L,U = lu(C)
		V_permu = P @ V
		sol4_permu = solve_system_LU(L,U,V_permu)
		sol4 = P.T @ sol4_permu
	

	# returns the list of wigners and its respective values
	return W, sol#sol0, sol2, sol3, sol4
		
		

def find_wigner(wigner0, print_recursion=True, 
						 print_matrices=False,
						 print_results=True):

	global recursion_count, iteration_coeffs, iteration_3js
	iteration_coeffs = []
	iteration_3js = []
	
	wigner_recursion(wigner0, print_recursion=print_recursion)	
	
	print("Number of steps: {}\n".format(recursion_count))

	Wigners, W_values = solve_wigner_system(iteration_3js, iteration_coeffs,
	#Wigners, sol0,sol2,sol3 = solve_wigner_system(iteration_3js, iteration_coeffs,
												  print_matrices)
	
	if print_results:
		print("\n****************************\n" + 
				"**********Results:**********\n" + 
				"****************************\n")
		
		
		# ===========TO-OJASWI-3=========
		# since we are not using the sympy.physics.wigner_3j
		# anymore, the line below must be modified to get the 
		# value from the known_3js
		for i, wigner in enumerate(Wigners):
			print_lll(wigner)
			w = float(pipeline.give_val_ana(*wigner[0],*wigner[1])) # TO-OJASWI: this line
			print("Calculated 0: " + str(W_values[i]) +
			#print("Calculated 0: " + str(sol0[i]) +# " / Rel. Error: " + str()
			   #"\nCalculated 2: " + str(float(sol2[i])) + 
			   #"\nCalculated 3: " + str(float(sol3[i])) + 
			   "\nCorrect: " + str(w) +
			   "\n-----------------------------------------------------")
	
	recursion_count = 0 # setting back to 0
	


if __name__=="__main__":
		
	#wigner0 = np.array([[9,10,5],[-4,7,-3]])
	#wigner0 = np.array([[9,10,5],[1,0,-1]])
	wigner0 = np.array([[120,130,140],[-4,3,1]])
	#wigner0 = np.array([[100,99,3],[-100,98,2]])
	#wigner0 = np.array([[100,99,19],[-8,4,4]])
	#wigner0 = np.array([[100,99,19],[-,1,2]])
	#wigner0 = np.array([[5,5,4],[-1,1,0]])

	print_recursion = True
	print_matrices = True
	print_results = True


	# ===========TO-OJASWI-1=========
	# the known_3js must be modified to receive the 
	# (l1,l2,l3,m1,m2,m3) and its respective value.

	known_3js = [[list(wigner0[0]),[-1,0,1]],
				 [list(wigner0[0]),[-1,1,0]],
				 [list(wigner0[0]),[0,-1,1]],
				 [list(wigner0[0]),[0,0,0]],
				 [list(wigner0[0]),[0,1,-1]],
				 [list(wigner0[0]),[1,-1,0]],
				 [list(wigner0[0]),[1,0,-1]],
				 wigner0.tolist()]
	# known_3js = [[list(wigner0[0]),[-2,0,2],pipeline.give_val_ana(wigner0[0],np.array([-2,0,2]))],
	# 			 [list(wigner0[0]),[-2,2,0],pipeline.give_val_ana(wigner0[0],np.array([-2,2,0]))],
	# 			 [list(wigner0[0]),[0,-2,2],pipeline.give_val_ana(wigner0[0],np.array([0,-2,2]))],
	# 			 [list(wigner0[0]),[0,0,0], pipeline.give_val_ana(wigner0[0],np.array([0,0,0]))],
	# 			 [list(wigner0[0]),[0,2,-2],pipeline.give_val_ana(wigner0[0],np.array([0,2,-2]))],
	# 			 [list(wigner0[0]),[2,-2,0],pipeline.give_val_ana(wigner0[0],np.array([2,-2,0]))],
	# 			 [list(wigner0[0]),[2,0,-2],pipeline.give_val_ana(wigner0[0],np.array([2,0,-2]))],
	# 			 [wigner0.tolist(), pipeline.give_val_ana(wigner0.tolist())]]
				  
				  # although wigner0 is not known yet,
				  # it is part of the stop condition.
				  


	#print("Starting point:")
	#print_lll(wigner0)

	l1,l2,l3 = 3,4,5
	#for m1 in range(-l1,l1+1):
	#	for m2 in range(-l2,l2+1):
	#		for m3 in range(-l3,l3+1):
	#			wigner0 = np.array([[l1,l2,l3],[m1,m2,m3]])
	print("Calculating recursion...\n")
	find_wigner(wigner0, print_recursion, print_matrices, print_results)

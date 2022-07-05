import numpy as np
from scipy.linalg import lu_factor, lu_solve, svd
from scipy import sparse

def solve_svd(A,b):
    U,s,Vh = svd(A)
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.conj().T,w)
    return x

def mprove(C,V):
	lu, piv = lu_factor(C)
	x = lu_solve((lu, piv), V)
	n=len(V)
	for _ in range(20):
		R=np.zeros((n))
		for i in range(n):
			R[i]=-V[i]+np.dot(C[i,:],x)
		delx=lu_solve((lu, piv), R)
		x = x-delx
	return x

seed=42
A=np.arange(8)
C=sparse.random(m=8, n=8, density=0.5, random_state=seed, format="array")
b=C@A
# print(A)
# print(C)
# print(b)
A1=A-solve_svd(C,b)
A2=A-mprove(C,b)
A3=A-np.linalg.inv(C.T @ C) @ C.T @ b
print(A1)
print(A2)
print(A3)
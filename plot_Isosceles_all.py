# encoding: utf-8
##############################################################################
## Algorithm for Countour Triangle     (Isosceles Triangule)              #####
##                                                                       #####
## 
## Authors: Karin Fornazier, Filipe Abdala
## Email: karin.fornazier@gmail.com
## Supervisor: F.B. Abdalla
## Latest Version June 2019
###########################################################################
###########################################################################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from astropy.io import fits
from math import pi, sin, cos, sqrt, log, floor
from sympy.physics.wigner import gaunt
import sys


#load file - Bispectrum code output
start = 1
end = 30    
for i in range(start, end):
    matrixf = np.loadtxt("B_ell_ell_Isosceles_Flask_21cm_Smooth_Cube_256_Teste_3A_Redshift_"+str(i)+ ".txt", delimiter=',')

#Load Bell values
    matrix_Bell = matrixf[:,0] 
    matrix_Bella = matrixf[:,0]
    matrix_Bellt=np.concatenate((matrix_Bell, matrix_Bella), axis=0)
    matrix_Bellt=matrix_Bellt.T 

#Load values ell3
    matrix_ell3 = matrixf[:,3]
    matrix_ell3a = matrixf[:,3]
    matrix_ell3t=np.concatenate((matrix_ell3, matrix_ell3a), axis=0)

#load values of ell1 and ell2. These values should be mirrored for triangle contour
    matrix_ell12 = matrixf[:,1]
    matrix_ell12a = matrixf[:,2] 
    matrix_ell12t=np.concatenate((matrix_ell12, matrix_ell12a), axis=0)

#matrix l3
#set the x dimension form matrix shape
    Y =matrix_ell3t 
    a = Y.max()
    b = np.min(Y)
    Yarr = np.arange(int(b),int(a)+1)

#matrix_ell1 = matrixell2
#set the x dimension form matrix shape
    X =matrix_ell12t
    a = X.max()
    b = np.min(X)
    Xarr = np.arange(int(b),int(a)+1)
    sizex = Xarr.size

    Bellmatrix = np.zeros((Xarr.size,Yarr.size))
    for j in np.arange(matrix_Bellt.size):
        Bellmatrix[int(matrix_ell12t[j]),int(matrix_ell3t[j])]=matrix_Bellt[j]


    cp = plt.contourf(Xarr, Yarr, Bellmatrix.T,cmap=cm.viridis )
    plt.colorbar(cp)
    plt.title("Contour plot Bispectrum Isosceles FG Teste 3A Redshift_"+ str(i))
    plt.xlabel('ell1=ell2')
    plt.ylabel('ell3')
    plt.savefig('B_ell_ell_Isosceles_Flask_21cm_Smooth_Cube_256_Teste_3A_Redshift_'+ str(i) + '.png')
    plt.clf()
plt.close()

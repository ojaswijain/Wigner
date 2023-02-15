# encoding: utf-8
##############################################################################
## Algorithm for Countour Triangle     (equisize Triangule)              #####
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


#load file and matrix - usou loadtxt para evitar problemas com int ou qq outra coisa nos numeros
#for i in range(start, end):
#        mapa = fits.getdata("Reconstructed_Cubo_Redshift_"+str(i)+".fits")# Usar para os mapas do Lucas(Foregrounds)
#        print("Reconstructed_Cubo_Redshift_",str(i))    
start =1
end = 5    
for i in range(start, end):
    matrixf = np.loadtxt("B_ell_ell_Cube_21_L0_Smooth_channel_"+str(i)+ ".txt", delimiter=',')
    
    #carregou a matriz com os valores de Bell
    matrix_Bell = matrixf[:,0] 
    matrix_Bella = matrixf[:,0]
    matrix_Bellt=np.concatenate((matrix_Bell, matrix_Bella), axis=0)
    matrix_Bellt=matrix_Bellt.T #gambi que fiz... nao tem razao logica...
    
    #carregou a matriz com os valores de l1 
    matrix_ell1 = matrixf[:,1]
    matrix_ell1a = matrixf[:,1]
    matrix_ell1t=np.concatenate((matrix_ell1, matrix_ell1a), axis=0)
    
    #carregou a matriz com os valores de l2 -l3 aqui precisa de correcao pois os valores são espelhados
    matrix_ell23 = matrixf[:,3]-matrixf[:,2]
    matrix_ell23a = matrixf[:,2]-matrixf[:,3]
    matrix_ell23t=np.concatenate((matrix_ell23, matrix_ell23a), axis=0)
    
    # matrix l1
    Y =matrix_ell1t#matrix_ell1#set the x dimension form matrix shape 
    #print Y
    
    a = Y.max()
    b = np.min(Y)
    
    Yarr = np.arange(int(b),int(a)+1)
    
    
    #matrix l2 - l3
    X =matrix_ell23t#matrix_ell1#set the x dimension form matrix shape 
    
    a = X.max()
    b = np.min(X)
    
    Xarr = np.arange(int(b),int(a)+1)
    
    sizex = Xarr.size
    
    Bellmatrix = np.zeros((Xarr.size,Yarr.size))
    
    for j in np.arange(matrix_Bellt.size):
        
        Bellmatrix[int(matrix_ell23t[j]),int(matrix_ell1t[j])]=matrix_Bellt[j]


    cp = plt.contourf(Xarr, Yarr, Bellmatrix.T,cmap=cm.inferno )
    plt.colorbar(cp)
    plt.title("Bispectrum $N_{side}=128$ Equisize GNILC Redshift_"+ str(i))
    plt.xlabel('$\ell_{2}-\ell{3}$')
    plt.ylabel('$\ell_{1}$')
    plt.set_zlim=(-6e-18,6e-18)
    plt.savefig('B_ell_ell_Cube_GNILC_128_Jordany_Redshift_'+ str(i) + '.png')
    plt.clf()
plt.close()

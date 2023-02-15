# encoding: utf-8

##############################################################################
##     Algorithm based on:                                                                                     #####
## Komatsu Thesis --> https://wwwmpa.mpa-garching.mpg.de/~komatsu/phdthesis.html                               #####
## Bucher et al  -->  https://arxiv.org/pdf/1509.08107.pdf
## Authors: Jordany Vieira de Melo and Karin Fornazier
## Email: jordanyv@gmail.com
## Supervisor: F.B. Abdalla
## Date: January 2019
###########################################################################


from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from astropy.io import fits
from math import pi, sin, cos, sqrt, log, floor
from sympy.physics.wigner import gaunt
from utils import to_str
from main import gaunt_gen
from pipeline import gaunt_oj
import pipeline
import time

###################################################################################
                                                              #
NSIDE = 64                                                    #This block is for create a thermal noise map
mu, sigma = 0, 0.1 # mean and standard deviation              #
#mapa = np.random.normal(mu, sigma, hp.nside2npix(NSIDE))      # gerar mapas de thermal noise
mapa = fits.getdata('c1.fits')# Usar para os mapas do Lucas(Foregrounds)
                                                              #
###################################################################################
                                      #
B_ell=[]                              #In this block we create the vector B_ell to save the bispectrum values for each l.
lmax=30                               #Define the lmax for the code.
alm1 = hp.map2alm(mapa[1,:])               #Use the function hp.map2alm() to get the a_lm for each map. In this case we use the same map for the 3 a_lm for a equilateral test with same map.
alm2 = hp.map2alm(mapa[1,:])               #Retirar o [1,:] para o caso de mapas thermal noise
alm3 = hp.map2alm(mapa[1,:])               #Deixar o [1,:] e modificar o número 1(dependendo do mapa que quiser usar) dentro do colchete para o trabalho com mapas de Foregrounds
sum = 0                               #The sum variable to compute the sum in bispectrum equation.
                                      #
##################################################################################
l_dic = {}
                                                                           #
def gauntB(l1, l2, l3, m1, m2, m3, p1, p2, p3):                            #This function is to calculate the total term on equation (2.3) of the paper https://arxiv.org/pdf/1509.08107.pdf
    b_ell = gaunt(l1,l2,l3,m1,m2,m3)*np.real(alm1[p1]*alm2[p2]*alm3[p3])   #The l1,l2,l3 and m1,m2,m3 refers to a_lm of the spherical harmonics.
    return b_ell                                                           #The p1,p2,p3 refers to the position in alm vector, which carries the map.
        
def gauntB_oj(l1, l2, l3, m1, m2, m3, p1, p2, p3):
    l_arr=np.array([l1, l2, l3])
    m_arr=np.array([m1, m2, m3])
    l_idx=to_str(l_arr)
    if l_idx not in l_dic.keys():
        gaunt_gen(l_arr)
        l_dic[l_idx]=1
    b_ell=gaunt_oj(l_arr,m_arr)*np.real(alm1[p1]*alm2[p2]*alm3[p3])    
    return b_ell

def alm_position(lmax, la, ma):                                            #This function use the hp.Alm.getidx() to get the position for the variables of the previous function(p1,p2,p3).
    alm_position = hp.Alm.getidx(lmax, la, ma)                             #The variables la and ma refers to the actual values of l and m in loop.
    return alm_position                                                    #
                                                                           #
####################################################################################
                                                                        #
l1 = 0                                                                  #Initial values for l1,l2 and l3.
l2 = 0                                                                  #
l3 = 0                                                                  #
 

def main(lmax):

    start = time.time()
    gaunts = 0
                                                                        #
    for l1 in range(0, lmax+1):                                             #First loop in l1, because we need to calculate the bispectrum over each l.
        l2 = l1                                                             #l2=l1 and l3=l1 because we calculate the equilateral bispectrum.
        l3 = l1 
        sum = 0                                                            #
        for m1 in range(-l1, l1+1):                                         #Now we make the loops over m's, those loops refers to calculate the bispectrum for each l's. Equation (2.3) of the paper https://arxiv.org/pdf/1509.08107.pdf
            p1 = alm_position(lmax, l1, abs(m1))                            #
            for m2 in range(-l1, l1+1):                                     #
                p2 = alm_position(lmax, l1, abs(m2))                        #
                for m3 in range(-l1, l1+1):                                 #
                    p3 = alm_position(lmax, l1, abs(m3))                    #
                    # sum += gauntB(l1, l2, l3, m1, m2, m3, p1, p2, p3)       #
                    if m1+m2+m3==0:
                        sum += gauntB_oj(l1, l2, l3, m1, m2, m3, p1, p2, p3)       #
                        gaunts += 1
                        # sum += gauntB(l1, l2, l3, m1, m2, m3, p1, p2, p3)       #
        B_ell.append(sum)                                                   #Here we use the .append() function to save the value of sum in B_ell vector.
        sum = 0   
        pipeline.gaunt_dic.clear()
        l_dic.clear()   
        
    print("lmax = ", lmax)                                                       #
    print("Number of Gaunts:", gaunts)
    end = time.time()

    print("Time taken: ", end-start, " seconds")
    print("\n")
    return lmax, end-start, gaunts
                                                            #
                                                                        #
####################################################################################
# Next block is only for plots the results.
###################################################################################

# main(30)

# bl_test = B_ell
# ell = np.arange(len(bl_test))
# #bl_testl = (ell * (ell + 1) * bl_test / (2*pi))



# plt.rcParams['figure.figsize'] = (12,8) #Normal scale plot.
#                                         #
# plt.xlabel('$\ell$')                    #
# plt.ylabel('$B_\ell$')                  #
# plt.plot(ell, bl_test, 'b.')                  #
# plt.savefig('Bispectrum_Foregrounds2.png')    #

# plt.rcParams['figure.figsize'] = (12,8) #Logaritmic scale plot.
# plt.xlabel('$\ell$')                    #
# plt.ylabel('$B_\ell$')                  #
# plt.plot(ell, bl_test, 'b.')                  #
# plt.xscale('log')                       #
# plt.savefig('Bispectrum_Foregroundslog2.png')#

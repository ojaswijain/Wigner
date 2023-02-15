# encoding: utf-8

##############################################################################
## Algorithm for Bispectrum Calculation      (Isosceles Triangule)        #####
##                                                                       #####
## Bucher et al  -->  https://arxiv.org/pdf/1509.08107.pdf
## Authors: Karin Fornazier, Filipe Abdalla, Jordany Vieira 
## Email: karin.fornazier@gmail.com
## Supervisor: F.B. Abdalla
## Latest Version June 2019
###########################################################################
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
import sys
from utils import to_str
from main import gaunt_gen
from pipeline import gaunt_oj
import pipeline
import time


###################################################################################

NSIDE = 256
nchannels=30
valorNside = hp.nside2npix(NSIDE)                                           
mu, sigma = 0, 0.1 # mean and standard deviation

l_dic = {}
gaunts=0


def anuncio(msg):
    print(msg)
    print("\n")

def gauntB(l1, l2, l3, m1, m2, m3, p1, p2, p3, alm1, alm2, alm3):
    b_ell = gaunt(l1,l2,l3,m1,m2,m3)*np.real(alm1[p1]*alm2[p2]*alm3[p3])
    return b_ell

def gauntB_oj(l1, l2, l3, m1, m2, m3):
    l_arr=np.array([l1, l2, l3])
    m_arr=np.array([m1, m2, m3])
    l_idx=to_str(l_arr)
    if l_idx not in l_dic.keys():
        gaunt_gen(l_arr)
        l_dic[l_idx]=1
    b_ell=gaunt_oj(l_arr,m_arr)   
    return b_ell

def alm_position(lmax, la, ma):
    alm_position = hp.Alm.getidx(lmax, la, ma)
    return alm_position

####################################################################################

def belltest(lmax,nrLoops,start,end):

    global gaunts

    ## Thermal Noise Maps
    #tot=int(lmax)+1
    #B_elltotal = np.zeros((nrLoops,tot))
    #for i in range(start, end):
    #    mapa = np.random.normal(mu, sigma, (nchannels,valorNside))
    #    fits.writeto('ThermalNoise'+str(i)+".fits", mapa)
    #    print("gerei mapa")

    mapa=np.zeros((30,786432))
    for i in range(start, end):
        mapa[i,:] = fits.getdata("c1_Channel_"+str(i)+".fits")
        # print("mapa")    
    ###################################################################################

        B_ell = []
        l1_ell = []
        l2_ell = []
        l3_ell = []

          
        alm1 = hp.map2alm(mapa[i,:])
        alm2 = hp.map2alm(mapa[i,:])
        alm3 = hp.map2alm(mapa[i,:])

    ##################################################################################
### Isosceles conditions lmax is defined in belltest(lmax, nrlooos, star, end)

        #l0 = 30
        l1 = 0
        l2 = 0
        l3 = 0

        # print("Entrei nos fors")
        for l1 in range(0, lmax+1):
            for l2 in range(0, lmax+1):
                for l3 in range(0, lmax+1):
                    if((l1<=l2<=l3) and (abs(l1-l2)<=l3<=abs(l1+l2)) and (l1==l2)):
                        l1_ell.append(l1)
                        l2_ell.append(l2)
                        l3_ell.append(l3)
                                                
                        sum=0           
                        pipeline.gaunt_dic.clear()
                        l_dic.clear()
                        for m1 in range(-l1, l1+1):
                            p1 = alm_position(lmax, l1, abs(m1))
                            x=alm1[p1]
                            for m2 in range(-l2, l2+1):                    
                                p2 = alm_position(lmax, l2, abs(m2))
                                y=alm2[p2]
                                for m3 in range(-l3, l3+1):                        
                                    p3 = alm_position(lmax, l3, abs(m3))
                                    z=alm3[p3]
                                    # sum =float(sum)+ gauntB(l1, l2, l3, m1, m2, m3, p1, p2, p3, alm1, alm2, alm3)
                                    if m1+m2+m3==0:
                                        gaunts+=1
                                        sum = float(sum)+ gauntB_oj(l1, l2, l3, m1, m2, m3)*np.real(x*y*z) 
                                    

                        # anuncio("Valor de saída da Gaunt:")
                        # anuncio(float(sum))
                        # anuncio("Calculado com valores -> i="+str(i)+" l1="+str(l1)+" l2="+str(l2)+" l3="+str(l3))
                        
                        B_ell.append(sum)
                        
                        # anuncio("Estado da matrix B_ell")
                        # anuncio(np.array(B_ell))
        np.savetxt('B_ell_isosceles_rec_no_noise_'+str(i)+".txt", B_ell, delimiter=',')
    #np.savetxt('B_ellRecalculado'+str(i)+".txt", B_ell, delimiter=',')
        np.savetxt('B_ell_ell_isosceles_rec_no_noise_'+str(i)+".txt", np.array([B_ell,l1_ell,l2_ell, l3_ell]).T, delimiter=',')

            
            #nstart=(i-start)
        
            
        # print("fim")

    #np.savetxt('B_ell_21Foregrounds_isosceles_2048'+str(i)+".txt", B_ell, delimiter=',')
    #np.savetxt('B_ellRecalculado'+str(i)+".txt", B_ell, delimiter=',')
    #np.savetxt('B_ell_ell_21Foregrounds_isosceles_2048'+str(i)+".txt", np.array([B_ell,l1_ell,l2_ell, l3_ell]).T, delimiter=',')

    return B_ell

####################################################################################
#belltest(lmax,nrLoops,start,end)
def iso(l):
    global gaunts
    
    gaunts=0
    start = time.time()
    belltest(l,1,0,1)
    end = time.time()
    print("lmax = ", l)
    print("Time taken:", end-start, "seconds")
    print("Number of Gaunts:", gaunts)
    print("\n")
    l_dic = {}
    return l, end-start, gaunts



# encoding: utf-8

##############################################################################
## Algorithm for save each redshift bin in a fits file                  #####
##                                                                      #####
## Authors: Karin Fornazier, Filipe Abdalla
## Email: karin.fornazier@gmail.com
## Supervisor: F.B. Abdalla
## Latest Version July 2019
###########################################################################
###########################################################################


from astropy.io import fits as pyfits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#for i in range(1,31):
#    a=i-1
#    mapas_21= pyfits.getdata('/share/data1/kfornazier/Analises/delta_flask_21_cube2048.fits')
#    print "lendo Flask-->", a
#    hp.mollview(mapas_21[a,:],title = "21cm_Redshift_"+ str(i))
    #plt.show()
#    plt.savefig('/share/data1/kfornazier/Analises/Flask_2048/delta_flask_21_cube2048_redshift_'+ str(i) + '.png')
#    pyfits.writeto('/share/data1/kfornazier/Analises/Flask_2048/delta_flask_21_cube2048_redshift_'+str(i)+ '.fits',mapas_21[a,:])


#map_total = pyfits.getdata('/share/data1/kfornazier/Analises/cube_finalmap_delta_foregrounds2048.fits')
#for i in range(1,31):
#    a=i-1
#    hp.mollview(map_total[a,:],title="Foregrounds + 21 cm Redshift ="+ str(i))
    #plt.show()
 #   print "lendo Cubo Total-->", a
#    plt.savefig('/share/data1/kfornazier/Analises/Total_2048/Foregrounds_21cm_cubo_Redshift_'+ str(i) + '.png')
#    pyfits.writeto('/share/data1/kfornazier/Analises/Total_2048/Foregrounds_21cm_cubo_Redshift_'+str(i)+ '.fits',map_total[a,:])


#foregrounds = pyfits.getdata('/share/karin/Desktop/gnilc/Cubo_Flask_crescente/foreground_cube_Nside256.fits')
#for i in range(1,31):
#    a=i-1
#    hp.mollview(foregrounds[a,:],title="Foregrounds Cube Redshift_"+ str(i))
    #plt.show()
#    print "lendo Foregrounds-->", a
#    plt.savefig('/home/karin/Desktop/gnilc/Cubo_Flask_crescente/Redshifts_bins/Foregrounds_Redshift/Foregrounds_Cubo_Redshift_'+ str(i) + '.png')
#    pyfits.writeto('/home/karin/Desktop/gnilc/Cubo_Flask_crescente/Redshifts_bins/Foregrounds_Redshift/Foregrounds_Cubo_Redshift_'+str(i)+ '.fits', foregrounds[a,:])

cubo_reconstructed = pyfits.getdata('c1.fits')
for i in range(0,30):
    a=i
    hp.mollview(cubo_reconstructed[a,:],title="Gnilc no noise Channel = "+ str(i))
    print("lendo--> Recontructed Map Gnilc ", a)
    plt.savefig('c1_Channel_'+ str(i) + '.png')
    pyfits.writeto('c1_Channel_'+str(i)+ '.fits', cubo_reconstructed[a,:])

    

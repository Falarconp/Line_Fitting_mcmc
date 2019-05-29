"""
Code created by Christian Flores
"""

#CAMBIAR PATHS, PA and Disk inclination
import numpy as np
import astropy.io.fits as pf
import math
import sys
import astropy.units as u
import astropy.constants as const
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#
#Global Variables
#
#-----------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#
#
#-------------------------------------------------S----------------------------
#incli = 152.0*math.pi/180.0					#Disk inclination
incli = 0.


PA = 24.00*math.pi/180.0

# constants in cgs units

h = const.h.cgs.value   #Planck Constant
c = const.c.cgs.value   #Speed of light
k = const.k_B.cgs.value  # Boltzmann Constant

rest_freq_C18O = 329330580193.9937				#C18O rest frequency
rest_freq_12CO = 345796018978.6035					#12CO rest frequency
rest_freq_13CO = 330587993021.5317				#13CO rest frequency

A_C18 = 2.172e-06						#Einstein Coefficients in s^-1
A_C13 = 2.181E-06
A_C12 = 2.497e-06




###### Transition 2-1
## A_C18=6.266e-08						#Einstein Coefficients in s^-1
## A_C13=6.294e-08
## A_C12=7-203e-08
####



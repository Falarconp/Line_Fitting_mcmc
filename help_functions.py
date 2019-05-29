"""
Code created by Christian Flores
"""
#Cambiar path of functions , r_size t_size(resolucion en radio y angulo)

from constants import*

import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import glob
import os
import math
import scipy.ndimage.interpolation as interpol
import scipy.ndimage.interpolation as maps


#------------------------------------------------------------------------------------------------
#
# transforming cartesian image into a polar one
#
#------------------------------------------------------------------------------------------------
def cartesian_to_polar_spline(image,r_size,t_size,orden):
	new_x_axis=len(image[0,:])
	new_y_axis=len(image[:,0])
	Y=np.empty(r_size*t_size)
	X=np.empty(t_size*r_size)
	radius=np.linspace(0,new_x_axis/2.0,r_size)
	theta=np.linspace(0,2*math.pi,t_size)
	i=0
	for r in radius:
		for t in theta:
			X[i]=-r*math.sin(t)
			Y[i]=r*math.cos(t)
			i=i+1
	X=X+new_x_axis/2
	Y=Y+new_x_axis/2
	return maps.map_coordinates(image,[Y,X],order=int(orden)).reshape(r_size,t_size)

#------------------------------------------------------------------------------------------------
#
# Cutting the image to get the center of the disk
#
#------------------------------------------------------------------------------------------------

def cut_image(image,x_center,y_center,size):
	y_min=y_center-int(size/2.0)
	y_max=y_center+int(size/2.0)
	x_min=x_center-int(size/2.0)
	x_max=x_center+int(size/2.0)
	return image[y_min:y_max,x_min:x_max]

#------------------------------------------------------------------------------------------------
#
# routine to get the global maximum of the emission at each radius
#
#------------------------------------------------------------------------------------------------

def getting_maximum(image,theta_ini,theta_fin,r_ini,r_fin,delta_r):
	lim_rad=len(image[:,0])
	lim_theta=len(image[0,:])
	nt_ini=int(theta_ini*lim_theta/360.0)
	nt_fin=int(theta_fin*lim_theta/360.0)
	nr_ini=int(r_ini*lim_rad)
	nr_fin=int(r_fin*lim_rad)
	rango=nt_fin-nt_ini
	r_max_pos=[]
	theta_max_pos=[]
	for i in range(rango):
		nr_ini=nr_ini+delta_r*i
		r_max_pos=np.append(r_max_pos,meas.maximum_position(image[nr_ini:nr_fin,nt_ini+i]))
		theta_max_pos=np.append(theta_max_pos,nt_ini+i)
	return (theta_max_pos,nr_ini+r_max_pos)


#------------------------------------------------------------------------------------------------
#
# routines to get the velocity from a given frequency and viceversa
#
#------------------------------------------------------------------------------------------------
def single_vel(frequen,nu_cero):
	return ( (frequen-nu_cero)*c )/nu_cero

def single_frequency(vel,nu_cero):
	return (vel*nu_cero/c)+nu_cero

#------------------------------------------------------------------------------------------------
#
#getting the parameters from data cubes
#
#------------------------------------------------------------------------------------------------

def temperature_cube(path):
	path_temp='simulated_cubes/'+path
	hdulist_temp=pf.open(path_temp)
	return hdulist_temp[0].data[0,:,:]

def turbulent_cube(path):
	path_turb='simulated_cubes/'+path
	hdulist_turb=pf.open(path_turb)
	return hdulist_turb[0].data[3,:,:]

def vel_centroid_cube(path):
	path_centroid='simulated_cubes/'+path
	hdulist_centroid=pf.open(path_centroid)
	return hdulist_centroid[0].data[4,:,:]

def centroid_cube(path):
	path_centroid='simulated_cubes/'+path
	hdulist_centroid=pf.open(path_centroid)
	return hdulist_centroid[0].data[5,:,:]


def continuum_cube(path):
	path_centroid='simulated_cubes/'+path
	hdulist_centroid=pf.open(path_centroid)
	return hdulist_centroid[0].data[1,:,:]

def nco_cube(path):
	path_nco='simulated_cubes/'+path
	hdulist_nco=pf.open(path_nco)
	return hdulist_nco[0].data[2,:,:]

def angle_cube(path):
	path_nco='simulated_cubes/'+path
	hdulist_nco=pf.open(path_nco)
	return hdulist_nco[0].data[6,:,:]


#------------------------------------------------------------------------------------------------
#
#Simpson 1/3 method
#
#------------------------------------------------------------------------------------------------


def simpson_1_3(funcion,intervalo):
	f=funcion
	a=intervalo[0]
	b=intervalo[-1]
	N=len(intervalo)
	h=(b-a)/N
	suma_par=0
	suma_impar=0
	nu=N/2
	xk=0
	for i in range(nu-1):
		xk=xk+1
		suma_impar=suma_impar+funcion[xk]
		xk=xk+1
		suma_par=suma_par+funcion[xk]
	suma_impar=suma_impar+funcion[xk+1]
	suma_par=2.0*suma_par+funcion[0]+funcion[-1]
	return (4.0*suma_impar+suma_par)*h/3.0


#------------------------------------------------------------------------------------------------
#
#Radial average
#
#------------------------------------------------------------------------------------------------

def radial_average(imagen,delta_phi,r_ini,r_fin):
	cut_imagen=cut_image(imagen,216,216,200)
	new_imagen=cartesian_to_polar_spline(image=cut_imagen,r_size=200,t_size=600,orden=2)
	r_size=200
	t_size=600
	delta_r=0.05*200.0/(r_size*2) 			#pixel size in arcsec * image size / reshape
	r_i=int(r_ini/delta_r)				# from arcsec to pixel
	r_f=int(r_fin/delta_r)
	delta_p=360.0/t_size
	delta_azimuth=int(delta_phi/delta_p)		#deg to pixel
	numero_zonas=int(360.0/delta_phi)
	average=np.empty(numero_zonas)
	#plt.pcolor(new_imagen)
	#plt.show()
	for k in range(numero_zonas):
		suma=0
		cont=0
		for i in range(r_f-r_i):
			for j in range(delta_azimuth):
				suma=suma+new_imagen[r_i+i,delta_azimuth*k+j]
				cont=cont+1
		average[k]=suma/cont

	return average

def deprojected_image(image):
	x_axis=len(image[0,:])
	y_axis=len(image[:,0])
	Y=np.empty([y_axis,x_axis])
	X=np.empty([y_axis,x_axis])


	radius=np.linspace(0,new_x_axis/2.0,r_size)
	theta=np.linspace(0,2*math.pi,t_size)
	i=0
	for r in radius:
		for t in theta:
			X[i]=-r*math.sin(t)
			Y[i]=r*math.cos(t)
			i=i+1
	X=X+new_x_axis/2
	Y=Y+new_x_axis/2
	return maps.map_coordinates(image,[Y,X],order=int(orden)).reshape(r_size,t_size)



def radial_cut(imagen,angle,r_ini,r_fin):
	cut_imagen=cut_image(imagen,216,216,200)
	new_imagen=cartesian_to_polar_spline(image=cut_imagen,r_size=200,t_size=600,orden=2)
	r_size=200
	t_size=600
	delta_r=0.05*200.0/(r_size*2) 			#pixel size in arcsec * image size / reshape
	r_i=int(r_ini/delta_r)				# from arcsec to pixel
	r_f=int(r_fin/delta_r)
	delta_p=360.0/t_size
	angulo=angle/delta_p					#deg to pixel
	return new_imagen[r_i:r_f,angulo]

"""
This code provides the functions that are necessary in order to get the
gas density structure, turbulent velocity and temperature profiles
of a protoplanetary disk.

By F. Alarcon
Minimize chi2

The main parameters are in lines 267-280.
For fixing variables change fix par in Minuit method in line 388 or 425 (if cont is true or false)
Output path in line 515
"""

import collision_cross_section	as ccs				# auxiliar quantities
import help_functions as hf					# finding maximum
from constants import *

from multiprocessing import Pool
import numpy as np
import scipy as sp
from scipy.integrate import quad
import pyfits as pf
import matplotlib.pyplot as plt
import os
import math
import pymc3 as pm
import sys
from astropy.convolution import Gaussian2DKernel, convolve_fft
import astropy.units as u
import astropy.constants as const
from astropy.analytic_functions import blackbody_nu

# constants in cgs units
h = const.h.cgs.value
c = const.c.cgs.value
k = const.k_B.cgs.value

incli = 28.*sp.pi/180.


def datafits(namefile):
    """
    Open a FITS image and return datacube and header, namefile without '.fits'
    """
    datacube = pf.open(namefile + '.fits')[0].data
    hdr = pf.open(namefile + '.fits')[0].header
    return datacube, hdr


def bbody(T,nu):
    """
    Blackbody flux for a given temperature and frequency erg / (cm2 Hz s sr) (cgs system)
    """
    return blackbody_nu(nu, T).cgs.value


def phi(t,nu,nu0,vturb,angle, m):
    """
    Returns the normalized line profile.
    t: Temperature.
    nu: Array of frecuencies to sample teh line profile.
    nu0: Center of line emission.
    vturb: Turbulent velocity or dispersion velocity along the line of sight(cgs system).
    m: Molecule mass.
    """
    #pa = (-20+90+angle)*math.pi/180.0  #-20 is the PA from east of north, and 90 to get the semi-minor axis
    #Q = 1
    #shear = (Q*math.sin(incli)**2)*(math.tan(incli)**2)*math.sin(2*pa)**2
    #deltav = (nu0/c)*math.sqrt(k*t/m + vturb**2 )*(1+shear)**(0.5) 
    #deltav = (nu0/c)*math.sqrt(2*k*t/m + vturb**2 )
    #phi0 = deltav*math.sqrt(math.pi)
    #gauss=sp.exp(-((nu-nu0)**2.0)/(deltav**2.0))
 
    deltav = (nu0/c)*math.sqrt(k*t/m + vturb**2 )
    phi0 = deltav*math.sqrt(2*math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    gauss /= phi0
    return gauss


def tau(t,nu,nu0,nco,vturb,angle,iso=12):
    """
    Returns the optical depth for a certain CO isotopologue.
    t: Temperature.
    nu: Array of frecuencies to sample the line profile.
    nu0: Center of line emission.
    nco: Column density for excited state
    vturb: Turbulent velocity or dispersion velocity along the line of sight(cgs system).
    iso: Isotopologue Number.
    """
    sigma = ccs.cross_section("13CO")
    m = ccs.cross_section("13CO")
    if iso==12:
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
    return (sp.exp(h*nu/(k*t))-1)*nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(incli)
    #return nco*sigma*phi(t,nu,nu0,vturb,angle, m)/math.cos(incli)


def intensity(nu, T, nu0, nco, vturb, angle, hdr, iso=12):
    """
        Solution to radiative transfer equation
        """
    if iso==13:
        nco/= 70.
    if iso==18:
        nco/=500.
    pix_deg = abs(hdr['CDELT1'])
    pix_rad = pix_deg*sp.pi/180.
    bminaxis =  pix_deg
    bmajaxis = pix_deg
    scaleaux = (u.erg/(u.s*u.cm*u.Hz*u.cm*u.sr)).to(u.Jy/(u.deg*u.deg)) #cgs to JY/deg^2
    scaling = scaleaux*pix_deg**2   ##cgs to Jy/pix^2
    bbody = 2*h*nu**3/(c**2*(sp.exp(h*nu/(k*T))-1.))
    sigma = ccs.cross_section("13CO")
    m = ccs.cross_section("13CO")
    if iso==12:
        sigma = ccs.cross_section("12CO")
        m = ccs.molecular_mass("12CO")
    elif iso==13:
        sigma = ccs.cross_section("13CO")
        m = ccs.molecular_mass("13CO")
    elif iso==18:
        sigma = ccs.cross_section("C18O")
        m = ccs.molecular_mass("C18O")
    #deltav = (nu0/c)*math.sqrt(k*T/m + vturb**2 )
    deltav = (nu0/c)*vturb
    phi0 = deltav*sp.sqrt(2*math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    gauss /= phi0
    phi = gauss
    opt_depth = (sp.exp(h*nu/(k*T))-1)*nco*sigma*phi/math.cos(incli)
    blackbody = bbody*(1.0-sp.exp(-opt_depth))*scaling
    return  blackbody


def intensity_continuum(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr,iso=12):
    """
        Solution to radiative transfer equation with continuum
        """
    pix_deg = abs(hdr['CDELT1'])
    pix_rad = pix_deg*sp.pi/180.
    bminaxis =  pix_deg  #arcsec
    bmajaxis = pix_deg  #arcsec,
    scaleaux = (u.erg/(u.s*u.cm*u.Hz*u.cm*u.sr)).to(u.Jy/(u.deg*u.deg)) #cgs to JY/deg^2
    scaling = scaleaux*pix_deg**2
    if iso==13:
        nco/= 80.
    if iso==18:
        nco/=500.
    deltav = (nu0/c)*vturb
    phi0 = deltav*sp.sqrt(2*math.pi)
    gauss=sp.exp(-((nu-nu0)**2.0)/(2*(deltav**2.0)))
    gauss /= phi0
    phi = gauss
    opt_depth = (sp.exp(h*nu/(k*T))-1)*nco*sigma*phi/math.cos(incli)
    blackbody = bbody*(1.0-sp.exp(-opt_depth))*scaling
    cont = i0*sp.exp(-opt_depth)*(nu/nu0)**alpha
    return  blackbody + cont


def intensity_dust_err(nu, nu0, T, alpha, i0, nco, vturb, angle, datos, hdr,iso=12):
    """
    Chi squared of data with fit of spectral line, considering continuum absorption from midplane.
    """
    model, tau0, taus = intensity_continuum(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, iso)
    aux = (datos-model)**2
    median = sp.median(datos)
    chi = sp.sum(aux/median**2)
    return chi*1e3

def partition(inputfile):
    """
    Return arrays with energy and weights of molecule from inputfile.
    inputfile: file with data for molecule.
    """
    level = sp.arange(1,42)
    energies = sp.zeros(41)
    gs = sp.zeros(41)
    
    arch = open(inputfile,'r')
    
    arch.readline()
    arch.readline()
    arch.readline()
    arch.readline()
    arch.readline()
    arch.readline()
    arch.readline()
    
    for i in range(41):
        l = arch.readline().split()
        level[i] = int(l[0])
        energies[i] = float(l[1])
        gs[i] = float(l[2])
    
    energies *= 0.000123986  #cm^-1 to eV
    energies *= u.eV.to(u.erg)  #eV to erg

    return energies, gs


def Part(energies, gs,T):
    """
    Returns Q
    T: Temperature
    """
    return sp.sum(gs*sp.exp(-energies/(k*T)))


def intensity_dust_err(nu, nu0, T, alpha, i0, nco, vturb, angle, datos, hdr,rms,iso=12):
    """
    Chi squared of data with fit of spectral line, considering continuum absorption from midplane.
    """
    model, tau0, taus = intensity_continuum(nu, T, nu0, alpha, nco, vturb, angle, i0, hdr, iso)
    aux = (datos-model)**2
    median = sp.median(datos)
    #chi = sp.sum(aux/median**2)
    chi = sp.sum(aux)/rms**2
    return chi#*1e3


def intensity_err(nu, nu0, T, nco, vturb, angle, datos, hdr,rms, iso=12):
    """
    Chi squared of data with fit of spectral line, both normalized.
    """
    model ,tau0, taus = intensity(nu, T, nu0, nco, vturb, angle, hdr, iso)
    aux = (datos-model)**2
    median = sp.median(datos)
    chi = sp.sum(aux/median**2)
    chi = sp.sum(aux)/rms**2
    return chi


def Convolv(cubo, head, Beam=0.25):
    resol = abs(head['CDELT1'])*3600
    stdev = Beam / (2 * sp.sqrt (2 * sp.log(2)))
    stdev /= resol
    x_size = int(8*stdev + 1.)

    print 'convolution with gaussian'
    print '\tbeam '+str(Beam)+' arcsec'
    print '\tbeam '+str(stdev)+' pixels'

    # circular Gaussian
    beam = Gaussian2DKernel (stddev = stdev, x_size = x_size, y_size = x_size,
                             model ='integrate')
    smooth =  np.zeros((256, 256))
    smooth += convolve_fft(cubo[:,:], beam)
    print '\tsmoothed'
    return smooth

"""
Fits a temperature profile, turbulent velocity and column density
using three CO isotopologues lines with iminuit package.
"""
cont = False
r = 50   # radius in pixels units where to make the fit
# CO isotopologue, it could be 12,13,18
iso = 13
Convolve = False
Beam = 0.25                                                         #Only used if Convolve=True, size of beam in arcsec.

tag_nco = './test_fo/'                                              #path with molecule input files.
g_u = 7                                                             #Degeneracy of state
Z = tag_nco + 'molecule_13c16o.inp'                                 #File with info of molecular transitions.
en, gs = partition(Z) 
incli = 0.*sp.pi/180.                                               #Inclination of disk in degrees.
path = './fo_pardisk/'                                                # Path to FITS files to fit.
image = path + 'image_13co32_nodust__o0_i180.00_phi90.00_PA-24.00'  #Input fits File without the '.fits'.
ncores = 10                                                         #Number of cores for parallel running.

print('Opening FITS images and fitting functions')

#  Isotopologue image and centroid map
cubo, head = datafits(image)

dnu = head['CDELT3']
len_nu = head['NAXIS3']
nui = head['CRVAL3']- head['CRPIX3']*dnu
nuf = nui + (len_nu-1)*dnu

nu = sp.linspace(nui, nuf, len_nu)
nu0 = sp.mean(nu)

#Gaussian Convolution
if False:
    resol = abs(head['CDELT1'])*3600
    stdev = Beam / (2 * sp.sqrt (2 * sp.log(2)))
    stdev /= resol
    x_size = int(8*stdev + 1.)

    print 'convolution with gaussian'
    print '\tbeam '+str(Beam)+' arcsec'
    print '\tbeam '+str(stdev)+' pixels'

    # circular Gaussian
    beam = Gaussian2DKernel (stddev = stdev, x_size = x_size, y_size = x_size,
                             model ='integrate')
    smooth =  np.zeros((80, 256, 256))
    for k in range(80):
        smooth[k, :,:] += convolve_fft(cubo[k,:,:], beam)
    print '\tsmoothed'
    cubo = smooth

cubomax = cubo.max()

stderr = sp.std(cubo)
alpha = 2.3
i0 = 0.0   # continuum guess
angle = 24.*sp.pi/180.   # PA, not used for fitting.
ndim = head['NAXIS1']
Temperature = sp.zeros((ndim,ndim,6))
tau0 = sp.zeros((ndim,ndim))
Denscol = sp.zeros((ndim,ndim,6))
Turbvel = sp.zeros((ndim,ndim,6))
vel_cen = sp.zeros((ndim,ndim,6))
dust = sp.zeros((ndim,ndim,cubo.shape[0]))
model = sp.zeros((ndim,ndim,cubo.shape[0]))
mom2 = sp.zeros((ndim,ndim))
velocities = sp.linspace(sp.around(((nu0-nui)*c*1e-5/nu0),1),sp.around((nu0-nuf)*c*1e-5/nu0),len_nu)

def parspar(n):
    i = n/ndim
    j = n%ndim
    data = cubo[:,i,j]
    datamax = data.max()
    noise_level = 0
    noise = sp.random.normal(scale=noise_level,size=len(data))
    data = data+ruido
    if (i-int(ndim/2.))**2 + (j-int(ndim/2.))**2 > r**2 or 0.4*cubomax>datamax:
        return [None]

    print i,j

    m0 = sp.integrate.simps(data,nu)
    m1 = sp.integrate.simps(velocities*data, nu)/m0
    mom2[i,j] = sp.integrate.simps(data*(velocities-m1)**2, nu)*1000/m0


    datamin = data.min()
    data1 = data -data.min()
    centroid = (nu*(cubo[:,i,j]-datamin)).sum()/(cubo[:,i,j]-datamin).sum()

    i0 = datamin
    if i0==0:
        i0=1e-10
    i0_lim=(0.5*i0,1.2*i0)


    r_ij = sp.sqrt((i-128)**2 +(j-128)**2)
    rho_ij = r_ij/sp.cos(incli)

    if rho_ij<30:
        temp_0 = 70
        tlim = (10,200)
    else:
        temp_0 = 70 * (r_ij / 30.)**(-0.5)
        tlim = (10,120)
    vels = (nu-nu0)*3e5/nu0
    velg = vels[data==data.max()][0]

    noise = data[(velocities<velg-1) | (velocities>velg+1.)]
    rms = np.sqrt(sp.sum(noise**2)/float(len(noise)))

## Cont=True Fit lines considering the presence of continuum.
    if cont:
        line_model = pm.Model()
    
        with line_model:
        
            var = ['Temp','nu_c','log(N_CO)','v_turb','Continuum']

# Priors for unknown model parameters
            Temp = pm.TruncatedNormal('Temp', mu=temp_0, sd=5, lower=tlim[0], upper=tlim[1])
            nu_c = pm.TruncatedNormal('nu_c', mu=centroid, sd=abs(dnu)/10., lower=centroid-0.5*abs(dnu), upper=centroid+0.5*abs(dnu))
            NCO = pm.Uniform('log(N_CO)', lower=10, upper=24)
            v_turb = pm.Uniform('v_turb', lower=sp.sqrt(k*tlim[0]/m), upper=300000)  ## This is really broadening in velocity space, not turbulent vel.
            i_0 = pm.TruncatedNormal('Continuum', mu=i0, sd=5, lower=i0_lim[0], upper=i0_lim[1])
    
    # Expected value of outcome
            predict = intensity_continuum(nu, Temp, nu_c, alpha, 10**NCO, v_turb, angle, i_0, head, iso)
    
    # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=predict, sd=rms, observed=data)
            step = pm.NUTS()
    
            st = {'Temp':temp_0,
                'nu_c':centroid,
                'log(N_CO)':20,
                'v_turb':20000,
                'Continuum':i0}

            trace = pm.sample(5000,tune=1000,cores=2,step=step,start=st)

            stats = pm.summary(trace)

            mean_pars = [stats['mean'][x] for x in var]
            hpd_2_5 = [stats['hpd_2.5'][x] for x in var]
            hpd_97_5 = [stats['hpd_2.5'][x] for x in var]
            var_std = [stats['sd'][x] for x in var]
            medians_pars = [sp.median(trace[x]) for x in var]
            Map = [float(pm.find_MAP(model=line_model)[x]) for x in var]
            fit = mean_pars
            model = intensity_continuum(nu, fit[0], fit[1],alpha, 10**fit[2], fit[3], angle,fit[4], head, iso)

    else:
        line_model = pm.Model()
    
        with line_model:
        
            var = ['Temp','nu_c','log(N_CO)','v_turb']
            
            # Priors for unknown model parameters
            Temp = pm.TruncatedNormal('Temp', mu=temp_0, sd=5, lower=tlim[0], upper=tlim[1])
            nu_c = pm.TruncatedNormal('nu_c', mu=centroid, sd=abs(dnu)/10., lower=centroid-0.5*abs(dnu), upper=centroid+0.5*abs(dnu))
            NCO = pm.Uniform('log(N_CO)', lower=10, upper=24)
            v_turb = pm.Uniform('v_turb', lower=sp.sqrt(k*tlim[0]/m), upper=300000) ## This is really broadening in velocity space, not turbulent vel.
            
            # Expected value of outcome
            predict = intensity(nu, Temp, nu_c, 10**NCO, v_turb, angle, head, iso)
            
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=predict, sd=rms, observed=data)
            step = pm.NUTS()
            
            st = {'Temp':temp_0,
                'nu_c':centroid,
                'log(N_CO)':20,
                'v_turb':20000}
        
            trace = pm.sample(5000,tune=1000,cores=2,step=step,start=st)
            
            stats = pm.summary(trace)
            
            mean_pars = [stats['mean'][x] for x in var]
            hpd_2_5 = [stats['hpd_2.5'][x] for x in var]
            hpd_97_5 = [stats['hpd_2.5'][x] for x in var]
            var_std = [stats['sd'][x] for x in var]
            medians_pars = [sp.median(trace[x]) for x in var]
            Map = [float(pm.find_MAP(model=line_model)[x]) for x in var]
            fit = mean_pars
            model = intensity(nu, fit[0], fit[1], 10**fit[2], fit[3], angle, head, iso)

    Temperature[i,j,:] = sp.array([fit[0],var_std[0],medians_pars[0],Map[0],hpd_2_5[0],hpd_2_5[0]])
    Denscol[i,j,:] = sp.array([10**fit[2],10**fit[2]*sp.log(10)*var_std[2],10**medians_pars[2],10**Map[2],10**hpd_2_5[2],10**hpd_2_5[2]])
    Turbvel[i,j,:] = sp.sqrt(((sp.array([fit[3],var_std[3],medians_pars[3],Map[3],hpd_2_5[3],hpd_2_5[3]]))**2 - k*fit[0]/m))*1e-5
    aux_nu = sp.array([fit[1],var_std[1],medians_pars[1],Map[1],hpd_2_5[1],hpd_2_5[1]])
    vel_cen[i,j,:] = sp.around(((nu0-aux_nu)*c*1e-5/nu0))
    return [i,j,Temperature[i,j,:], Denscol[i,j,:], Turbvel[i,j,:],vel_cen[i,j,:]]



p = Pool(ncores)
todo = p.map(parspar, range(cubo[0,:,:].size))
todo = sp.array(todo)
#todo = todo[todo!=None]
for ls in todo:
    if len(ls)!=6:
        continue
    i = ls[0]
    j = ls[1]
    Temperature[i,j,:] = ls[2]
    Denscol[i,j,:] = ls[3]
    Turbvel[i,j,:] = ls[4]
    vel_cen[i,j,:] = ls[5]


# Parameter error (confidence intervals)
#errmodel= sp.array(errmodel)
#model = sp.array(model)
#model = sp.swapaxes(model, 0,1)
#model = sp.swapaxes(model,0,2)
print('Calculated Fits')
#model = sp.nan_to_num(model)
#dust = sp.nan_to_num(dust)
Denscol = sp.nan_to_num(Denscol)
Turbvel = sp.nan_to_num(Turbvel)
Temperature = sp.nan_to_num(Temperature)
vel_cen = sp.nan_to_num(vel_cen)
#tau0 = sp.nan_to_num(tau0)

if Convolve:
    Denscol = Convolv(Denscol, head)
    Temperature = Convolv(Temperature, head)
    Turbvel = Convolv(Turbvel, head)
    mom2 = Convolv(mom2, head)

#r1 = pf.PrimaryHDU(model)
r2 = pf.PrimaryHDU(Temperature)
r3 = pf.PrimaryHDU(Turbvel)
r4 = pf.PrimaryHDU(Denscol)
#r5 = pf.PrimaryHDU(errmodel)
#r6 = pf.PrimaryHDU(tau0)
r9 = pf.PrimaryHDU(mom2)
#r10 = pf.PrimaryHDU(dust)
head1 = head
head2 = head
head3 = head
head4 = head
#r1.header = head
r2.header = head1;
head1['BUNIT'] = 'K'
head2['BUNIT'] = 'cm/s'
head2['BTYPE'] = 'Velocity'
head3['BUNIT'] = 'm-2'
head3['BTYPE'] = 'Column Density'
head3['BUNIT'] = 'cm'
head4['BTYPE'] = 'Optical Depth'
r3.header = head2
r4.header = head3
#r5.header = head
#r6.header = head4
r9.header = head2
#r10.header = head

## Path where FITS files are saved.
inputimage = './fo_pardisk/outs/i180/sims_' + str(iso)
out1 = inputimage + '_model.fits'
out2 = inputimage + '_Temp.fits'
out3 = inputimage + '_v_turb.fits'
out4 = inputimage + '_NCO.fits'
out6 = inputimage + '_tau_nu0.fits'
out5 = inputimage + '_errfit.fits'
out9 = inputimage + '_mom2.fits'
out10 = inputimage + '_dust.fits'
print('Writing images')
#r1.writeto(out1, clobber=True)
r2.writeto(out2, clobber=True)
r3.writeto(out3, clobber=True)
r4.writeto(out4, clobber=True)
#r5.writeto(out5, clobber=True)
#r6.writeto(out6, clobber=True)
r9.writeto(out9, clobber=True)
#r10.writeto(out10, clobber=True)
#pf.writeto(out1, model, head, clobber=True)
pf.writeto(out2, Temperature, head1, clobber=True)
pf.writeto(out3, Turbvel, head2, clobber=True)
pf.writeto(out4, Denscol, head3, clobber=True)
#pf.writeto(out5, errmodel, head, clobber=True)
pf.writeto(out9, mom2, head2, clobber=True)
#pf.writeto(out6, tau0, head4, clobber=True)
#pf.writeto(out10, dust, head, clobber=True)





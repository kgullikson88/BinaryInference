"""
Several functions for making and interacting
with the orbit prior and completeness functions
"""
from __future__ import print_function, absolute_import, division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import SpectralTypeRelations
import pandas as pd
import Mamajek_Table
import Feiden
from Feiden import Feiden_Isochrone
import scipy.interpolate
import logging
import StarData




MS = SpectralTypeRelations.MainSequence()
MT = Mamajek_Table.MamajekTable()
sptnum2mass = MT.get_interpolator('SpTNum', 'Msun')

# Read in the David & Hillenbrand summary
dh2015 = pd.read_csv('/home/kgullikson/Dropbox/School/Research/AstarStuff/TargetLists/David_and_Hillenbrand2015/dh2015-table5.csv')

def read_dh2015_posterior(hipnum,
                          basedir='/home/kgullikson/Dropbox/School/Research/AstarStuff/TargetLists/David_and_Hillenbrand2015',
                          var='mass'):
    fname = os.path.join(basedir, 'HIP{}-{}.txt'.format(hipnum, var))
    if not os.path.exists(fname):
        raise OSError('Filename {} does not exist!'.format(fname))
    return np.loadtxt(fname, unpack=True)

def get_cdf(x, P, accuracy=10):
    cdf = np.zeros_like(x)
    dx = np.diff(x)
    cdf[1:] = np.cumsum(dx * 0.5*(P[1:] + P[:-1]))
    return np.round(cdf/cdf[-1], decimals=accuracy)


def get_padova_posterior(star, hdf5_filename='Primary_Parameters.h5', var='mass'):
    """ Search the given HDF5 file for an entry for star. If it exists, return the mass samples.
    """
    try:
        df = pd.read_hdf(hdf5_filename, star.replace(' ', '_'))
        return df[var].values
    except KeyError:
        return None    
    

def get_primary_mass(star, spt, size=1e4, mass_err=0.1):
    """
    Get 'size' samples from the primary star mass. Try in the following order:
    1. David & Hillenbrand sample
    2. Use mass from the evolutionary grid fits I did. Uses teff/logg from primary star ccfs.
    3. Spectral type
    
    Parameters:
    ==========
    - star:     String
                The name of the star
                
    - spt:      String
                The spectral type of the star
                
    - size:     integer
                The number of samples to take
                
    - mass_err: float
                The relative error to assume for the mass.
                Only used if the star is not found in the DH2015 sample
    
    Returns:
    ========
    Samples of the primary star mass
    """
    # Is this star in the david & hillenbrand sample?
    if star.startswith('HIP'):
        hipnum = star.split('HIP')[-1].strip()
        if int(hipnum) in dh2015.HIP.values:
            # Get the cdf in discrete steps
            mass, P = read_dh2015_posterior(hipnum=hipnum, var='mass')
            if not np.any(np.isnan(P)):
                cdf = get_cdf(mass, P)
                
                # Remove duplicate cdf values
                df = pd.DataFrame(data=dict(cdf=cdf, mass=mass))
                df.drop_duplicates(subset=['cdf'], inplace=True)
                
                # Calculate the inverse cdf
                inv_cdf = spline(df.cdf.values, df.mass.values, k=1)
                
                mass_samples = inv_cdf(np.random.uniform(size=size))
                return mass_samples
    
    # Try the evolutionary grid mass
    mass_samples = get_padova_posterior(star, var='mass')
    
    if mass_samples is None:
        # Fall back on the spectral type and assume main sequence
        sptnum = MS.SpT_To_Number(spt)
        mass = sptnum2mass(sptnum)
        mass_samples = np.random.normal(loc=mass, scale=mass_err*mass, size=size)
        
    return mass_samples

def get_ages(starname, spt, size=1e4):
    """
    Get age samples for the given star. Try in the following order:
    1. David & Hillenbrand sample
    2. Use mass from the evolutionary grid fits I did. Uses teff/logg from primary star ccfs.
    3. Spectral type (uniformly sample the main sequence age)

    :param starname: The name of the star. To find in the posteriod data, it must be of the form HIPNNNNN
                     (giving the hipparchos number)
    :param N_age: The number of age samples to take
    :return: a numpy array of random samples from the age of the star
    """
    # Is this star in the david & hillenbrand sample?
    if starname.startswith('HIP'):
        hipnum = starname.split('HIP')[-1].strip()
        if int(hipnum) in dh2015.HIP.values:
            logging.debug('Star found in DH2015 sample!')
            # Get the cdf in discrete steps
            age, P = read_dh2015_posterior(hipnum=hipnum, var='age')
            if not np.any(np.isnan(P)):
                cdf = get_cdf(age, P)
                
                # Remove duplicate cdf values
                df = pd.DataFrame(data=dict(cdf=cdf, age=age))
                df.drop_duplicates(subset=['cdf'], inplace=True)
                
                # Calculate the inverse cdf
                inv_cdf = spline(df.cdf.values, df.age.values, k=1)
                
                age_samples = inv_cdf(np.random.uniform(size=size))
                return 10**(age_samples-6.0)
    
    # Try the evolutionary grid mass
    age_samples = get_padova_posterior(starname, var='age')
    
    if age_samples is None:
        logging.debug('Falling back to Main Sequence Age')
        # Fall back on the spectral type and assume main sequence
        data = StarData.GetData(starname)
        spt = data.spectype
        sptnum = MS.SpT_To_Number(spt)

        # Get the age from the Mamajek table
        fcn = MT.get_interpolator('SpTNum', 'logAge')
        ms_age = 10**fcn(sptnum) / 1e6
        logging.info('Main Sequence age for {} is {:.0f} Myr'.format(starname, ms_age))

        # Sample from 0 to the ms_age
        age_samples = np.random.uniform(0, ms_age, size=size)
        return age_samples
        
    logging.debug('Using Padova Evolutionary Grids')
    return 10**(age_samples-6)



class IsochroneInterpolator(object):
    def __init__(self):
        df = Feiden.MASTERDF
        df['logAge'] = np.log10(df.Age)

        points = np.zeros((len(df),2))
        points[:,0] = df.logT.values
        points[:,1] = df.logAge.values
        self.fcn = scipy.interpolate.LinearNDInterpolator(points, df.Msun.values)
    def __call__(self, teff, age):
        return self.fcn(np.log10(teff), age)

    
class SpectralTypeInterpolator(object):
    def __init__(self):
        MS = SpectralTypeRelations.MainSequence()
        MT = Mamajek_Table.MamajekTable()
        self.fcn = MT.get_interpolator('Teff', 'Msun')
    
    def __call__(self, teff):
        return self.fcn(teff)


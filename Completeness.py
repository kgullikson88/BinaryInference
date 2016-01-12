"""
Functions for estimating the completeness functions 
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import multiprocessing
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.stats import gaussian_kde
import pandas as pd 
import Mamajek_Table
from astropy import units as u, constants
import os
import h5py
import StarData
import Priors
import logging
import pickle
from scipy.interpolate import LinearNDInterpolator
from HelperFunctions import IsListlike

PRIOR_HDF5 = 'data/OrbitPrior.h5'


# Read in the Barnes & Kim (2010) table (it is in LaTex format)
home = os.environ['HOME']
bk_filename = '{}/Dropbox/School/Research/Databases/SpT_Relations/Barnes_and_Kim2010_table1.txt'.format(home)
bk_data = pd.read_csv(bk_filename, sep='&', usecols=(0,1,2,3,4), dtype={'Mass': np.float,
                                                                      'logT': np.float,
                                                                      'logL': np.float,
                                                                      'Age': np.float,
                                                                      'global_tc': np.float})

bk_data = bk_data.assign(Teff = lambda x: 10**x.logT)
bk_data = bk_data.assign(Radius = lambda x: 10**(0.5*x.logL) * (x.Teff/5680)**(-2))

# Interpolate
teff2tau_int = spline(bk_data.Teff, bk_data.global_tc)
teff2radius_int = spline(bk_data.Teff, bk_data.Radius)

# Make functions that treat extrapolation in a slightly more sane way
Tmin = bk_data.Teff.min()
Tmax = bk_data.Teff.max()
def teff2tau(T):
    T = np.atleast_1d(T)
    lowT = T < Tmin
    highT = T > Tmax
    good = (T >= Tmin) & (T <= Tmax)
    retval = np.empty_like(T, dtype=np.float)
    retval[lowT] = teff2tau_int(Tmin)
    retval[highT] = teff2tau_int(Tmax)
    retval[good] = np.maximum(0.1, teff2tau_int(T[good]))
    return retval

def teff2radius(T):
    T = np.atleast_1d(T)
    lowT = T < Tmin
    highT = T > Tmax
    good = (T >= Tmin) & (T <= Tmax)
    retval = np.empty_like(T, dtype=np.float)
    retval[lowT] = teff2radius_int(Tmin)
    retval[highT] = teff2radius_int(Tmax)
    retval[good] = np.maximum(0.1, teff2radius_int(T[good]))
    return retval


def lnlike(P, P_0, t, tau, k_C, k_I):
    """
    This is the likelihood function for getting the period out of Equation 19 in Barnes (2010)
    :param P: Period (what we will be solving for)
    :param P_0: Initial period
    :param t: age
    :param tau: convection turnover timescale
    :param k_C: parameter (constant)
    :param k_I: parameter (constant)
    :return:
    """
    penalty = 0.0
    if P < 1e-6:
        penalty = P ** 2
        P = 0.1
    retval = (k_C * t / tau - np.log(P / P_0) - k_I * k_C / (2.0 * tau ** 2) * (P ** 2 - P_0 ** 2)) ** 2 + penalty
    return retval


def poolfcn(args):
    return minimize_scalar(args[0], bracket=args[1], bounds=args[2], method='brent', args=args[3:]).x


def get_period_dist_parallel(ages, P0_min, P0_max, T_star, N_P0=1000, k_C=0.646, k_I=452, nproc=None, safe=True):
    """
    All-important function to get the period distribution out of stuff that I know - parallel version!
    :param ages: Random samples for the age of the system (Myr)
    :param P0_min, P0_max: The minimum and maximum values of P0, the initial period. (days)
                           We will choose random values in equal log-spacing.
    :param T_star: The temperature of the star, in Kelvin
    :keyword N_age, N_P0: The number of samples to take in age and initial period
    :keyword k_C, k_I: The parameters fit in Barnes 2010
    :keyword safe: Remove the periods with values < 0?
    """

    # Set up multiprocessing and a fit function with only one argument set
    pool = multiprocessing.Pool(processes=nproc)
    def errfcn(P, args):
        return lnlike(P, *args)
    bracket = [4.0, 7.0]
    bounds = [0.1, 100]

    # Convert temperature to convection turnover timescale
    tau = teff2tau(T_star)
    period_list = []
    args = []
    #args = [(lnlike, bracket, bounds, P0, age, tau, k_C, k_I) for P0 in np.random.uniform(P0_min, P0_max, size=N_P0) for age in ages]
    for age in ages:
        #P0_vals = np.random.uniform(P0_min, P0_max, size=N_P0)
        P0_vals = 10**np.random.uniform(np.log10(P0_min), np.log10(P0_max), size=N_P0)
        a = [(lnlike, bracket, bounds, P0, age, tau, k_C, k_I) for P0 in P0_vals]
        args.extend(a)
    #    period_list.extend(pool.map(poolfcn, args))
    period_list = pool.map(poolfcn, args)
    p0_list = [ag[3] for ag in args]
    age_list = [ag[4] for ag in args]

    pool.close()
    pool.join()

    P = np.array(period_list)
    if P.ndim > 1:
        P = P[:, 0]
    P0 = np.array(p0_list)
    age = np.array(age_list)
    teff = np.ones_like(P)*T_star
    if safe:
        return P[P > 0], P0[P > 0], age[P > 0], teff[P > 0]
    return P, P0, age, teff


def get_gyro_vsini_samples(T_sec, age, age_err=None, P0_min=0.1, P0_max=5, N_age=1000, N_P0=1000, k_C=0.646, k_I=452,
                      nproc=None):
    """
    Get samples from the probability distribution function of vsini for a star of the given temperature, at the given age
    :param T_sec: float - the temperature of the companion
    :param age: float, or numpy array: If a numpy array, should be a bunch of random samples of the age using
                whatever PDF you want for the age. Could be MCMC samples, for instance. If a float, this
                function will generate N_age samples from a gaussian distribution with mean age and
                standard deviation age_err. Units: Myr
    :param age_err: float - Only used if age is a float or length-1 array. Gives the standard deviation
                    of the gaussian with which we will draw ages.
    :param P0_min: float - The minimum inital period (days). Should be near the breakup velocity of the star.
    :param P0_max: float - The maximum initial period, in days. Generally, should be of order 1-10.
    :param N_age: The number of age samples to draw. Ignored if age is an numpy array with size > 1
    :param N_P0: The number of initial period samples to draw.
    :param k_C: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :param k_I: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :return: A numpy array with samples of the vsini for the star
    """

    # Figure out what to do with age
    if isinstance(age, float) or (isinstance(age, np.ndarray) and age.size == 1):
        if age_err is None:
            raise ValueError('Must either give several samples of age, or an age error!')
        age = np.random.normal(loc=age, scale=age_err, size=N_age)

    # Get the period distribution
    periods, _ = get_period_dist_parallel(age, P0_min, P0_max, T_sec, N_P0=N_P0, k_C=k_C, k_I=k_I, nproc=nproc)

    # Convert to an equatorial velocity distribution by using the radius
    R = teff2radius(T_sec)
    v_eq = 2.0*np.pi*R*constants.R_sun/(periods*u.day)

    # Finally, sample random inclinations to get a distribution of vsini
    vsini = v_eq.to(u.km/u.s) * np.random.uniform(0, 1., size=periods.size)

    return vsini


def get_age(star):
    """ Get samples from the age for the given star.
    """
    with h5py.File(PRIOR_HDF5, 'r') as infile:
        if star in infile.keys():
            return infile[star]['system_age'].value 

    from Priors import get_age
    spt = StarData.GetData(star).spectype
    return get_ages(star, spt)



def maxwellian(v, alpha, l=0):
    prob = (v-l)**2 * np.sqrt(2/np.pi) / alpha**3 * np.exp(-(v-l)**2 / (2*alpha**2))
    prob[v < l] = 0.0
    return prob


def get_zr2011_velocity(mass, size=1e4, reduce_arr=True, df=None):
    """ 
    Get samples from the velocity pdf as tabulated
    by Table 4 in Zorec & Royer (2011)
    
    Parameters:
    ============
    - mass:       The mass of the star (in solar masses)
    - size:       The number of samples 
    - reduce_arr: Boolean flag. Should re only take the unique masses or all of them?
    - df:         A dataframe with all of the necessary columns. Don't give this 
                  unless you know what you are doing!

    Returns:
    ========
    Samples from the PDF for the equatorial velocity (in km/s)
    """

    # Read in and update the columns a bit.
    if df is None:
        df = pd.read_csv('data/velocity_pdfs.csv', header=1)
        df['mid_mass'] = (df.mass_high + df.mass_low) / 2.0
        df['slow_alpha'] = df.slow_mu / np.sqrt(2)
        df['fast_alpha'] = df.fast_mu / np.sqrt(2)
        df['slow_frac'] /= 100.0
        df['fast_frac'] /= 100.0

    # Reduce the size of the mass array, if requested
    if reduce_arr:
        mass = np.unique(mass)

    # Interpolate the parameters as a function of mass
    columns = ['slow_frac', 'slow_alpha', 'fast_frac', 'fast_alpha', 'fast_l']
    interpolators = {col: spline(df.mid_mass, df[col], k=1) for col in columns}
    pars = {col: interpolators[col](mass) for col in interpolators.keys()}

    # Make large arrays. This may be memory intensive!
    v_arr = np.arange(0, 500, 0.1)
    v, slow_frac = np.meshgrid(v_arr, pars['slow_frac'])
    v, slow_alpha = np.meshgrid(v_arr, pars['slow_alpha'])
    v, fast_frac = np.meshgrid(v_arr, pars['fast_frac'])
    v, fast_alpha = np.meshgrid(v_arr, pars['fast_alpha'])
    v, fast_l = np.meshgrid(v_arr, pars['fast_l'])

    # Get the probability for each mass point
    prob = (slow_frac*maxwellian(v=v, alpha=slow_alpha, l=0.0) + 
            fast_frac*maxwellian(v=v, alpha=fast_alpha, l=fast_l))

    # Marginalize over the mass
    P_avg = np.mean(prob, axis=0)

    # Convert to cdf
    cdf = Priors.get_cdf(v_arr, P_avg)

    # Remove duplicate cdf values
    tmp = pd.DataFrame(data=dict(cdf=cdf, velocity=v_arr))
    tmp.drop_duplicates(subset=['cdf'], inplace=True)
            
    # Calculate the inverse cdf
    inv_cdf = spline(tmp.cdf.values, tmp.velocity.values, k=1)
            
    # Calculate samples from the inverse cdf
    velocity_samples = inv_cdf(np.random.uniform(size=size))

    return velocity_samples



def parse_braganca(fname='data/Braganca2012.tsv'):
    """ Parse the braganca 2012 data file.
    """
    import re
    pattern = '([OB][0-9])'
    df = pd.read_csv('data/Braganca2012.tsv', sep='|', header=70)[2:].reset_index()

    spt = []
    hip = []
    vsini = []
    teff = []
    for _, row in df.iterrows():
        for m in re.finditer(pattern, row['SpT']):
            spt.append(row['SpT'][m.start():m.end()])
            hip.append(row['HIP'])
            teff.append(row['Teff'])
            vsini.append(row['<vsini>'])

    df = pd.DataFrame(data=dict(HIP=hip, SpT=spt,
                                vsini=vsini, Teff=teff))
    for col in ['Teff', 'vsini']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_braganca_samples(df=None, size=1):
    # Get the pdf function
    vsini_pdf = get_braganca_pdf()
    v = np.arange(0, 500, 0.1)
    P = vsini_pdf(v)

    # Convert to cdf
    cdf = Priors.get_cdf(v, P)

    # Remove duplicate cdf values
    tmp = pd.DataFrame(data=dict(cdf=cdf, velocity=v))
    tmp.drop_duplicates(subset=['cdf'], inplace=True)
            
    # Calculate the inverse cdf
    inv_cdf = spline(tmp.cdf.values, tmp.velocity.values, k=1)
            
    # Calculate samples from the inverse cdf
    velocity_samples = inv_cdf(np.random.uniform(size=size))

    return velocity_samples


def get_braganca_pdf(df=None):
    """ Return a callable that gives the PDF of the vsini distribution from Braganca2012
    """
    if df is None:
        df = parse_braganca()

    vsini = df.loc[df.vsini.notnull(), 'vsini'].values
    kde = gaussian_kde(vsini, bw_method='scott')
    return kde


def get_barnes_interpolator():
    with open('data/Gyro.pkl', 'rb') as f:
        tri, values = pickle.load(f)
    gyro_raw_fcn = LinearNDInterpolator(tri, values)

    def Period(Teff, age, P0):
        """ 
        Get the current rotation period of a star with the given parameters.

        Parameters:
        ===========
        -Teff:      The effective temperature, in Kelvin
        -age:       The age of the star, in Myr
        -P0:        The initial rotation period, in days.

        Returns:
        ========
        The current rotation period, in days.
        """
        return gyro_raw_fcn(np.log10(Teff), np.log10(age)+6, P0)

    return Period




class VelocityPDF(object):
    def __init__(self):
        self.zr2011_df = None
        return        
    

    def _compute_zr2011_dataframe(self):
        """ 
        Get the dataframe needed for the mid-range temperatures, 
        and add a mass bin for 6000 K using the gyrochronology relation.
        """
        # Read in the dataframe from disk.
        df = pd.read_csv('data/velocity_pdfs.csv', header=1)

        # Compute equatorial velocities for a 6000 K star at this age.
        teff = np.ones_like(self.age) * 6000.0
        v_eq = self._gyro_velocities(teff, self.age).to(u.km/u.s).value 
        v_eq[v_eq > 500] = np.nan  # Remove unphysical vsini values.

        # Calculate approximate maxwellian parameters from the velocities.
        alpha = np.sqrt(np.nanvar(v_eq) * np.pi / (3*np.pi - 8))
        l = np.nanmedian(v_eq) - 2*alpha*np.sqrt(2/np.pi)

        # Add a row to the dataframe with this information
        df.loc[df.index.max()+1] = [1.0, 1.24, 0, 25, 100, alpha*np.sqrt(2), l]

        # Calculate a few more columns for the dataframe
        df['mid_mass'] = (df.mass_high + df.mass_low) / 2.0
        df['slow_alpha'] = df.slow_mu / np.sqrt(2)
        df['fast_alpha'] = df.fast_mu / np.sqrt(2)
        df['slow_frac'] /= 100.0
        df['fast_frac'] /= 100.0

        # Sort so that interpolation works
        df = df.sort_values(by='mid_mass').reset_index()

        return df

    
    def _gyro_velocities(self, teff, age, P0_min=0.1, P0_max=5.0):
        try:
            period_fcn = self.period_fcn
        except AttributeError:
            period_fcn = get_barnes_interpolator()
            self.period_fcn = period_fcn
        
        #P0 = np.random.uniform(P0_min, P0_max, teff.size)
        P0 = 10**np.random.uniform(np.log10(P0_min), np.log10(P0_max), size=teff.size)
        period = period_fcn(teff, age, P0)

        # Convert to an equatorial velocity distribution by using the radius
        R = teff2radius(teff)
        v_eq = 2.0*np.pi*R*constants.R_sun/(period*u.day)
        return v_eq


    def _lowT(self, teff, age):
        v_eq = self._gyro_velocities(teff, age)

        # Sample random inclinations to get a distribution of vsini
        return v_eq.to(u.km/u.s) * np.random.uniform(0, 1., size=teff.size)
    
    
    def _midT(self, mass):
        """ TODO: Treat 6000 to 7000 K differently"""
        if self.zr2011_df is None:
            self.zr2011_df = self._compute_zr2011_dataframe()
        v_eq = get_zr2011_velocity(mass, size=mass.shape, df=self.zr2011_df)
        return v_eq * np.random.uniform(0, 1., size=v_eq.shape)
    
    
    def _highT(self, size):
        return get_braganca_samples(size=size)
    
    
    
    def get_vsini_samples(self, Teff, age, size=1):
        """ TODO: Insert docstring
        """
        # Make sure everything is a numpy array with commensurate shapes
        Teff = np.atleast_1d(Teff)
        age = np.atleast_1d(age)
        if Teff.size > 1 and age.size > 1:
            assert Teff.size == age.size 
        if Teff.size == 1 and age.size == 1:
            Teff = np.ones(size)*Teff
            age = np.ones(size)*age 
        else:
            if Teff.size > 1 and age.size == 1:
                age = np.ones_like(Teff) * age 
            elif Teff.size == 1 and age.size > 1:
                Teff = np.ones_like(age) * Teff
            size = Teff.size

        # Convert temperature to mass, assuming main-sequence
        MT = Mamajek_Table.MamajekTable()
        teff2mass = MT.get_interpolator('Teff', 'Msun')
        mass = teff2mass(Teff)
        
        # Save class variables
        self.Teff = Teff
        self.age = age
        self.mass = mass
        
        # Split by temperature
        lowT = self.Teff < 6000
        midT = (self.Teff >= 6000) & (self.Teff < 14000)
        highT = self.Teff >= 14000
        
        # Make a vsini array
        vsini = np.empty_like(self.Teff, dtype=np.float)
        
        # Fill by temperature
        if lowT.sum() > 0:
            vsini[lowT] = self._lowT(self.Teff[lowT], self.age[lowT])
        if midT.sum() > 0:
            vsini[midT] = self._midT(self.mass[midT])
        if highT.sum() > 0:
            vsini[highT] = self._highT(size=highT.sum())
        
        return vsini



def get_completeness(starname, date):
    """ 
    Get the completeness as a function of mass-ratio for the 
    given star and observation date.
    """

    # Read in the completeness vs. temperature
    with h5py.File('data/Completeness.h5', 'r') as comp:
        if starname in comp.keys() and date in comp[starname].keys():
            ds = comp[starname][date]['marginalized']
            comp_df = pd.DataFrame(data=ds.value, columns=ds.attrs['columns'])
            comp_df['Instrument'] = ds.attrs['Instrument']
        else:
            logging.warn('starname/date combination not found in dataset!')
            return None

    # Get the primary mass 
    spt = StarData.GetData(starname).spectype
    primary_mass_samples = Priors.get_primary_mass(starname, spt)
    comp_df['M1'] = np.median(primary_mass_samples)

    # Convert the companion temperature to mass using spectral type relations 
    spt_int = Priors.SpectralTypeInterpolator()
    comp_df['M2'] = comp_df.Temperature.map(spt_int)

    # Make mass ratio and return
    comp_df['q'] = comp_df.M2 / comp_df.M1

    # Make sure all columns are floats
    for col in ['Temperature', 'Detection_Rate', 'vsini', 'M1', 'M2', 'q']:
        comp_df[col] = pd.to_numeric(comp_df[col], errors='coerce')
    return comp_df 

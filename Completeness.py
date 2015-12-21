"""
Functions for estimating the completeness functions 
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import multiprocessing
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import pandas as pd 
from astropy import units as u, constants
import os
import h5py
import StarData

PRIOR_HDF5 = 'OrbitPrior.h5'


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
    if T < Tmin:
        return teff2tau(Tmin)
    elif T > Tmax:
        return teff2tau(Tmax)
    return max(0.1, teff2tau_int(T))
def teff2radius(T):
    if T < Tmin:
        return teff2radius_int(Tmin)
    elif T > Tmax:
        return teff2radius_int(Tmax)
    return teff2radius_int(T)


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


def get_period_dist_parallel(ages, P0_min, P0_max, T_star, N_P0=1000, k_C=0.646, k_I=452, nproc=None):
    """
    All-important function to get the period distribution out of stuff that I know - parallel version!
    :param ages: Random samples for the age of the system (Myr)
    :param P0_min, P0_max: The minimum and maximum values of P0, the initial period. (days)
                           We will choose random values in equal log-spacing.
    :param T_star: The temperature of the star, in Kelvin
    :keyword N_age, N_P0: The number of samples to take in age and initial period
    :keyword k_C, k_I: The parameters fit in Barnes 2010
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
        P0_vals = np.random.uniform(P0_min, P0_max, size=N_P0)
        a = [(lnlike, bracket, bounds, P0, age, tau, k_C, k_I) for P0 in P0_vals]
        args.extend(a)
    #    period_list.extend(pool.map(poolfcn, args))
    period_list = pool.map(poolfcn, args)

    pool.close()
    pool.join()
    P = np.array(period_list)
    return P[P > 0]


def get_vsini_pdf(T_sec, age, age_err=None, P0_min=0.1, P0_max=5, N_age=1000, N_P0=1000, k_C=0.646, k_I=452,
                  nproc=None):
    """
    Get the probability distribution function of vsini for a star of the given temperature, at the given age
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
    periods = get_period_dist_parallel(age, P0_min, P0_max, T_sec, N_P0=N_P0, k_C=k_C, k_I=k_I, nproc=nproc)

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
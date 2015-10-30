import numpy as np
import scipy.stats
from astropy import units as u, constants
from scipy.optimize import newton
import pandas as pd

import HelperFunctions
import IMF_utils
from Mamajek_Table import MamajekTable
import Orbit
import Simulation


def truncated_expon(scale=0.0, Nsamp=100):
    """ Get random samples from an exponential distribution confined to [0,1]
    """
    u = np.random.uniform(size=Nsamp)
    x = u**(1./(1-scale))
    return x


def get_eccentric_anomaly(M, e):
    """
    Get the eccentric anomaly (E) from the mean anomaly (M) and orbital eccentricity (e)
    Uses the equation M = E - esinE
    """
    if HelperFunctions.IsListlike(M):
        return np.array([get_eccentric_anomaly(Mi, e) for Mi in M])

    func = lambda E: E - e*np.sin(E) - M
    dfunc = lambda E: 1.0 - e*np.cos(E)
    d2func = lambda E: e*np.sin(E)
    
    output = newton(func, np.pi, fprime=dfunc, fprime2=d2func)

    #output = minimize_scalar(chisquare, bounds=[0, 2*np.pi], method='brent')
    return output


def get_true_anomaly(E, e):
    """
    Get the true anomaly from the eccentric anomaly (E) and the eccentricity
    """
    A = (np.cos(E) - e)/(1-e*np.cos(E))
    B = (np.sqrt(1.-e**2) * np.sin(E)) / (1.-e*np.cos(E))
    return np.arctan2(B, A)


def get_rv(phase, e, K1, w):
    """
    Get the radial velocity at the given phase, given the parameters:
    e = eccentricity
    K1 = semiamplitude
    w = longitude of pericenter (radians)
    """
    M = 2.0*np.pi*phase
    Erad = get_eccentric_anomaly(M, e)
    nu = get_true_anomaly(Erad, e)

    return K1 * (np.cos(nu+w) + e*np.cos(w))



class ForwardModeler(object):
    """
    A class to do forward modeling to measure population parameters amongst observational biases
    """
    
    def __init__(self, *args, **kwargs):
        """ Eventually, this should take all the data I have in some reasonable way
        """
        MT = MamajekTable()
        self.mass2Vmag = MT.get_interpolator('Msun', 'Mv')
        # Log-normal parameters for the parallax. 
        #These come from fitting the Hipparchos data and give a good 'chi-by-eye' fit.
        self.plx_shape, self.plx_scale, self.plx_loc = (0.86414506569840466, 5.2836463711499855, -0.36291878650952103)
    
    def IMF(self, low_mass, high_mass, size=1):
        """
        Draw 'size' samples from the IMF between low_mass and high_mass
        """
        masses = IMF_utils.inverse_imf(np.random.uniform(size=size), mmin=low_mass, mmax=high_mass)
        return masses
        
    
    def model(self, pars, N_obs=None, Vmag_lim=None, N_total=10000, *args, **kwargs):
        """
        Generate samples from the primary star distributions given in pars.
        
        Parameters
        ==========
         - pars: An iterable containing the parameters for the parent population
         - N_obs: The number of observations to pull from the parent population (volume-limited sample).
                  Either this or Vmag_lim **MUST** be given.
         - Vmag_lim: The V-magnitude limiting magnitude to include in the sample. Either this or N_obs
                     **MUST** be given.
         - N_total: The number of samples to draw representing the parent population. This gives a 
                    tradeoff between accuracy and speed.
         
        Returns:
        =========
        A pandas DataFrame containing lots of information about the synthetic binary stars.
        """
        # First, make sure we were given one of N_obs or Vmag_lim
        if N_obs is None and Vmag_lim is None:
            raise ValueError('You must give at least one of N_obs or Vmag_lim!')
        print(N_obs, Vmag_lim)
        
        # physical parameters
        mult_rate = pars[0]  # Overall multiplicity rate
        mrd_gamma = pars[1]  # Mass-ratio distribution power law factor
        sep_mu, sep_sigma = pars[2:4]  # Separation distribution parameters (assumed log-normal)
        ecc_alpha = pars[4]  # Eccentricity distribution power law factor
        
        # Observational parameters
        Mlow, Mhigh = pars[5:7]  # Lowest and highest masses in the sample
        
        # Make a parent distribution with the correct multiplicity rate
        targets = np.random.uniform(size=N_total)
        population = pd.DataFrame(data=dict(binary=(targets < mult_rate)))
        population['q'] = np.nan
        population['e'] = np.nan
        population['a'] = np.nan
        population['vel'] = np.nan
        binary_idx = population.index[population.binary]
        
        # Take random samples from the parent population
        population['M1'] = self.IMF(Mlow, Mhigh, size=N_total)
        N_binary = sum(population.binary)
        population.ix[binary_idx, 'q'] = truncated_expon(scale=mrd_gamma, Nsamp=N_binary)
        population.ix[binary_idx, 'e'] = truncated_expon(scale=ecc_alpha, Nsamp=N_binary)
        population.ix[binary_idx, 'a'] = np.exp(np.random.normal(loc=sep_mu, scale=sep_sigma, size=N_binary))
        
        # Calculate the velocity of each of the binary observations
        phase = np.random.uniform(size=N_binary)
        omega = 2*np.pi*np.random.uniform(size=N_binary)
        sini = np.random.uniform(size=N_binary)
        M1 = population.ix[binary_idx, 'M1'].values * u.M_sun
        a = population.ix[binary_idx, 'a'].values * u.AU
        e = population.ix[binary_idx, 'e'].values
        q = population.ix[binary_idx, 'q'].values
        K1 = sini * np.sqrt((constants.G * M1 / ((1+q)*a*(1-e**2))).decompose()).to(u.km/u.s).value
        #K1 = K1.to(u.km/u.s).value
        population.ix[binary_idx, 'vel'] = np.array([get_rv(phi, ecc, K, w) for phi, ecc, K, w in zip(phase, e, K1, omega)])
        
        
        # Choose random distances for the population stars using the Hipparchos parallaxes for a distribution.
        # This ASSUMES that there is no dependence of spectral type vs. separation (there shouldn't be, right?)
        plx = scipy.stats.lognorm.rvs(loc=self.plx_loc, s=self.plx_shape, scale=self.plx_scale, size=N_total)
        population['distance'] = 1000. / plx
        
        # Get the angular separation for the binaries
        # TODO: Use the phase/orbit information to do this right!
        f = 1.0 # Edit this with the inclination and such later
        population['ang_sep'] = f * population['a'] / population['distance']
        
        # Put the primary star magnitudes into the population dataframe.
        #If the angular separation is < 1", combine the magnitude from the primary and secondary stars.
        prim_mag = self.mass2Vmag(population.M1)
        sec_mag = self.mass2Vmag(population.M1 * population.q)
        population['Vmag'] = prim_mag + 5*np.log10(population.distance) - 5.0
        condition = (population.binary) & (population.ang_sep < 1.0)
        population.loc[condition, 'Vmag'] = HelperFunctions.add_magnitudes(prim_mag[condition.values], sec_mag[condition.values])
        
        # Draw the sample from the population in the appropriate way
        if Vmag_lim is not None:
            sample = population.loc[population.Vmag < Vmag_lim].copy()
            if N_obs is not None:
                sample = sample.sample(n=N_obs).reset_index()
        else:
            sample = population.sample(n=N_obs).reset_index()
        
        return sample


def make_representative_sample(gamma, mu, sigma, eta, size=1, min_mass=1.5, max_mass=20.0):
    """ Generate a representative sample from the population parameters gamma, mu, sigma, and eta.
    """
    a = np.random.lognormal(mean=mu, sigma=sigma, size=size) * u.AU
    e = truncated_expon(scale=eta, Nsamp=size)
    q = truncated_expon(scale=gamma, Nsamp=size)
    prim_mass = IMF_utils.inverse_imf(np.random.uniform(size=size), mmin=min_mass, mmax=max_mass) * u.M_sun
    M0 = np.random.uniform(0, 2 * np.pi, size=size) * u.radian
    Omega = np.random.uniform(0, 2 * np.pi, size=size) * u.radian
    omega = np.random.uniform(0, 2 * np.pi, size=size) * u.radian
    sini = np.random.uniform(0, 1, size=size)
    i = np.arcsin(sini) * u.radian
    Period = np.sqrt(4 * np.pi ** 2 * a ** 3 / (constants.G * (prim_mass + prim_mass * q))).to(u.year)

    sample = pd.DataFrame(data=dict(a=a, e=e, q=q, M_prim=prim_mass, Period=Period.to(u.year),
                                    M0=M0.to(u.degree), big_omega=Omega.to(u.degree),
                                    little_omega=omega.to(u.degree), i=i.to(u.degree)))
    return sample


def make_malmquist_sample(gamma, mu, sigma, eta, size=1, min_mass=1.5, max_mass=20.0, Vlim=6.0):
    """ Make a magnitude-limited sample from the population parameters gamma, mu, sigma, and eta.
    """

    # Make a representative sample with 10x the size
    sample = make_representative_sample(gamma, mu, sigma, eta, size=size * 1e4, min_mass=min_mass, max_mass=max_mass)

    # Sample distances such that they uniformly fill a sphere
    # distance = (np.random.uniform(size=len(sample))) ** (1. / 3.) * 1000.0
    # Sample distances from a milky-way-like disk

    r, z = Simulation.sample_disk(Rmax=1e4, Npoints=len(sample))
    distance = np.sqrt(r ** 2 + z ** 2)

    # Get the mamajek table. Sort by mass, and remove the NaNs
    MT = MamajekTable()
    df = MT.mam_df.dropna(subset=['Msun']).sort_values(by='Msun')
    mass = df['Msun']
    vmag = df['Mv']

    # Interpolate the table with a smoothing spline.
    from scipy.interpolate import UnivariateSpline

    spline = UnivariateSpline(mass, vmag, s=0.9,
                              ext=3)  # The s parameter was fine-tuned. See the MalmquistBias notebook.

    def get_vmag(q, mass, d):
        """ Get the V-band magnitude as a function of mass-ratio (q), primary mass(mass),
        and distance (d)
        """

        M1 = spline(mass)
        M2 = spline(q * mass)
        M_total = HelperFunctions.add_magnitudes(M1, M2)
        V = M_total + 5 * np.log10(d) - 5
        return V

    # Get the V magnitude of the sample stars
    sample['Vmag'] = get_vmag(sample['q'].values, sample['M_prim'].values, distance)
    sample['distance'] = distance

    # Keep 'size' of the stars with Vmag < Vlim
    return sample.loc[sample.Vmag < Vlim].sample(size).reset_index().copy()


def sample_orbit(star, N_rv, N_imag, rv1_err=None, rv2_err=None, pos_err=None, distance=100):
    """ Sample the binary orbit represented by the 'star' structure at N_rv random times for
    RV measurements, and N_imag random times for imaging measurements

    Parameters:
    ===========
    - star:    a pandas DataFrame with the orbital parameters (such as returned by make_representative_sample)
    - N_rv, N_image:  The number of radial velocity and imaging observations.
    - rv1_err: Uncertainty on the primary star velocity measurements (in km/s or velocity Quantity)
    - rv2_err: Uncertainty on the secondary star velocity measurements (in km/s or velocity Quantity)
    - pos_err: On-sky positional error (in arcseconds or angle Quantity)
    - distance: The distance to the star (in parsecs or a distance Quantity)

    Returns:
    ========
    - Times of RV observations, primary and secondary rv measurements,
      times of imaging observations, rho/theta measurements.
    """
    orbit = Orbit.OrbitCalculator(P=star['Period'], M0=star['M0'], a=star['a'], e=star['e'],
                                  big_omega=star['big_omega'], little_omega=star['little_omega'],
                                  i=star['i'], q=star['q'], primary_mass=star['M_prim'])
    print('K1 = {}\nK2 = {}'.format(orbit.K1, orbit.K2))
    rv_times = np.random.uniform(0, star['Period'], size=N_rv)
    image_times = np.random.uniform(0, star['Period'], size=N_imag)
    rv_primary_measurements = orbit.get_rv(rv_times, component='primary')
    rv_secondary_measurements = -rv_primary_measurements * orbit.K2 / orbit.K1
    rho_measurements, theta_measurements = orbit.get_imaging_observables(image_times, distance=distance)

    # Add errors where requested
    if rv1_err is not None:
        rv_primary_measurements += np.random.normal(loc=0, scale=rv1_err, size=len(rv_times))
    if rv2_err is not None:
        rv_secondary_measurements += np.random.normal(loc=0, scale=rv2_err, size=len(rv_times))
    if pos_err is not None:
        x = rho_measurements * np.cos(theta_measurements) + np.random.normal(loc=0, scale=pos_err,
                                                                             size=len(image_times))
        y = rho_measurements * np.sin(theta_measurements) + np.random.normal(loc=0, scale=pos_err,
                                                                             size=len(image_times))
        rho_measurements = np.sqrt(x ** 2 + y ** 2)
        theta_measurements = np.arctan2(y, x)

    return rv_times, rv_primary_measurements, rv_secondary_measurements, image_times, rho_measurements, theta_measurements, orbit.K1, orbit.K2

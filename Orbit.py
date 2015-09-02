import logging

import numpy as np
from astropy import units as u, constants
from scipy.optimize import newton
import scipy.interpolate

import HelperFunctions
import Fitters


cache = None
G = constants.G.cgs.value  # gravitational constant in cgs units
M_sun = constants.M_sun.cgs.value
year2sec = u.year.to(u.second)
unit_factor = (M_sun / year2sec) ** (1. / 3.) * 1e-5

class OrbitCalculator(object):
    """
    Calculates various quantities for an orbit, 
    given the Keplerian elements
    """
    def __init__(self, P, M0, e, a, big_omega, little_omega, i, 
                 q=1.0, primary_mass=2.0, K1=None, K2=None, 
                 precompute=True):
        """
        Initialize the OrbitCalculator class.

        Parameters
        ===========
        - P:            float
                        Orbital period in years.

        - M0:           float
                        The mean anomaly at epoch (in degrees).

        - e:            float
                        Orbital eccentricity

        - a:            float
                        Semimajor axis in AU. 

        - big_omega:    float
                        The longitude of the ascending node (in degrees).

        - little_omega: float
                        The argument of periastron (in degrees).

        - i:            float
                        Orbital inclination (in degrees).

        - q:            float
                        The mass-ratio of the binary system.

        - primary_mass: float
                        The mass of the primary star in solar masses

        - K1:           float
                        The semiamplitude velocity of the primary star (in km/s).
                        Overrides the calculation from mass, semi-major axis, etc if given

        - K2:           float
                        The semiamplitude velocity of the secondary star (in km/s)
                        Overrides the calculation from mass, semi-major axis, etc if given

        - precompute:   If true, it calculates a bunch of combinations of mean anomaly
                        and eccentricity, and interpolates between them. This provides 
                        about an order of magnitude speedup at the expense of a few seconds
                        of startup time.
        """

        # Save most of the variables as instance variables for use in various functions.
        self.P = P
        self.M0 = M0 * np.pi/180.
        self.e = e
        self.a = a
        self.big_omega = big_omega * np.pi/180.
        self.little_omega = little_omega * np.pi/180.
        self.primary_mass = primary_mass

        # Pre-compute sin(i) and cos(i), two useful quantities.
        inc = i * np.pi / 180. 
        self.sini = np.sin(inc)
        self.cosi = np.cos(inc)

        # Compute the orbit radial velocity semi-amplitude.
        if K1 is None:
            self.K1 = q * self.sini * np.sqrt(
                constants.G * self.primary_mass * u.M_sun / (self.a * u.AU * (1 - e ** 2) * (1 + q))).to(
                u.km / u.s).value
        else:
            self.K1 = K1
        if K2 is None:
            self.K2 = self.K1 / q
        else:
            self.K2 = K2

        # Compute the Thiele-Innes elements for cartesian calculations
        # Formulas from http://ugastro.berkeley.edu/infrared10/adaptiveoptics/binary_orbit.pdf
        self.A = self.a * (np.cos(self.little_omega)*np.cos(self.big_omega) - 
                           np.sin(self.little_omega)*np.sin(self.big_omega)*self.cosi)
        self.B = self.a * (np.cos(self.little_omega)*np.sin(self.big_omega) + 
                           np.sin(self.little_omega)*np.cos(self.big_omega)*self.cosi)
        self.F = self.a * (-np.sin(self.little_omega)*np.cos(self.big_omega) - 
                           np.cos(self.little_omega)*np.sin(self.big_omega)*self.cosi)
        self.G= self.a * (-np.sin(self.little_omega)*np.sin(self.big_omega) - 
                           np.cos(self.little_omega)*np.cos(self.big_omega)*self.cosi)
        
        self.anomaly_interpolator = None
        if precompute:
            self.anomaly_interpolator = self.get_anomaly_interpolator()
        return


    def get_anomaly_interpolator(self):
        """ Generate or retrieve an interpolator 
        for the eccentric anomaly"""
        global cache
        if cache is not None:
            return cache
        ee = np.arange(0, 1.01, 0.01)
        MM = np.arange(0, 2*np.pi+0.01, 0.01)
        E = np.empty(ee.size*MM.size)
        for i, e in enumerate(ee):
            logging.debug('Calculating eccentric anomalies for e = {}'.format(e))
            E[i*MM.size:MM.size*(i+1)] = self.calculate_eccentric_anomaly(MM, e)
        
        cache = scipy.interpolate.RegularGridInterpolator((ee, MM), E.reshape((ee.size, MM.size)))
        return cache


    def calculate_eccentric_anomaly(self, M, e):
        """
        Get the eccentric anomaly (E) from the mean anomaly (M) and orbital eccentricity (e)
        Uses the equation M = E - esinE
        """
        if self.anomaly_interpolator is not None:
            output = self.anomaly_interpolator((e, M))
            return output

        if HelperFunctions.IsListlike(M):
            return np.array([self.calculate_eccentric_anomaly(Mi, e) for Mi in M])

        func = lambda E: E - e*np.sin(E) - M
        dfunc = lambda E: 1.0 - e*np.cos(E)
        d2func = lambda E: e*np.sin(E)

        output = newton(func, np.pi, fprime=dfunc, fprime2=d2func)
        return output


    def get_true_anomaly(self, time_since_epoch, ret_ecc_anomaly=False):
        """Get the true anomaly of the orbit some time after the epoch at which self.M0 is defined.

        Parameters:
        ===========
        - time_since_epoch: float
                            Gives the time since the epoch at which self.M0 is defined (in years)

        - ret_ecc_anomaly:  boolean
                            Flag for returning the eccentric anomaly as well as the true anomaly.
                            Default = False

        Returns:
        ========
        - True anomaly:     float
                            The true anomaly of the orbit, as an angle (in radians).
        """
        M = (self.M0 + (2*np.pi*time_since_epoch/self.P)) % (2*np.pi)
        E = self.calculate_eccentric_anomaly(M, self.e)
        A = (np.cos(E) - self.e)/(1-self.e*np.cos(E))
        B = (np.sqrt(1.-self.e**2) * np.sin(E)) / (1.-self.e*np.cos(E))
        nu = np.arctan2(B, A)

        return (nu, E) if ret_ecc_anomaly else nu


    def get_rv(self, time_since_epoch, component='primary'):
        """ Get the radial velocity of the selected component, at the given time since epoch.

        Parameters:
        ===========
        - time_since_epoch: float
                            Gives the time since the epoch at which self.M0 is defined (in years)

        - component:        string
                            Which binary component to get the velocity of. Choices are 'primary'
                            and 'secondary', and 'primary' is the default.

        Returns:
        ========
        - Radial velocity:  float 
                            The radial velocity of the chosen binary component (in km/s).
        """
        nu = self.get_true_anomaly(time_since_epoch)
        K = self.K1 if component == 'primary' else self.K2
        return K * (np.cos(nu+self.little_omega) + self.e*np.cos(self.little_omega))


    def get_imaging_observables(self, time_since_epoch, distance=None, parallax=None):
        """ Get the separation and position angle of the star in the plane of the sky.

        Parameters:
        ===========
        - time_since_epoch: float
                            Gives the time since the epoch at which self.M0 is defined (in years).

        - distance :        float
                            The distance from Earth to the star in parsecs. 
                            Either this or parallax MUST be given!

        - parallax :        float
                            The parallax of the star system in arcsecondss.
                            Either this or distance MUST be given!

        Returns:
        ========
        - rho:              float
                            The angular separation between the primary and secondary star (in arcseconds)

        - theta:            float
                            The position angle of the companion, in relation to the primary star (in radians).
        """
        # Make sure either distance or parallax is given
        assert distance is not None or parallax is not None

        if distance is None:
            distance = 1.0 / parallax

        # Calculate the cartesian coordinates first, to get the quadrant right in theta.
        nu, E = self.get_true_anomaly(time_since_epoch, ret_ecc_anomaly=True)
        X = np.cos(E) - self.e
        Y = np.sin(E) * np.sqrt(1-self.e**2)
        x = self.A*X + self.F*Y
        y = self.B*X + self.G*Y

        # Convert to rho/theta
        rho = self.a * (1 - self.e * np.cos(E))
        theta = self.big_omega + np.arctan2(y, x)

        # rho is currently in AU. Convert to on-sky distance in arcsecondss
        rho = rho / distance

        return rho, theta


class FullOrbitFitter(Fitters.Bayesian_LS):
    """
    Fit a binary orbit from radial velocity measurements of the primary and secondary, and
    from imaging observables.

    Parameters:
    ===========
     - rv_times:             ndarray of floats
                             The julian dates at which the radial velocity measurements were taken
                             
     - imaging_times:        ndarray of floats
                             The julian dates at which the imaging measurements were taken
                             
     - rv1_measurements:     ndarray of floats
                             The radial velocity measurements of the primary star (in km/s)
                             
     - rv1_err:              ndarray of floats, or float
                             Uncertainty in the radial velocity measurements of the primary star (in km/s)
                             
     - rv2_measurements:     ndarray of floats
                             The radial velocity measurements of the secondary star (in km/s)
                             
     - rv2_err:              ndarray of floats, or float
                             Uncertainty in the radial velocity measurements of the secondary star (in km/s)
                             
     - rho_measurements:     ndarray of floats
                             measurements of the primary-secondary separation (in arcseconds)
                             
     - theta_measurements:   ndarray of floats
                             measurements of the position angle (in radians)
                             
     - pos_err:              ndarray of floats, or float
                             Uncertainty in the position of the secondary star relative to the primary (in arcseconds).
                             
     - distance:             float
                             Distance from the Earth to the star in question (in parsecs)

    """
    def __init__(self, rv_times, imaging_times, 
                 rv1_measurements, rv1_err, rv2_measurements, rv2_err, 
                 rho_measurements, theta_measurements, pos_err,
                 distance=100.0):
        
        # Put the data into appropriate dictionaries
        x = dict(t_rv=rv_times, t_im=imaging_times)
        y = dict(rv1=rv1_measurements, rv2=rv2_measurements,
                 xpos=rho_measurements*np.cos(theta_measurements),
                 ypos=rho_measurements*np.sin(theta_measurements))
        yerr = dict(rv1=rv1_err, rv2=rv2_err, pos=pos_err)
        
        # List the parameter names
        parnames = ['Period', '$M_0$', 'a', 'e', '$\Omega$', '$\omega$', 'i', '$K_1$', '$K_2$', 'dv1', 'dv2']

        super(FullOrbitFitter, self).__init__(x, y, yerr, param_names=parnames)
        self.distance = distance
        return
        
        
    def model(self, p, x):
        """ Generate observables from the model parameters
        
        Parameters:
        ============
         -p: a list of parameters giving the orbital elements
         -x: A dictionary with keys for the time of the rv and imaging observations
        
        Returns:
        ======== 
           The primary/secondary rv, and the on-sky x- and y-positions
        """
        period, M0, a, e, Omega, omega, i, K1, K2, dv1, dv2 = p
        orbit = OrbitCalculator(P=period, M0=M0, a=a, e=e, 
                                big_omega=Omega, little_omega=omega,
                                i=i, K1=K1, K2=K2)

        rv1 = orbit.get_rv(x['t_rv'], component='primary')
        rv2 = -rv1 * orbit.K2 / orbit.K1
        rho, theta = orbit.get_imaging_observables(x['t_im'], distance=self.distance)
        xpos = rho*np.cos(theta)
        ypos = rho*np.sin(theta)
        return rv1 + dv1, rv2 + dv2, xpos, ypos
    
    def lnlike_rv(self, rv1_pred, rv2_pred, primary=True, secondary=True):
        s = 0.0
        if primary:
            s += -0.5 * np.nansum((rv1_pred - self.y['rv1']) ** 2 / self.yerr['rv1'] ** 2)
        if secondary:
            s += -0.5 * np.nansum((rv2_pred - self.y['rv2']) ** 2 / self.yerr['rv2'] ** 2)
        return s
    
    def lnlike_imaging(self, xpos_pred, ypos_pred):
        return -0.5 * np.nansum(
            ((xpos_pred - self.y['xpos']) ** 2 + (ypos_pred - self.y['ypos']) ** 2) / self.yerr['pos'] ** 2)
        
    def _lnlike(self, pars, primary=True, secondary=True):
        rv1, rv2, xpos, ypos = self.model(pars, self.x)
        s = 0.0
        if len(rv1) > 0:
            s += self.lnlike_rv(rv1, rv2, primary=primary, secondary=secondary)
        if len(xpos) > 0:
            s += self.lnlike_imaging(xpos, ypos)
        return s
    
    def mnest_prior(self, cube, ndim, nparames):
        # Multinest prior
        cube[0] = 10**(cube[0]*5)    # log-uniform in period
        cube[1] = cube[1] * 2*np.pi  # Uniform in mean anomaly at epoch (M0)
        cube[2] = 10**(cube[2]*5)    # Log-uniform in semi-major axis
        cube[3] = cube[3]            # Uniform in eccentricity
        cube[4] = cube[4] * 2*np.pi  # Uniform in big omega
        cube[5] = cube[5] * 2*np.pi  # Uniform in little omega
        cube[6] = cube[6] * 2*np.pi  # Uniform in inclination
        cube[7] = 10**(cube[7]*3)    # Log-uniform in rv semiamplitude (primary)
        cube[8] = 10**(cube[8]*3)    # Log-uniform in rv semiamplitude (secondary)
        cube[9] = cube[9] * 40 - 20  # Uniform in dv1
        cube[10] = cube[10] * 40 - 20  # Uniform in dv2
        return
    
    def lnprior(self, pars):
        # emcee prior
        period, M0, a, e, Omega, omega, i, K1, K2, dv1, dv2 = pars
        if (0 < period < 1e5 and 0 < a < 1e5 and 0 < e < 1 and 0 < Omega < 360. and 0 < omega < 360.
            and 0 < i < 360. and 0 < K1 < 1e3 and 0 < K2 < 1e3 and -20 < dv1 < 20 and -20 < dv2 < 20):

            return 0.0

        return -np.inf


class SpectroscopicOrbitFitter(Fitters.Bayesian_LS):
    """
    Fit a binary orbit from radial velocity measurements of the primary and (optionally) secondary

    Parameters:
    ===========
     - rv_times:             ndarray of floats
                             The julian dates at which the radial velocity measurements were taken

     - rv1_measurements:     ndarray of floats
                             The radial velocity measurements of the primary star (in km/s)

     - rv1_err:              ndarray of floats, or float
                             Uncertainty in the radial velocity measurements of the primary star (in km/s)

     - rv2_measurements:     ndarray of floats
                             The radial velocity measurements of the secondary star (in km/s)

     - rv2_err:              ndarray of floats, or float
                             Uncertainty in the radial velocity measurements of the secondary star (in km/s)

     - gamma:                float (default = 0.4)
                             Parameter describing the mass-ratio distribution exponential decay:
                             ```
                             P(q) = (1-gamma)q^(-gamma)
                             ```

     - mu, sigma:            floats (defaults: mu = 5.3, sigma=2.3)
                             Parameters describing the semi-major axis distribution (log-normal):

     - eta:                  float (default = 0.7)
                             Parameter describing the eccentricity distribution exponential decay:
                             ```
                             P(e) = (1-eta)e^(-eta)
                             ```

     - primary_mass          float (default = 2 Msun)
                             The mass (in solar masses) of the primary star. Used in the prior to convert
                             from period to semimajor axis.

    """

    def __init__(self, rv_times, rv1_measurements, rv1_err, rv2_measurements=None, rv2_err=None,
                 gamma=0.4, mu=np.log(200), sigma=np.log(10), eta=0.7, primary_mass=2.0):

        if rv2_measurements is None:
            rv2_measurements = np.nan * np.ones_like(rv1_measurements)
            rv2_err = 1.0

        # Put the data into appropriate dictionaries
        x = dict(t_rv=rv_times)
        y = dict(rv1=rv1_measurements, rv2=rv2_measurements)
        yerr = dict(rv1=rv1_err, rv2=rv2_err)

        # List the parameter names
        parnames = ['Period', '$M_0$', 'e', '$\omega$', '$K_1$', 'q', 'dv1', 'dv2']

        super(SpectroscopicOrbitFitter, self).__init__(x, y, yerr, param_names=parnames)
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.primary_mass = primary_mass
        return


    def model(self, p, x):
        """ Generate observables from the model parameters

        Parameters:
        ============
         -p: a list of parameters giving the orbital elements
         -x: A dictionary with keys for the time of the rv and imaging observations

        Returns:
        ========
           The primary/secondary rv, and the on-sky x- and y-positions
        """
        period, M0, e, omega, K1, q, dv1, dv2 = p
        orbit = OrbitCalculator(P=period, M0=M0, a=1.0, e=e,
                                big_omega=90.0, little_omega=omega,
                                i=90.0, K1=K1, K2=K1 / q)

        rv1 = orbit.get_rv(x['t_rv'], component='primary')
        rv2 = -rv1 / q

        return rv1 + dv1, rv2 + dv2

    def lnlike_rv(self, rv1_pred, rv2_pred, primary=True, secondary=True):
        s = 0.0
        if primary:
            s += -0.5 * np.nansum((rv1_pred - self.y['rv1']) ** 2 / self.yerr['rv1'] ** 2 +
                                  np.log(2 * np.pi * self.yerr['rv1'] ** 2))
        if secondary:
            s += -0.5 * np.nansum((rv2_pred - self.y['rv2']) ** 2 / self.yerr['rv2'] ** 2 +
                                  np.log(2 * np.pi * self.yerr['rv2'] ** 2))
        return s


    def _lnlike(self, pars, primary=True, secondary=True):
        rv1, rv2 = self.model(pars, self.x)
        return self.lnlike_rv(rv1, rv2, primary=primary, secondary=secondary)


    def mnest_prior(self, cube, ndim, nparams):
        # Multinest prior
        q = cube[5] ** (1.0 / (1.0 - self.gamma))

        # pretend cube[0] (the period) is actually semimajor axis to calculate inverse CDF
        lna = scipy.stats.norm.ppf(cube[0], loc=self.mu, scale=self.sigma)
        mass = self.primary_mass * (1 + q)  # Total system mass (depends on mass-ratio)
        cube[0] = np.exp(1.5 * lna) / mass

        cube[1] = cube[1] * 360.  # Uniform in mean anomaly at epoch (M0)
        cube[3] = cube[3] * 360.  # Uniform in little omega

        # The eccentricity is exponential in [0,1])
        cube[2] = cube[2] ** (1.0 / (1.0 - self.eta))

        # The rv semi-amplitude is a bit tricky. We will sample sini uniformly, assume we know M1.
        # the mass-ratio is in cube[5], and the period is (now) in cube[0]
        sini = cube[4]
        cube[4] = q * sini / np.sqrt(1 - cube[2] ** 2) * (2 * np.pi * G * self.primary_mass * (1 + q) / cube[0]) ** (
        1. / 3.) * unit_factor
        cube[5] = q

        # Give the RV offsets uniform priors
        cube[6] = cube[6] * 40 - 20
        cube[7] = cube[7] * 40 - 20
        return


    def lnprior(self, pars):
        # emcee prior
        period, M0, e, omega, K1, q, dv1, dv2 = pars
        gamma, mu, sigma, eta = self.gamma, self.mu, self.sigma, self.eta
        mass = self.primary_mass * (1 + q)
        lna = 2. / 3. * np.log(period) + 1. / 3. * np.log(mass)
        if (0 < period < 1e5 and -20 < M0 < 380 and 0 < e < 1 and -20 < omega < 380.
            and 0 < K1 < 1e3 and 0 < q < 1 and -20 < dv1 < 20 and -20 < dv2 < 20):
            ecc_prior = np.log(1 - eta) - eta * np.log(e)
            q_prior = np.log(1 - gamma) - gamma * np.log(q)
            a_prior = -0.5 * (np.log(2 * np.pi * sigma ** 2) + (lna - mu) ** 2 / sigma ** 2)
            return ecc_prior + q_prior + a_prior

        return -np.inf


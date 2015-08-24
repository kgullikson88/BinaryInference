import numpy as np
from astropy import units as u, constants
import HelperFunctions
from scipy.optimize import newton
import scipy.interpolate
import logging

cache = None

class OrbitCalculator(object):
    """
    Calculates various quantities for an orbit, 
    given the Keplerian elements
    """
    def __init__(self, P, M0, e, a, big_omega, little_omega, i, q=1.0, primary_mass=2.0*u.M_sun, precompute=True):
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
        self.K1 = (1+q) * self.sini * np.sqrt(constants.G * self.primary_mass*u.M_sun * (1+q) / (self.a*u.AU*(1-e**2))).to(u.km/u.s).value
        self.K2 = self.K1 / q

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
            return output*u.radian

        if HelperFunctions.IsListlike(M):
            return np.array([self.calculate_eccentric_anomaly(Mi, e).value for Mi in M])

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
                            The angular separation between the primary and secondary star (in radians)

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




#class ProbabilisticCalculator(object):
#    """ This class assumes uniform distributions of the orbital orientation parameters, 
#    and samples a bunch of them to get probability of observables"""
#    def __init__(self, )





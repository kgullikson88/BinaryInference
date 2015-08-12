import numpy as np
from astropy import units as u, constants
import HelperFunctions
from scipy.optimize import newton

class OrbitCalculator(object):
    """
    Calculates various quantities for an orbit, 
    given the Keplerian elements
    """
    def __init__(self, P, M0, e, a, big_omega, little_omega, i, q=1.0, primary_mass=2.0*u.M_sun):
        """
        Initialize the OrbitCalculator class.

        Parameters
        ===========
        - P:            float, or astropy quantity for time.
                        Orbital period. Assumed to be in years if not an astropy quantity.

        - M0:           float, or astropy quantity for angle.
                        The mean anomaly at epoch. Assumed to be in degrees in not an
                        astropy quantity.

        - e:            float
                        Orbital eccentricity

        - a:            float, or astropy quantity for distance.
                        Semimajor axis. Assumed to be in AU if not an astropy quantity.

        - big_omega:    float, or astropy quantity for angle.
                        The longitude of the ascending node. Assumed to be in degrees 
                        if not as astropy quantity.

        - little_omega: float, or astropy quantity for angle.
                        The argument of periastron. Assumed to be in degrees 
                        if not as astropy quantity.

        - i:            float, or astropy quantity for angle.
                        Orbital inclination. Assumed to be in degrees 
                        if not as astropy quantity.

        - q:            float
                        The mass-ratio of the binary system.

        - primary_mass: float, or astropy quantity for mass
                        The mass of the primary star. Assumed to be in solar masses
                        if not an astropy quantity.
        """

        # Save most of the variables as instance variables for use in various functions.
        self.P = P if isinstance(P, u.Quantity) else P*u.year
        self.M0 = M0 if isinstance(M0, u.Quantity) else M0*u.degree
        self.e = e
        self.a = a if isinstance(a, u.Quantity) else a*u.AU
        self.big_omega = big_omega if isinstance(big_omega, u.Quantity) else big_omega*u.degree
        self.little_omega = little_omega if isinstance(little_omega, u.Quantity) else little_omega*u.degree
        self.primary_mass = primary_mass if isinstance(primary_mass, u.Quantity) else primary_mass*u.M_sun

        # Pre-compute sin(i) and cos(i), two useful quantities.
        inc = i if isinstance(i, u.Quantity) else i*u.degree
        self.sini = np.sin(inc)
        self.cosi = np.cos(inc)

        # Compute the orbit radial velocity semi-amplitude.
        self.K1 = (1+q) * self.sini * np.sqrt(constants.G * self.primary_mass * (1+q) / (self.a*(1-e**2))).to(u.km/u.s)
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


    def calculate_eccentric_anomaly(self, M, e):
        """
        Get the eccentric anomaly (E) from the mean anomaly (M) and orbital eccentricity (e)
        Uses the equation M = E - esinE
        """
        if HelperFunctions.IsListlike(M):
            return np.array([self.calculate_eccentric_anomaly(Mi, e).value for Mi in M])*u.radian
        
        M = M.to(u.radian).value

        func = lambda E: E - e*np.sin(E) - M
        dfunc = lambda E: 1.0 - e*np.cos(E)
        d2func = lambda E: e*np.sin(E)

        output = newton(func, np.pi, fprime=dfunc, fprime2=d2func)
        return output*u.radian


    def get_true_anomaly(self, time_since_epoch, ret_ecc_anomaly=False):
        """Get the true anomaly of the orbit some time after the epoch at which self.M0 is defined.

        Parameters:
        ===========
        - time_since_epoch: float, or astropy quantity for any time unit.
                            Gives the time since the epoch at which self.M0 is defined. Assumed
                            to be in years if not an astropy quantity instance.

        - ret_ecc_anomaly:  boolean
                            Flag for returning the eccentric anomaly as well as the true anomaly.
                            Default = False

        Returns:
        ========
        - True anomaly:     astropy.Quantity 
                            The true anomaly of the orbit, as an angle.
        """
        dt = time_since_epoch if isinstance(time_since_epoch, u.Quantity) else time_since_epoch*u.year
        M = self.M0 + (2*np.pi*dt / self.P).to(u.radian, equivalencies=u.dimensionless_angles())
        E = self.calculate_eccentric_anomaly(M, self.e)
        A = (np.cos(E) - self.e)/(1-self.e*np.cos(E))
        B = (np.sqrt(1.-self.e**2) * np.sin(E)) / (1.-self.e*np.cos(E))
        nu = np.arctan2(B, A)

        return (nu, E) if ret_ecc_anomaly else nu


    def get_rv(self, time_since_epoch, component='primary'):
        """ Get the radial velocity of the selected component, at the given time since epoch.

        Parameters:
        ===========
        - time_since_epoch: float, or astropy quantity for any time unit.
                            Gives the time since the epoch at which self.M0 is defined. Assumed
                            to be in years if not an astropy quantity instance.

        - component:        string
                            Which binary component to get the velocity of. Choices are 'primary'
                            and 'secondary', and 'primary' is the default.

        Returns:
        ========
        - Radial velocity:  astropy.Quantity 
                            The radial velocity of the chosen binary component.
        """
        nu = self.get_true_anomaly(time_since_epoch)
        K = self.K1 if component == 'primary' else self.K2
        return K * (np.cos(nu+self.little_omega) + self.e*np.cos(self.little_omega))


    def get_imaging_observables(self, time_since_epoch, distance=None, parallax=None):
        """ Get the separation and position angle of the star in the plane of the sky.

        Parameters:
        ===========
        - time_since_epoch: float, or astropy quantity for any time unit.
                            Gives the time since the epoch at which self.M0 is defined. Assumed
                            to be in years if not an astropy quantity instance.

        - distance :        float, or astropy quantity for distance.
                            The distance from Earth to the star. If not astropy quantity, assumed
                            to have units of parsecs. Either this or parallax MUST be given!

        - parallax :        float, or astropy quantity for angle.
                            The parallax of the star system. If not astropy quantity, 
                            assumed to have units of arcsec. Either this or distance 
                            MUST be given!

        Returns:
        ========
        - rho:              astropy.Quantity 
                            The angular separation between the primary and secondary star

        - theta:            astropy.Quantity
                            The position angle of the companion, in relation to the primary star.
        """
        # Calculate the cartesian coordinates, first, to get the quadrant right in theta.
        nu, E = self.get_true_anomaly(time_since_epoch, ret_ecc_anomaly=True)
        X = np.cos(E) - self.e
        Y = np.sin(E) * np.sqrt(1-self.e**2)
        x = self.A*X + self.F*Y
        y = self.B*X + self.G*Y

        # Convert to rho/theta
        #r = self.a * (1-self.e**2) / (1 + self.e*np.cos(nu))
        #rho = r*np.sqrt(x**2 + y**2)
        rho = self.a * (1 - self.e * np.cos(E))
        theta = self.big_omega + np.arctan2(y, x)

        return rho, theta




#class ProbabilisticCalculator(object):
#    """ This class assumes uniform distributions of the orbital orientation parameters, 
#    and samples a bunch of them to get probability of observables"""
#    def __init__(self, )





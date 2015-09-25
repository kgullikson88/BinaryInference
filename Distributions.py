import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm, gaussian_kde

import Fitters
import Mamajek_Table

MT = Mamajek_Table.MamajekTable()
teff2mass = MT.get_interpolator('Teff', 'Msun')
mass2teff = MT.get_interpolator('Msun', 'Teff')


class DistributionFitter(Fitters.Bayesian_LS):
    """
    Fit parameters to the mass-ratio, separation, and eccentricity distribution from
    a series of orbit fits. This class implements the likelihood function given in
    Foreman-Mackey et al (2014), equation 11.

    Parameters:
    ===========
    - mcmc_samples:   Numpy.ndarray of shape [Nstars, N_mcmc, 3]
                      One array to hold all of the mcmc samples for the orbital fits for all Nstars stars.
                      N_mcmc is the number of MCMC samples for each orbit fit.
                      The samples for each parameter (q, a, e) are:

                      >>> q = mcmc_samples[:, :, 0]
                      >>> a = mcmc_samples[:, :, 1]
                      >>> e = mcmc_samples[:, :, 2]

    - prior_fcn:      A callable that takes the arguments q, a, e
                      Returns the prior probability function for the semimajor axis, eccentricity, and mass-ratio.
                      This should encode the (normalized) priors you used to get the MCMC samples,
                      and is different from the prior on the distribution parameters (gamma, mu, sigma, and eta).

    - completeness_fcn: A callable that takes the arguments q, a, e
                        Returns the completeness fraction for those parameters. This is the Q(w) in Equation 11.

    - integral_fcn:     A callable that takes the arguments gamma, mu, sigma, eta
                        Should return the integral in Equation 11

    - malm_pars:        An iterable of any size
                        The iterable should parameterize a function
                        f(q) = malm_pars[0] + q*malm_pars[1] + q^2 * malm_pars[2] + ...
                        such that the probability of observing a mass-ratio q is given
                        (up to a normalization constant) by
                        P(q) = f(q)Gamma(q)
    """

    def __init__(self, mcmc_samples, prior_fcn=None, completeness_fcn=None, integral_fcn=None, malm_pars=(1.0,)):
        self.param_names = ['$\gamma$', '$\mu$', '$\sigma$', '$\eta$']
        self.n_params = len(self.param_names)
        self.q = mcmc_samples[:, :, 0]
        self.a = mcmc_samples[:, :, 1]
        self.e = mcmc_samples[:, :, 2]
        self.malm_pars = np.atleast_1d(malm_pars)
        self.prior = prior_fcn(self.q, self.a, self.e) if prior_fcn is not None else 1.0
        self.completeness = completeness_fcn(self.q, self.a, self.e) if completeness_fcn is not None else 1.0
        
        # Pre-compute logs
        self.lnq = np.log(self.q)
        self.lna = np.log(self.a)
        self.lne = np.log(self.e)
        self.lnp = np.log(self.prior)
        self.ln_completeness = np.log(self.completeness)

        # Register the integral function
        if integral_fcn is not None:
            self.integral_fcn = integral_fcn
        else:
            self.integral_fcn = self._setup_generic_integral_function()


    def _setup_generic_integral_function(self):
        """
            Use a self-compiled integral function. The details are available at
            https://gist.github.com/1288a4698ad5ff7a3640
        """
        import platform
        import os
        import ctypes
        from scipy.integrate import tplquad

        if 'linux' in platform.system().lower():
            lib_name = '{}/School/Research/libintegrand_linux.so'.format(os.environ['HOME'])
        else:
            lib_name = '{}/School/Research/libintegrand_macosx.so'.format(os.environ['HOME'])

        lib = ctypes.CDLL(lib_name)
        c_integrand = lib.integrand  # Assign specific function to name c_integrand (for simplicity)
        c_integrand.restype = ctypes.c_double
        c_integrand.argtypes = (ctypes.c_int, ctypes.c_double)

        return lambda gamma, mu, sigma, eta: tplquad(c_integrand, 0, 1,
                                                     lambda x: 0, lambda x: 1,
                                                     lambda x, y: -3, lambda x, y: 20,
                                                     args=(gamma, mu, sigma, eta))


    def _lnlike_normal(self, pars):
        gamma, mu, sigma, eta = pars
        Gamma_q = (1 - gamma) * self.q ** (-gamma)
        Gamma_e = (1 - eta) * self.e ** (-eta)
        Gamma_a = 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.log(self.a) - mu) ** 2 / sigma ** 2)

        #Gamma = Gamma_q * Gamma_e * Gamma_a * self.completeness / self.prior
        Gamma = Gamma_q * self.completeness / self.prior
        return np.sum(np.log(np.nanmean(Gamma, axis=1))) - self.integral_fcn(gamma, mu, sigma, eta)


    def _lnlike_stable(self, pars):
        gamma, mu, sigma, eta = pars
        ln_gamma_q = np.log(1 - gamma) - gamma * self.lnq
        ln_gamma_e = np.log(1-eta) - eta*self.lne - np.log(1.0 - 10**(-20*(1-eta)))
        ln_gamma_a = -0.5*(self.lna-mu)**2/sigma**2 - 0.5*np.log(2*np.pi*sigma**2)

        # Adjust for malmquist bias
        malm_func, denominator = self._malmquist(gamma)
        ln_gamma_q += np.log(malm_func(self.q)) - np.log(denominator)  # This could probably be made more efficient...

        ln_gamma = ln_gamma_q + ln_gamma_e + ln_gamma_a
        #ln_gamma = ln_gamma_q
        ln_summand = ln_gamma + self.ln_completeness - self.lnp
        N_k = self.q.shape[1] - np.isnan(self.q).sum(axis=1)
        return np.sum(np.log(np.nansum(np.exp(ln_summand), axis=1)) - np.log(N_k)) - self.integral_fcn(gamma, mu, sigma, eta)
        #return np.sum(np.log(np.nansum(ln_summand, axis=1)) - np.log(N_k)) - self.integral_fcn(gamma, mu, sigma, eta)  # GIVES NANS ALWAYS


    def _malmquist(self, gamma):
        """ Return the malmquist adjustment function as well as the normalization constant
        """
        func = np.poly1d(self.malm_pars[::-1])
        denom = np.sum([p * (1.0 - gamma) / (i + 1 - gamma) for i, p in enumerate(self.malm_pars)])
        return func, denom


    
    def _lnlike(self, pars):
        return self._lnlike_stable(pars)


    def lnprior(self, pars):
        gamma, mu, sigma, eta = pars
        if gamma < 1 and mu > 0 and sigma > 0 and eta < 1:
            return 0.0
        return -np.inf

    def mnest_prior(self, cube, ndim, nparams):
        cube[1] = cube[1] * 10  # Uniform in mean (log) separation
        cube[2] = cube[2] * 10 + 1e-3  # Uniform in (log) separation spread
        # cube[0] and cube[3] encode the gamma and eta parameters, which are uniform on [0,1]
        return

    def guess_fit_parameters(self):
        """
        Do a normal (non-bayesian) fit to the data.
        The result will be saved for use as initial guess parameters in the full MCMC fit.
        """

        def errfcn(pars):
            lnl = -self._lnlike(pars)
            p = list(pars)
            p.append(lnl)
            logging.info(p)
            return lnl if np.isfinite(lnl) else np.sign(lnl) * 9e9

        initial_pars = [0.5, 5, 5, 0.5]
        out = minimize(errfcn, initial_pars, bounds=[[0, 0.999], [0, 10], [1e-3, 10], [0, 0.999]])
        self.guess_pars = out.x
        return out.x


class OrbitPrior(object):
    """ Object to compute the prior on my parameters, including the empirical mass-ratio distribution prior
        from the companion temperature.
        TODO: This should be able to take 2d arrays for M1_vals and T2_vals, and then forego the random sampling
              (i.e. give it samples from the primary star mass and companion temperature using whatever distributions
              I want).
        TODO: Allow user to give custom function for teff2mass (using evolutionary tracks or something)
    """

    def __init__(self, M1_vals, T2_vals, N_samp=10000, gamma=0.4):
        """Initialize the orbit prior object

        Parameters:
        ===========
        - M1_vals:     numpy array, or float
                       The primary star masses

        - T2_vals:     numpy array of same shape as M1_vals, or float
                       The companion star temperatures

        - N_samp:      The number of random samples to take for computing the mass-ratio distribution samples

        - gamma:       The mass-ratio distribution power-law exponent

        Returns:
        =========
        None
        """
        M1_vals = np.atleast_1d(M1_vals)
        T2_vals = np.atleast_1d(T2_vals)

        # Estimate the mass-ratio prior
        M1_std = np.maximum(0.5, 0.2 * M1_vals)
        a, b = (1.5 - M1_vals) / M1_std, np.ones_like(M1_vals) * np.inf
        M1_samples = np.array(
            [truncnorm.rvs(a=a[i], b=b[i], loc=M1_vals[i], scale=M1_std[i], size=N_samp) for i in range(M1_vals.size)])
        T2_samples = np.array([np.random.normal(loc=T2_vals[i], scale=200, size=N_samp) for i in range(T2_vals.size)])
        M2_samples = teff2mass(T2_samples)
        q_samples = M2_samples / M1_samples

        self.empirical_q_prior = [gaussian_kde(q_samples[i, :]) for i in range(q_samples.shape[0])]
        self.gamma = gamma

    def _evaluate_empirical_q_prior(self, q):
        q = np.atleast_1d(q)
        assert q.shape[0] == len(self.empirical_q_prior)
        return np.array([self.empirical_q_prior[i](q[i]) for i in range(q.shape[0])])

    def evaluate(self, q, a, e):
        empirical_prior = self._evaluate_empirical_q_prior(q)
        return (1 - self.gamma) * q ** (-self.gamma) * empirical_prior / (120 * a * e * np.log(10) ** 2)

    def __call__(self, q, a, e):
        return self.evaluate(q, a, e)


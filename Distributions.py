import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm, gaussian_kde
from scipy.integrate import quad

import Mamajek_Table
import fitters

MT = Mamajek_Table.MamajekTable()
teff2mass = MT.get_interpolator('Teff', 'Msun')
mass2teff = MT.get_interpolator('Msun', 'Teff')


class DistributionFitter(fitters.Bayesian_LS):
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

    - prior_fcn:      A callable that takes the arguments lnq, lna, lne
                      Returns the log-prior probability function for the mass-ratio, semimajor axis, and eccentricity.
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

    - fix_bin_frac:     boolean, default=True 
                        Should the overall binary fraction be fixed to 1.0? If not, you can estimate it but it is
                        rather degenerate with the mass-ratio distribution, especially in a malmquist-biased sample!

    - low_q:            float, default=0.0
                        What is the lowest mass ratio in the sample? (Used for normalized the PDF)

    - high_q:           float, default=1.0
                        What is the highest mass ratio in the sample? (Used for normalizing the PDF)
    """

    def __init__(self, mcmc_samples, prior_fcn=None, completeness_fcn=None, integral_fcn=None, malm_pars=(1.0,),
                 fix_bin_frac=True, low_q=0.0, high_q=1.0):
        if fix_bin_frac:
            self.param_names = ['$\gamma$', '$\mu$', '$\sigma$', '$\eta$']
        else:
            self.param_names = [r'$f_{\rm bin}$', '$\gamma$', '$\mu$', '$\sigma$', '$\eta$']

        self.vary_bin_frac = not fix_bin_frac
        self.n_params = len(self.param_names)
        self.q = mcmc_samples[:, :, 0]
        self.a = mcmc_samples[:, :, 1]
        self.e = mcmc_samples[:, :, 2]
        self.malm_pars = np.atleast_1d(malm_pars)
        
        # Pre-compute logs
        self.lnq = np.log(self.q)
        self.lna = np.log(self.a)
        self.lne = np.log(self.e)

        # Compute the prior and completeness (neither depend on the parameters)
        self.lnp = prior_fcn(self.lnq, self.lna, self.lne) if prior_fcn is not None else 0.0
        self.completeness = completeness_fcn(self.q, self.a, self.e) if completeness_fcn is not None else 1.0
        self.ln_completeness = np.log(self.completeness)

        # Compute the number of MCMC samples for each star's orbit fit
        self.N_k = self.q.shape[1] - np.isnan(self.q).sum(axis=1)
        self.good_idx = self.N_k > 0

        # Register the integral function
        if integral_fcn is not None:
            self.integral_fcn = integral_fcn
        else:
            self.integral_fcn = self._setup_generic_integral_function()

        # Save the q-limits
        self.low_q = low_q
        self.high_q = high_q


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



    def _lnlike_stable(self, pars):
        if self.vary_bin_frac:
            f_bin, gamma, mu, sigma, eta = pars
        else:
            gamma, mu, sigma, eta = pars
            f_bin = 1.0
        ln_gamma_q = np.log(1 - gamma) - np.log(
            self.high_q ** (1 - gamma) - self.low_q ** (1 - gamma)) - gamma * self.lnq
        ln_gamma_e = np.log(1-eta) - eta*self.lne
        ln_gamma_a = -0.5*(self.lna-mu)**2/sigma**2 - 0.5*np.log(2*np.pi*sigma**2) - self.lna

        # Adjust for malmquist bias
        malm_func, denominator = self._malmquist(gamma)
        ln_gamma_q += np.log(malm_func(self.q)) - np.log(denominator)  # This could probably be made more efficient...

        ln_gamma = ln_gamma_q + ln_gamma_e + ln_gamma_a + np.log(f_bin)
        ln_summand = ln_gamma + self.ln_completeness - self.lnp
        
        summation = np.nansum(np.exp(ln_summand[self.good_idx]), axis=1)
        #summation[np.isinf(summation)] = 1e300
        return np.sum(np.log(summation) - np.log(self.N_k[self.good_idx])) - self.integral_fcn(f_bin, gamma, mu, sigma, eta, self.malm_pars)

        #return np.sum(np.log(np.nansum(np.exp(ln_summand[self.good_idx]), axis=1)) - np.log(self.N_k[self.good_idx])) - \
        #       self.integral_fcn(f_bin, gamma, mu, sigma, eta, self.malm_pars)
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
        if self.vary_bin_frac:
            f_bin, gamma, mu, sigma, eta = pars
        else:
            gamma, mu, sigma, eta = pars
            f_bin = 1.0
        if 0 <= f_bin <= 1 and gamma < 1 and mu > 0 and sigma > 0 and eta < 1:
            return 0.0
        return -np.inf

    def mnest_prior(self, cube, ndim, nparams):
        cube[1] = cube[1] * 20 + 1e-3  # Uniform in mean (log) separation
        cube[2] = cube[2] * 10 + 1e-3  # Uniform in (log) separation spread
        # cube[0] and cube[3] encode the gamma and eta parameters, which are uniform on [0,1]
        return

    def guess_fit_parameters(self):
        """
        Do a normal (non-bayesian) fit to the data.
        The result will be saved for use as initial guess parameters in the full MCMC fit.
        """

        def errfcn(pars):
            lnl = -self._lnprob(pars)
            p = list(pars)
            p.append(lnl)
            logging.info(p)
            return lnl if np.isfinite(lnl) else np.sign(lnl) * 9e9

        if self.vary_bin_frac:
            initial_pars = [0.5, 0.5, 5, 5, 0.5]
            bounds_list = [[0.0, 1.0], [0, 0.999], [0, 100], [1e-3, 100], [0, 0.999]]
        else:
            initial_pars = [0.5, 5, 5, 0.5]
            bounds_list = [[0, 0.999], [0, 100], [1e-3, 100], [0, 0.999]]
        out = minimize(errfcn, initial_pars, bounds=bounds_list)
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

    def __init__(self, M1_vals, T2_vals, N_samp=10000, gamma=0.4, cache=False, interpolate_qprior=False):
        """Initialize the orbit prior object

        Parameters:
        ===========
        - M1_vals:     numpy array, or float
                       The primary star masses

        - T2_vals:     numpy array of same shape as M1_vals, or float
                       The companion star temperatures

        - N_samp:      The number of random samples to take for computing the mass-ratio distribution samples

        - gamma:       The mass-ratio distribution power-law exponent

        - cache:       boolean
                       Should we cache the empirical prior to make lookups faster?
                       If the input q changes, this will give THE WRONG ANSWER!

        - interpolate_qprior: boolean
                              Interpolate the q-prior to make a faster function call at the expense of some accuracy?

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
        #q_samples[q_samples > 1] = 1.0/q_samples[q_samples > 1]

        if interpolate_qprior:
            from scipy.interpolate import InterpolatedUnivariateSpline as spline
            q_vals = np.arange(0.01, 2.0, 0.01)
            self.empirical_q_prior = [spline(q_vals, gaussian_kde(q_samples[i, :])(q_vals)) for i in range(q_samples.shape[0])]
        else:
            # self.empirical_q_prior = [gaussian_kde(q_samples[i, :]) for i in range(q_samples.shape[0])]
            self.empirical_q_prior = gaussian_kde(q_samples)
        self.gamma = gamma
        self._cache_empirical = cache
        self._cache = None

    def _evaluate_empirical_q_prior(self, q):
        if self._cache_empirical and self._cache is not None:
            return self._cache
        q = np.atleast_1d(q)
        # assert q.shape[0] == len(self.empirical_q_prior)
        #emp_prior = np.array([self.empirical_q_prior[i](q[i]) for i in range(q.shape[0])])
        emp_prior = self.empirical_q_prior(q)
        if self._cache_empirical:
            self._cache = emp_prior
        return emp_prior

    def evaluate(self, q, a, e):
        empirical_prior = self._evaluate_empirical_q_prior(q)
        return (1 - self.gamma) * q ** (-self.gamma) * empirical_prior / (200 * a * e * np.log(10) ** 2)

    def log_evaluate(self, lnq, lna, lne):
        empirical_prior = np.log(self._evaluate_empirical_q_prior(np.exp(lnq)))
        return empirical_prior + np.log(1-self.gamma) - self.gamma*lnq - np.log(200*np.log(10)**2) - lna - lne 

    def __call__(self, lnq, lna, lne):
        return self.log_evaluate(lnq, lna, lne)


class CensoredCompleteness(object):
    """ A class for calculating the completeness function that varies based on the star
        In this case, it assumes each star has a completeness function of shape
    """

    def __init__(self, alpha_vals, beta_vals):
        """
        A helper function for calculating the completeness function and corresponding integral.
        The completeness is defined as:

        .. math::
            Q(q|\alpha, \beta) = \frac{1}{1+e^{-\alpha (q-\beta)}}

        Parameters:
        ============
         - alpha_vals:  An iterable of length N
                        Holds all the values for alpha

         - beta_vals:   An iterable of length N
                        Holds all the values for beta
        """
        self.alpha_vals = np.atleast_1d(alpha_vals)
        self.beta_vals = np.atleast_1d(beta_vals)
        assert len(alpha_vals) == len(beta_vals), 'alpha_vals and beta_vals must be the same length!'

        import ctypes
        import os

        try:
            lib = ctypes.CDLL('{}/School/Research/BinaryInference/integrandlib.so'.format(os.environ['HOME']))
        except OSError:
            lib = ctypes.CDLL('{}/integrandlib.so'.format(os.getcwd()))
        self.c_integrand = lib.q_integrand_logisticQ_malmquist  # Assign specific function to name c_integrand (for simplicity)
        self.c_integrand.restype = ctypes.c_double
        self.c_integrand.argtypes = (ctypes.c_int, ctypes.c_double)


    @classmethod
    def sigmoid(cls, q, alpha, beta):
        return 1.0 / (1.0 + np.exp(-alpha * (q - beta)))

    def integral(self, f_bin, gamma, mu, sigma, eta, malm_pars=np.array([1.])):
        """
        Returns the integral normalization factor in Equation 11

        Parameters:
        ===========
         - f_bin:    float
                     The overall binary fraction

         - gamma:    float
                     The mass-ratio power law exponent

         - mu:       float
                     The semimajor-axis log-normal mean

         - sigma:    float
                     The semimajor-axis log-normal width

         - eta:      float
                     The eccentricity power law exponent

        Returns:
        =========
         float - the value of the integral for the input set of parameters
        """
        s = 0.0
        for alpha, beta in zip(self.alpha_vals, self.beta_vals):
            arg_list = [gamma, alpha, beta, len(malm_pars)]
            arg_list.extend(malm_pars)
            s += quad(self.c_integrand, 0, 1, args=tuple(arg_list))[0]
        return s*f_bin
        #return f_bin * np.sum([quad(self.c_integrand, 0, 1, args=arg_list)[0] for alpha, beta in
        #               zip(self.alpha_vals, self.beta_vals)])


    def __call__(self, q, a, e):
        """
        Gives the overall completeness over the whole sample

        Parameters:
        ===========
         - q:    float, or numpy.ndarray
                 The mass-ratio

         - a:    float, or numpy.ndarray
                 The semimajor-axis (in AU)

         - e:    float, or numpy.ndarray
                 The eccentricity

        Returns:
        =========
         float, or numpy.ndarray of the same shape as the inputs, containing the completeness
        """
        completeness = np.zeros_like(q)
        for alpha, beta in zip(self.alpha_vals, self.beta_vals):
            completeness += self.sigmoid(q, alpha, beta)
        return completeness

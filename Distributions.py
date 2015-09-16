import logging

import numpy as np
from scipy.optimize import minimize

import Fitters


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
    
    - q_limits:         An iterable of length 2
                        Gives the hard limits for the mass-ratio function. Defaults to (0,1).
    """

    def __init__(self, mcmc_samples, prior_fcn=None, completeness_fcn=None, integral_fcn=None, q_limits=(0.0, 1.0)):
        self.param_names = ['$\gamma$', '$\mu$', '$\sigma$', '$\eta$']
        self.n_params = len(self.param_names)
        self.q = mcmc_samples[:, :, 0]
        self.a = mcmc_samples[:, :, 1]
        self.e = mcmc_samples[:, :, 2]
        self.prior = prior_fcn(self.q, self.a, self.e) if prior_fcn is not None else 1.0
        self.completeness = completeness_fcn(self.q, self.a, self.e) if completeness_fcn is not None else 1.0
        
        # Pre-compute logs
        self.lnq = np.log(self.q)
        self.lna = np.log(self.a)
        self.lne = np.log(self.e)
        self.lnp = np.log(self.prior)
        self.ln_completeness = np.log(self.completeness)
        self.q_limits = q_limits[:2]

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
        #low = 0.05
        #high = 0.8
        #high = 1.0
        low, high = self.q_limits
        ln_gamma_q = np.log(1 - gamma) - gamma*self.lnq - np.log(high**(1-gamma) - low**(1-gamma))
        ln_gamma_e = np.log(1-eta) - eta*self.lne
        ln_gamma_a = -0.5*(self.lna-mu)**2/sigma**2 - 0.5*np.log(2*np.pi*sigma**2)

        ln_gamma = ln_gamma_q + ln_gamma_e + ln_gamma_a
        #ln_gamma = ln_gamma_q
        ln_summand = ln_gamma + self.ln_completeness - self.lnp
        N_k = self.q.shape[1] - np.isnan(self.q).sum(axis=1)
        return np.sum(np.log(np.nansum(np.exp(ln_summand), axis=1)) - np.log(N_k)) - self.integral_fcn(gamma, mu, sigma, eta)
        #return np.sum(np.log(np.nansum(ln_summand, axis=1)) - np.log(N_k)) - self.integral_fcn(gamma, mu, sigma, eta)  # GIVES NANS ALWAYS

    
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


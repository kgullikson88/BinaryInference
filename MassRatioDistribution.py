import logging
from george import kernels, GP

import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm, gaussian_kde
from scipy.integrate import quad

from kglib.utils.HelperFunctions import BinomialErrors
from kglib.spectral_type import Mamajek_Table
from kglib import fitters


MT = Mamajek_Table.MamajekTable()
teff2mass = MT.get_interpolator('Teff', 'Msun')
mass2teff = MT.get_interpolator('Msun', 'Teff')

class GammaFitter(fitters.Bayesian_LS):
    """
    Fit parameters to the mass-ratio, separation, and eccentricity distribution from
    a series of orbit fits. This class implements the likelihood function given in
    Foreman-Mackey et al (2014), equation 11.

    Parameters:
    ===========
    - mcmc_samples:   Numpy.ndarray of shape [Nstars, N_mcmc]
                      One array to hold all of the mcmc samples for the orbital fits for all Nstars stars.
                      N_mcmc is the number of MCMC samples for each orbit fit.
                      The values should be the mass-ratio mcmc samples.

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

    - Pobs:             float, default=0.00711310498183
                        The probability of observing a star, given that it is NOT a binary system.
                        The default value was calculated in the Malmquist.ipynb notebook.
    """

    def __init__(self, mcmc_samples, prior_fcn=None, completeness_fcn=None, integral_fcn=None, malm_pars=(1.0,),
                 fix_bin_frac=True, low_q=0.0, high_q=1.0, Pobs=0.00711310498183):
        if fix_bin_frac:
            self.param_names = ['$\gamma$']
        else:
            self.param_names = [r'$f_{\rm bin}$', '$\gamma$']

        self.vary_bin_frac = not fix_bin_frac
        self.n_params = len(self.param_names)
        self.q = mcmc_samples
        self.malm_pars = np.atleast_1d(malm_pars)
        self.Pobs = Pobs
        
        # Pre-compute logs
        self.lnq = np.log(self.q)

        # Compute the prior and completeness (neither depend on the parameters)
        self.lnp = prior_fcn(self.lnq) if prior_fcn is not None else 0.0
        self.completeness = completeness_fcn(self.q) if completeness_fcn is not None else 1.0
        self.ln_completeness = np.log(self.completeness)

        # Compute the number of MCMC samples for each star's orbit fit
        self.N_k = self.q.shape[1] - np.isnan(self.q).sum(axis=1)
        self.good_idx = self.N_k > 0

        # Register the integral function
        if integral_fcn is not None:
            self.integral_fcn = integral_fcn
        else:
            self.integral_fcn = self._setup_generic_integral_function()
        
        self.low_q = low_q
        self.high_q = high_q


    def _setup_generic_integral_function(self):
        """
            Use a self-compiled integral function. The details are available at
            https://gist.github.com/1288a4698ad5ff7a3640
        """
        raise NotImplementedError


    def _lnlike_plain(self, pars):
        if self.vary_bin_frac:
            f_bin, gamma = pars
        else:
            gamma = pars
            f_bin = 1.0

        gamma_q = (1-gamma) / (self.high_q ** (1 - gamma) - self.low_q ** (1 - gamma)) * self.q**(-gamma)
        malm_func, denominator = self._malmquist(gamma)
        gamma_q *= malm_func(self.q) / denominator
        summand = gamma_q * f_bin * self.completeness / np.exp(self.lnp)
        summation = np.nanmean(summand[self.good_idx], axis=1)
        return np.sum(np.log(summation)) - self.integral_fcn(f_bin, gamma, self.malm_pars)

    def _lnlike_stable(self, pars):
        if self.vary_bin_frac:
            f_bin, gamma = pars
        else:
            gamma = pars
            f_bin = 1.0
        logging.debug('f_bin, gamma = {:.3f}, {:.3f}'.format(f_bin, gamma))
        #ln_gamma_q = (np.log(1 - gamma)
        #              - np.log(self.high_q ** (1 - gamma) - self.low_q ** (1 - gamma))
        #              - gamma * self.lnq)

        # Adjust for malmquist bias
        #malm_func, denominator = self._malmquist(gamma)
        #ln_gamma_q += np.log(malm_func(self.q)) - np.log(denominator)  # This could probably be made more efficient...
        #if self.malm_pars.size > 1 or self.malm_pars != 1:
            # malmquist-correct the binary fraction
        #    f_bin = f_bin * denominator / (f_bin * denominator + (1 - f_bin) * self.Pobs)
        #logging.debug('Modified f_bin, Pobs|binary, Pobs|not binary = {}, {}, {}'.format(f_bin, denominator, self.Pobs))
        
        # Get the malmquist-adjusted log-rate density
        ln_gamma = self._malmquist_q_lngamma(gamma, f_bin)

        #ln_gamma = ln_gamma_q + np.log(f_bin)
        ln_summand = ln_gamma + self.ln_completeness - self.lnp
        summation = np.nanmean(np.exp(ln_summand[self.good_idx]), axis=1)
        return np.sum(np.log(summation)) - self.integral_fcn(f_bin, gamma, malm_pars=self.malm_pars, Pobs=self.Pobs)

    def _malmquist_q_lngamma(self, gamma, f_bin):
        """ Return the rate density for q, including f_bin and malmquist bias
        """
        func = np.poly1d(self.malm_pars[::-1])
        const_factor = (1 - gamma) / (self.high_q ** (1 - gamma) - self.low_q ** (1 - gamma))
        integral = np.sum(
            [p * const_factor / (i + 1 - gamma) * (self.high_q ** (i + 1 - gamma) - self.low_q ** (i + 1 - gamma))
             for i, p in enumerate(self.malm_pars)])
        Pobs = integral if self.malm_pars.size == 1 and self.malm_pars[0] == 1 else self.Pobs
        denom = f_bin*integral + (1-f_bin)*Pobs
        logging.debug('Denominator = {}\nIntegral = {}\nPobs = {}\n'.format(denom, integral, Pobs))
        return np.log(func(self.q)) + np.log(f_bin) + np.log(const_factor) - gamma*self.lnq - np.log(denom)

    def _malmquist(self, gamma):
        """ Return the malmquist adjustment function as well as the normalization constant
        """
        func = np.poly1d(self.malm_pars[::-1])
        const_factor = (1 - gamma) / (self.high_q ** (1 - gamma) - self.low_q ** (1 - gamma))
        denom = np.sum(
            [p * const_factor / (i + 1 - gamma) * (self.high_q ** (i + 1 - gamma) - self.low_q ** (i + 1 - gamma))
             for i, p in enumerate(self.malm_pars)])
        return func, denom


        
    def lnlike(self, pars):
        return self._lnlike_stable(pars)


    def lnprior(self, pars):
        if self.vary_bin_frac:
            f_bin, gamma = pars
        else:
            gamma = pars
            f_bin = 1.0
        if 0 <= f_bin <= 1 and gamma < 1:
            return 0.0
        return -np.inf

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
            initial_pars = [0.5, 0.5]
            bounds_list = [[0.0, 1.0], [0, 0.999]]
        else:
            initial_pars = [0.5]
            bounds_list = [[0, 0.999]]
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

    def __init__(self, M1_vals, T2_vals, N_samp=10000, gamma=0.4, cache=False, low_q=0.0, high_q=1.0):
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

        - low_q:       float, default=0.0
                       What is the lowest mass ratio in the sample? (Used for normalized the PDF)

        - high_q:      float, default=1.0
                       What is the highest mass ratio in the sample? (Used for normalizing the PDF)

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
        self._cache_empirical = cache
        self._cache = None
        self.low_q = low_q
        self.high_q = high_q

    def _evaluate_empirical_q_prior(self, q, clip=1e-40):
        if self._cache_empirical and self._cache is not None:
            return self._cache
        q = np.atleast_1d(q)
        assert q.shape[0] == len(self.empirical_q_prior)
        emp_prior = np.array([self.empirical_q_prior[i](q[i]) for i in range(q.shape[0])])
        if self._cache_empirical:
            self._cache = emp_prior
        emp_prior[emp_prior < clip] = clip
        return emp_prior

    def evaluate(self, q):
        empirical_prior = self._evaluate_empirical_q_prior(q)
        return (1 - self.gamma) * q ** (-self.gamma) * empirical_prior / (self.high_q**(1-self.gamma) - self.low_q**(1-self.gamma))

    def log_evaluate(self, lnq):
        empirical_prior = np.log(self._evaluate_empirical_q_prior(np.exp(lnq)))
        return empirical_prior + np.log(1-self.gamma) - np.log(self.high_q**(1-self.gamma) - self.low_q**(1-self.gamma)) - self.gamma*lnq 

    def __call__(self, lnq):
        return self.log_evaluate(lnq)


class CensoredCompleteness(object):
    """ A class for calculating the completeness function that varies based on the star
        In this case, it assumes each star has a completeness function of shape
    """

    def __init__(self, alpha_vals, beta_vals, low_q=0.0, high_q=1.0):
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

         - low_q:       float, default=0.0
                        What is the lowest mass ratio in the sample? (Used for normalized the PDF)

         - high_q:      float, default=1.0
                        What is the highest mass ratio in the sample? (Used for normalizing the PDF)
        """
        self.alpha_vals = np.atleast_1d(alpha_vals)
        self.beta_vals = np.atleast_1d(beta_vals)
        self.low_q = low_q
        self.high_q = high_q
        assert len(alpha_vals) == len(beta_vals), 'alpha_vals and beta_vals must be the same length!'

        import ctypes
        import os

        try:
            lib = ctypes.CDLL('{}/School/Research/BinaryInference/integrandlib.so'.format(os.environ['HOME']))
        except OSError:
            lib = ctypes.CDLL('{}/integrandlib.so'.format(os.getcwd()))
        self.c_integrand = lib.q_integrand_logisticQ_malmquist_cutoff  # Assign specific function to name c_integrand (for simplicity)
        self.c_integrand.restype = ctypes.c_double
        self.c_integrand.argtypes = (ctypes.c_int, ctypes.c_double)


    @classmethod
    def sigmoid(cls, q, alpha, beta):
        return 1.0 / (1.0 + np.exp(-alpha * (q - beta)))

    def integral(self, f_bin, gamma, malm_pars=np.array([1.]), Pobs=0.00711310498183):
        """
        Returns the integral normalization factor in Equation 11

        Parameters:
        ===========
         - f_bin:    float
                     The overall binary fraction

         - gamma:    float
                     The mass-ratio power law exponent

        Returns:
        =========
         float - the value of the integral for the input set of parameters
        """
        s = 0.0
        for alpha, beta in zip(self.alpha_vals, self.beta_vals):
            #arg_list = [gamma, alpha, beta, self.low_q, self.high_q, len(malm_pars)]
            arg_list = [gamma, f_bin, Pobs, alpha, beta, 0.0, 1.0, len(malm_pars)]
            arg_list.extend(malm_pars)
            s += quad(self.c_integrand, self.low_q, self.high_q, args=tuple(arg_list))[0]
        #return s*f_bin
        return s
        #return f_bin * np.sum([quad(self.c_integrand, 0, 1, args=arg_list)[0] for alpha, beta in
        #               zip(self.alpha_vals, self.beta_vals)])


    def __call__(self, q):
        """
        Gives the overall completeness over the whole sample

        Parameters:
        ===========
         - q:    float, or numpy.ndarray
                 The mass-ratio

        Returns:
        =========
         float, or numpy.ndarray of the same shape as the inputs, containing the completeness
        """
        completeness = np.zeros_like(q)
        for alpha, beta in zip(self.alpha_vals, self.beta_vals):
            completeness += self.sigmoid(q, alpha, beta)
        return completeness




class Hist(object):
    def __init__(self, x, bin_edges, Nsamp):
        """
        Calculate the raw histogram, the completeness-corrected one, and the malmquist-corrected one.
        """
        raw_vals, bin_edges = np.histogram(x, bins=bin_edges, normed=False)
        self.bin_edges = bin_edges
        self.bin_widths = np.diff(self.bin_edges)
        self.bin_centers = 0.5*(self.bin_edges[:-1] + self.bin_edges[1:])
        
        P, low, high = np.array([BinomialErrors(v, Nsamp) for v in raw_vals]).T
        self.raw_vals = P
        self.raw_low = low
        self.raw_high = high
        self.complete_vals = None
        self.malm_vals = None
        return 
    
    def completeness(self, completeness_fcn=None, completeness_integrals=None):
        """
        Correct the raw histogram for completeness effects.
        
        Parameters:
        ===========
        - completeness_fcn:        callable
                                   a function that takes an 'x' value, and returns the completenss for that x
                                   
        - completenss_integrals:   iterable of size N_bins
                                   Pre-computed integrals for each bin. Useful if the completeness
                                   function is analytically integrable.
        """
        if completeness_integrals is None:
            from scipy.integrate import quad
            completeness_integrals = [quad(completeness_fcn, x0, x1, maxp1=200)[0]/(x1-x0) for x0, x1 in 
                                      zip(self.bin_edges[:-1], self.bin_edges[1:])]
            logging.debug(completeness_integrals)
        
        self.complete_vals = np.array([v/I for v, I in zip(self.raw_vals, completeness_integrals)])
        self.complete_low = np.array([v/I for v, I in zip(self.raw_low, completeness_integrals)])
        self.complete_high = np.array([v/I for v, I in zip(self.raw_high, completeness_integrals)])
        return self.complete_vals - self.raw_vals
    
    
    def _fit_thetas(self, raw_thetas, malm_integrals):
        thetas = np.array(raw_thetas)
        thetas[:-1] = np.array([malm_integrals[-1]/Mi * p for Mi, p in zip(malm_integrals[:-1], raw_thetas[:-1])])
        return thetas
    
        
    def _integrate_malmquist(self, malm_pars, q0, q1):
        """ Integrate the malmquist-correction factor from q0 --> q1
        """
        return np.sum([p/(i+1.0)*(q1**(i+1.0) - q0**(i+1.0)) for i, p in enumerate(malm_pars)])
    

    def malmquist(self, malm_pars, correct_errors=True):
        """
        Correct the histogram for malmquist bias. If your have already completeness-corrected the raw
        histogram, it will use that. Otherwise, it will just use the raw histogram values.
        
        Parameters:
        ===========
        - malm_pars:      iterable
                          polynomial parameters describing the malmquist correction. 
                          They should be in order of the coefficients of increasing order
                          (i.e. malm_pars[0] + malm_pars[1]*x + ...)
        
        -Pobs:            float:
                          The probability of observation, given that the star is single.
        
        """
        
        # First, calculate the malmquist integrals for every bin
        malm_integrals = np.array([self._integrate_malmquist(malm_pars, q0, q1) for q0, q1 
                                   in zip(self.bin_edges[:-1], self.bin_edges[1:])])
        malm_integrals = np.array(malm_integrals)
        
        raw_thetas = self.raw_vals if self.complete_vals is None else self.complete_vals
        self.malm_vals = self._fit_thetas(raw_thetas, malm_integrals)
        norm = np.sum(self.malm_vals * self.bin_widths) / np.sum(raw_thetas * self.bin_widths)
        self.malm_vals /= norm
        
        # Adjust the errors too
        if correct_errors:
            logging.info('Correcting the lower bound')
            err = self.raw_low if self.complete_vals is None else self.complete_low
            self.malm_low = self._fit_thetas(err, malm_integrals) / norm
            logging.info('Correcting the upper bound')
            err = self.raw_high if self.complete_vals is None else self.complete_high
            self.malm_high = self._fit_thetas(err, malm_integrals) / norm
        return self.malm_vals - raw_thetas
    
    def plot(self, heights=None, ax=None, **hist_kwargs):
        """ Make a histogram. Uses a malmquist-corrected histogram if available, then tries
            a completeness-corrected one, and finally uses the raw histogram (if heights is None)
        """
        if heights is not None:
            vals = heights
        elif self.malm_vals is not None:
            vals = self.malm_vals
        elif self.complete_vals is not None:
            vals = self.complete_vals
        else:
            vals = self.raw_vals
        
        if ax is None:
            import matplotlib.pyplot as plt 
            fig, ax = plt.subplots(1, 1)
        return ax.bar(left=self.bin_edges[:-1], height=vals, width=self.bin_widths, **hist_kwargs)


class HistFitter(fitters.Bayesian_LS):
    def __init__(self, qvals, bin_edges):
        """
        Histogram Inference a la Dan Foreman-Mackey

        Parameters:
        ===========
        - qvals:      numpy array of shape (Nobs, Nsamples)
                      The MCMC samples for the mass-ratio distribution of all companions

        - bin_edges:  numpy array
                      The edges of the histogram bins to use.

        """
        self.qvals = qvals
        self.bin_edges = bin_edges
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = np.diff(self.bin_edges)
        self.Nbins = self.bin_widths.size
        self.Nobs = self.qvals.shape[0]

        # Find which bin each q falls in
        self.bin_idx = np.digitize(self.qvals, self.bin_edges) - 1

        # Determine the censoring function for each bin (used in the integral)
        self.censor_integrals = np.array([quad(func=self.censoring_fcn,
                                               a=left, b=right)[0] for (left, right) in
                                          zip(self.bin_edges[:-1], self.bin_edges[1:])])

        # Set values needed for multinest fitting
        self.n_params = self.Nbins
        self.param_names = [r'$\theta_{}$'.format(i) for i in range(self.Nbins)]


    def lnlike(self, pars):
        # Pull theta out of pars
        theta = pars[:self.Nbins]

        # Normalize theta
        # theta /= np.sum(theta * self.bin_widths)

        # Generate the inner summation
        gamma = np.ones_like(self.bin_idx) * np.nan
        good = (self.bin_idx < self.Nbins) & (self.bin_idx >= 0)  # nans in q get put in nonexistent bins
        gamma[good] = self.Nobs * self.censoring_fcn(self.qvals[good]) * theta[self.bin_idx[good]]
        summation = np.nanmean(gamma, axis=1)

        # Calculate the integral
        I = self._integral_fcn(theta)

        # Generate the log-likelihood
        ll = -I + np.nansum(np.log(summation))
        return ll


    def lnprior(self, pars):
        # Pull theta out of pars
        theta = pars[:self.Nbins]

        return 0.0


    def lnprob(self, pars):
        lp = self.lnprior(pars)
        return lp + self.lnlike(pars) if np.isfinite(lp) else -np.inf


    def _integral_fcn(self, theta):
        return np.sum(theta * self.censor_integrals) * self.Nobs


    def censoring_fcn(self, q):
        """
        Censoring function. This should take a mass-ratio (or array of mass-ratios), and return the completeness
        as a number between 0 and 1.
        """
        return 1.0


    def guess_fit(self):
        from scipy.optimize import minimize

        def errfcn(pars):
            ll = self.lnprob(pars)
            return -ll

        initial_guess = np.ones_like(self.bin_centers)
        bounds = [[1e-3, None] for p in initial_guess]
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x


    def mnest_prior(self, cube, ndim, nparams):
        for i in range(self.Nbins):
            cube[i] *= 10

        return


class CensoredHistFitter(HistFitter):
    def censoring_fcn(self, q, alpha=40, beta=0.25):
        # sigmoid censoring function. Change this for the real deal!
        return 1.0 / (1.0 + np.exp(-alpha * (q - beta)))


class SmoothHistFitter(CensoredHistFitter):
    """ A subclass of HistogramFitter that puts a gaussian process smoothing prior on the bin heights
    """

    def __init__(self, *args, **kwargs):
        super(SmoothHistFitter, self).__init__(*args, **kwargs)
        self.smoothing = self.qvals.shape[0] / self.Nbins
        self.n_params = self.Nbins + 4
        self.param_names = [r'$\theta_{}$'.format(i) for i in range(self.Nbins)]
        self.param_names.extend(('lna', 'lntau', 'lnerr', 'mean'))

    def lnprior(self, pars):
        theta = pars[:self.Nbins]
        if np.any(theta < 0):
            return -np.inf
        a, tau, err = np.exp(pars[self.Nbins:-1])
        mean = pars[-1]
        kernel = a * kernels.ExpSquaredKernel(tau)
        gp = GP(kernel, mean=mean)
        gp.compute(self.bin_centers, yerr=err)
        return gp.lnlikelihood(theta) / self.smoothing

    def guess_fit(self):
        from scipy.optimize import minimize

        def errfcn(pars):
            ll = self.lnprob(pars)
            # print(pars, ll)
            return -ll

        initial_guess = np.ones(self.bin_centers.size + 4)
        initial_guess[-4] = 0.0
        initial_guess[-3] = -0.25
        initial_guess[-2] = -1.0
        initial_guess[-1] = -1.0
        bounds = [[1e-3, None] for p in self.bin_centers]
        bounds.append([-10, 20])
        bounds.append([-10, 10])
        bounds.append((-1, 5))
        bounds.append((-10, 10))
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x

    def _lnlike(self, pars):
        return self.lnprob(pars)

    def mnest_prior(self, cube, ndim, nparams):
        for i in range(self.Nbins):
            cube[i] *= 10
        # cube[:self.Nbins] *= 15
        cube[self.Nbins] = cube[self.Nbins] * 30 - 10
        cube[self.Nbins + 1] = cube[self.Nbins + 1] * 20 - 10
        cube[self.Nbins + 2] = cube[self.Nbins + 2] * 7 - 2
        cube[self.Nbins + 3] = cube[self.Nbins + 3] * 20 - 10
        return
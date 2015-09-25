""" Utility functions for simulating my proposed analysis method.
"""
import logging

import matplotlib.pyplot as plt
import h5py
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

import ForwardModeling
from Distributions import mass2teff, OrbitPrior, CensoredCompleteness
import Orbit


def make_comparison_plot(true_vals, fitted_vals, low=0, high=1, axes=None):
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axes[0].scatter(true_vals, fitted_vals, alpha=0.5)
    axes[0].plot([low, high], [low, high], 'r-', lw=2)
    axes[0].set_xlabel('True value')
    axes[0].set_ylabel('Fitted value')

    _, bins, _ = axes[1].hist(true_vals, bins=10, cumulative=True, histtype='step', normed=True)
    axes[1].hist(fitted_vals, bins=bins, cumulative=True, histtype='step', normed=True)

    ks_stat, p_value = ks_2samp(true_vals, fitted_vals)
    print('KS-test p-value = {:.3g}'.format(p_value))

    return fig, axes


def comparison_plots(mcmc_samples, true_q, true_a, true_e):
    """
    Make comparison plots for all of the variables we care about
    """

    q = np.nanmean(mcmc_samples[:, :, 0], axis=1)
    loga = np.nanmean(np.log10(mcmc_samples[:, :, 1]), axis=1)
    loge = np.nanmean(np.log10(mcmc_samples[:, :, 2]), axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 21))

    # Mass-ratio
    axes[0] = make_comparison_plot(true_q, q, axes=axes[0])
    axes[0][0].set_xlabel('True Mass-Ratio')
    axes[0][0].set_ylabel('Fitted Mass-Ratio')
    axes[0][1].set_xlabel('Mass-Ratio')
    axes[0][1].set_ylabel('Cumulative Frequency')

    # Mass-ratio
    axes[1] = make_comparison_plot(np.log10(true_a), loga, axes=axes[1], low=-2, high=8)
    axes[1][0].set_xlabel('True semimajor axis (log)')
    axes[1][0].set_ylabel('Fitted semimajor axis (log)')
    axes[1][1].set_xlabel('$\log{a}$')
    axes[1][1].set_ylabel('Cumulative Frequency')

    # Mass-ratio
    axes[2] = make_comparison_plot(np.log10(true_e), loge, axes=axes[2], low=-20, high=0)
    axes[2][0].set_xlabel('True log-eccentricity')
    axes[2][0].set_ylabel('Fitted log-eccentricity')
    axes[2][1].set_xlabel('$\log{e}$')
    axes[2][1].set_ylabel('Cumulative Frequency')

    plt.tight_layout()
    return fig, axes


def fit_orbits_multinest(output_base, outfilename, sample_parameters, rv1_err=0.1, rv2_err=0.2, use_qprior=True):
    """ Fit orbits to all of the simulated stars with parameters given in sample_parameters

    Parameters:
    ===========
    - output_base:       string
                         The base name (folder) to store the multinest files in. This will also be the group name
                         in the HDF5 file 'outfilename'

    - outfilename:       string:
                         The filename of an HDF5 file to store the MCMC samples in.

    - sample_parameters: pandas DataFrame
                         Each row is a set of parameters fully describing a binary system. This is generated in
                         ForwardModeling.make_representative_sample

    - rv1_err, rv2_err:  The error in the primary (rv1) and secondary(rv2) velocity measurements.

    - use_qprior:        boolean
                         Should we use the companion temperature to define a prior on the mass-ratio?

    Returns:
    =========
    None
    """

    outfile = h5py.File(outfilename, 'a')
    for i, row in sample_parameters.iterrows():
        print '\n\n{}'.format(i)
        # Make some observations of this binary system
        t_rv, rv1, rv2, t_im, rho, theta, K1, K2 = ForwardModeling.sample_orbit(row, N_rv=2, N_imag=0,
                                                                                rv1_err=rv1_err, rv2_err=rv2_err)

        # Estimate the mass-ratio prior
        M1_mean = row['M_prim']
        if use_qprior:
            T2_mean = mass2teff(row['M_prim'] * row['q'])
            prior = OrbitPrior(M1_mean, T2_mean)
            q_prior = prior._evaluate_empirical_q_prior
        else:
            q_prior = None

        # Fit the orbit
        logging.debug('Creating Fitter...')
        fitter = Orbit.SpectroscopicOrbitFitter(rv_times=t_rv, rv1_measurements=rv1, rv1_err=rv1_err,
                                                rv2_measurements=rv2, rv2_err=rv2_err,
                                                primary_mass=M1_mean,
                                                q_prior=q_prior)
        logging.debug('Fitting...')
        fitter.fit(backend='multinest', basename='{}/idx{}-'.format(output_base, i), overwrite=False)

        # Save the MCMC samples and the true parameters
        logging.debug('Saving MCMC samples')
        try:
            ds = outfile.create_dataset('{}/ds{}'.format(output_base, i), data=fitter.samples, maxshape=(None, 8))
        except RuntimeError:
            ds = outfile['{}/ds{}'.format(output_base, i)]
            ds.resize(fitter.samples.shape)
            ds[:] = fitter.samples
        for par in row.keys():
            ds.attrs[par] = row[par]

        ds.attrs['K1'] = K1
        ds.attrs['K2'] = K2
        ds.attrs['df_columns'] = list(fitter.samples.columns)
        ds.attrs['t_rv'] = t_rv
        ds.attrs['rv1'] = rv1
        ds.attrs['rv2'] = rv2
        ds.attrs['rv1_err'] = rv1_err
        ds.attrs['rv2_err'] = rv2_err
        outfile.flush()

    outfile.close()

    return


def read_orbit_samples(hdf5_file, group_name, sample_parameters=None, censor=False):
    """
    Read in orbit MCMC samples computed by fit_orbits_multinest

    Parameters:
    ===========
    - hdf5_file:         string:
                         The filename of an HDF5 file to store the MCMC samples in.

    - group_name:        string
                         The group name in the HDF5 file 'hdf5_file', where the samples can be found

    sample_parameters:   pandas DataFrame
                         This should have the same length as the sample parameters given in the corresponding call
                         to fit_orbits_multinest, and have the columns 'alpha' and 'beta'. The meaning of alpha
                         and beta are described in Distributions.OrbitPrior, but describe which mass-ratios are
                         detectable. This MUST be given if censor=True

    censor:              boolean
                         Should we censor the dataset? (i.e. decide whether a given companion was detected based
                         one its mass-ratio?)

    Returns:
    ========
    mcmc_samples:        Numpy array of shape (N_detected, length, 3)
                         Contains the MCMC samples for the mass-ratio, semimajor axis, and eccentricity
                         for each star. To make this an array, extra values are padded as NaNs so make
                         sure to use nanmean and such to analyze!

    M_prim, M_sec:       numpy arrays of shape (N_detected,)
                         The true values of the primary and secondary masses

    a, e:        numpy arrays of shape (N_detected,)
                         The true values of the period, semimajor axis, and eccentricity
    """
    if censor and sample_parameters is None:
        raise ValueError('Must give sample_parameters if you want to censor the dataset!')


    # Put all the relevant samples in a numpy array, censoring to "not detect" anything with q < 0.1.
    with h5py.File(hdf5_file, 'r') as f:
        if censor:
            # Loop through to determine which companions are detected
            maxlen = 0
            detected = np.zeros(len(f[group_name]))
            for ds_name, dataset in f[group_name].iteritems():
                idx = int(ds_name[2:])
                alpha = sample_parameters.ix[idx]['alpha']
                beta = sample_parameters.ix[idx]['beta']
                Q = CensoredCompleteness.sigmoid(dataset.attrs['q'], alpha, beta)
                r = np.random.uniform()
                if r < Q:
                    # Detected!
                    detected[idx] = 1
                    if dataset.shape[0] > maxlen:
                        maxlen = dataset.shape[0]
            n_datasets = detected.sum()
        else:
            n_datasets = len(f[group_name])
            maxlen = np.max([ds.shape[0] for _, ds in f[group_name].iteritems()])
            detected = np.ones(n_datasets, dtype=np.bool)

        # Make a big numpy array filled with NaNs of max shape
        data = np.ones((n_datasets, maxlen, 3)) * np.nan
        M_prim = np.ones(n_datasets) * np.nan
        M_sec = np.ones(n_datasets) * np.nan
        a = np.ones(n_datasets) * np.nan
        e = np.ones(n_datasets) * np.nan

        # Fill the numpy array where possible
        i = 0
        for ds_name, dataset in f[group_name].iteritems():
            idx = int(ds_name[2:])
            if detected[idx]:
                length = dataset.shape[0]
                df = pd.DataFrame(data=dataset.value, columns=dataset.attrs['df_columns'])
                M1 = dataset.attrs['M_prim']
                df['a'] = 10 ** (df['Period'] * (2. / 3.)) * (M1 * (1 + df['q'])) ** (1. / 3.)
                df['e'] = 10 ** (df['e'])
                values = df[['q', 'a', 'e']]

                data[i, :length, :] = values
                M_prim[i] = M1
                M_sec[i] = M1 * dataset.attrs['q']
                a[i] = dataset.attrs['a']
                e[i] = dataset.attrs['e']
                i += 1

    return data, M_prim, M_sec, a, e

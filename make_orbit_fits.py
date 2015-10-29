import numpy as np

import Simulation
import ForwardModeling


GAMMA = 0.4
MU = np.log(200)
SIGMA = np.log(10)
ETA = 0.7
MIN_MASS = 2.0  # Minimum primary mass
MAX_MASS = 20.0  # maximum primary mass
N_SAMPLE = 500

OUTPUT_BASE = 'malmquist_pool'
OUTFILENAME = 'Simulation_Data_Malmquist.h5'

if __name__ == '__main__':
    # Make the sample
    np.random.seed(1)
    sample_parameters = ForwardModeling.make_malmquist_sample(GAMMA, MU, SIGMA, ETA,
                                                              size=N_SAMPLE,
                                                              min_mass=MIN_MASS,
                                                              max_mass=MAX_MASS)
    sample_parameters['alpha'] = np.random.normal(loc=30, scale=5, size=N_SAMPLE)
    sample_parameters['beta'] = np.random.normal(loc=0.1, scale=0.03, size=N_SAMPLE)

    # Fit all the orbits
    Simulation.fit_orbits_multinest(OUTPUT_BASE, OUTFILENAME, sample_parameters,
                                    rv1_err=2.5, rv2_err=0.5, use_qprior=True)
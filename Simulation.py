""" Utility functions for simulating my proposed analysis method.
"""
import numpy as np




def sample_disk(Rmax=1e4, scale_height=300.0, Npoints=1e4):
    """ Uniformly sample a disk out to distance Rmax and with a given scale height.
    :param Rmax:
    :param scale_height:
    :return:
    """
    u1 = np.random.uniform(0, 1, Npoints)
    u2 = np.random.uniform(0, 1, Npoints)
    u3 = np.random.uniform(-1, 1, Npoints)
    r = Rmax * np.sqrt(u1)
    z = -scale_height * np.log(1 - u2)
    return r, np.sign(u3) * z


import sys
import os

from isochrones import StarModel
from isochrones.padova import Padova_Isochrone
import pandas as pd
import numpy as np


HDF5_OUTPUT = 'Primary_Parameters_full.h5'

class MyModel(StarModel):
    def _make_samples(self):
        chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))

        mass = chain[:,0]
        age = chain[:,1]
        feh = chain[:,2]
        distance = chain[:,3]
        AV = chain[:,4]
        lnprob = chain[:,-1]

        df = pd.DataFrame(data=dict(mass=mass, age=age, feh=feh, distance=distance, AV=AV, lnprob=lnprob))
        self._samples = df.copy()


if __name__ == '__main__':
    pad = Padova_Isochrone()

    i = int(sys.argv[1])
    pars = pd.read_csv('data/Primary_Parameters_full.csv').loc[i]
    model = MyModel(pad, Teff=(pars.teff, pars.teff_err),
                      logg=(pars.logg, pars.logg_err),
                      feh=(pars.feh, pars.feh_err))
    starname = pars.star.replace(' ', '_')
    print(starname)
    model.fit(basename=os.path.join('ParFitter', starname),
              overwrite=False)
    print(model.samples.describe())

    model.samples.to_hdf(HDF5_OUTPUT, key=starname)
    #with h5py.File(HDF5_OUTPUT, 'a') as outfile:
    #    ds = outfile.create_dataset(starname, data=model.samples)
    #    ds.attrs['star'] = pars.star 
    #    ds.attrs['columns'] = model.samples.columns

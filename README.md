# Binary Inference

This repository supplements the paper available [here](www.example.com) [TODO: link to paper once I have it]. To run this code, you will need my personal research library, kglib. You can use the [github version](https://github.com/kgullikson88/gullikson-scripts), or just

```bash
pip install kglib
```

The analysis was done in a series of jupyter notebooks using legacy python (python 2.7). The relevant notebooks are:

- **CompileObsStats.ipynb**: Compile observation data to make Tables 1, 2, and 4. Also makes a sky map of all the targets. This notebook will not run all the way through, because it contains some hard-coded paths to directories and files on my computer.
- **CompileCompanionData.ipynb**: Compiles the measured temperature data for the companions into a csv file.
- **EstimateMasses.ipynb**: Estimates the primary star mass and system age, as well as the companion mass. Saves everything into `data/MassSamples.h5`
- **ImagingAnalysis.ipynb**: Finds all stars in my reduced NIRI data, and estimates the separation, position angle, and flux ratio between stars.
- **MakeCCF_Plots.ipynb**: Makes cross-correlation function plots similar to Figure 1, but for every star.
- **MakeImagingPlots.ipynb**: Makes Figure 2
- **Malmquist_Bias.ipynb**: Does the simulation described in Section n 6.2 to estimate the malmquist bias P(obs}|q). This notebook is where the polynomial coefficients present in other notebooks are derived.
- **RealData.ipynb**: does all of the following
    - Compile the mass ratio samples for each star I detect
    - Estimate the completeness as a function of mass ratio
    - Fit the data to:
        - histogram
        - gaussian
        - power law
        - beta distribution
    - Compare my results to the VAST survey close companions. Find that they are unlikely to be drawn from the same distribution.
- 

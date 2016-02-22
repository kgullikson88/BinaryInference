# Binary Inference

This repository supplements the paper available [here](www.example.com) [TODO: link to paper once I have it]. To run this code, you will need my personal research library, kglib. You can use the [github version](https://github.com/kgullikson88/gullikson-scripts), or just

```bash
pip install kglib
```

The analysis was done in a series of jupyter notebooks using legacy python (python 2.7). The relevant notebooks are:

- **RealData.ipynb**: does all of the following
    - Compile the mass ratio samples for each star I detect
    - Estimate the completeness as a function of mass ratio
    - Fit the data to:
        - histogram
        - gaussian
        - power law
        - beta distribution
    - Compare my results to the VAST survey close companions. Find that they are unlikely to be drawn from the same distribution.
- **EstimateMasses.ipynb**: Estimates the primary star mass and system age, as well as the companion mass. Saves everything into `data/MassSamples.h5`
- **CompileObsStats.ipynb**: Compile observation data to make Tables 1, 2, and 4. Also makes a sky map of all the targets. This notebook will not run all the way through, because it contains some hard-coded paths to directories and files on my computer.
- **CompileCompanionData.ipynb**: TODO: describe this and the remaining notebooks.

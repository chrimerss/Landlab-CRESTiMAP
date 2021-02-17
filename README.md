# Landlab - CREST-iMAP

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django)![commit](https://img.shields.io/github/last-commit/chrimerss/Landlab-CRESTiMAP)

This is a fork from Landlab, we implemented a coupled Hydrologic&Hydraulic (H&H) model based on Landlab structure.

Original Landlab repo: https://github.com/landlab/landlab

## What is CREST-iMAP?

CREST-iMAP, namely the Coupled Routing and Excess STorage - inundation MAPping and Predction, is an extension of [CREST](http://ef5.ou.edu/index.html/) model, which was developed and released at the University of Oklahoma, the HyDROS lab. CREST model is a hydrologic model that only simulates streamflow. As an increasing convern of flooding, CREST-iMAP builds upon CREST and empowers inundation mapping efficiently,

## How to use this module?

**install dependencies**

We recommend using Anaconda to manage python dependencies. For now, this package is not officially published, so only develop mode is available, which requires some pre-configuration.


```
conda env create --file=_environment.yml
python setup.py develop
```

## TODO
- [ ] Add calibration and sensitivity schemes
- [ ] Add control file and cmd support
- [ ] Add user examples

## Contact

Allen Li (li1995@ou.edu)


# Landlab - CREST-iMAP

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django)![commit](https://img.shields.io/github/last-commit/chrimerss/Landlab-CRESTiMAP)

<pre>
======================================================================================
   _____   _____    ______    _____   _______            _   __  __              _____
  / ____| |  __ \  |  ____|  / ____| |__   __|          (_) |  \/  |     /\     |  __ \
 | |      | |__) | | |__    | (___      | |     ______   _  | \  / |    /  \    | |__) |
 | |      |  _  /  |  __|    \___ \     | |    |______| | | | |\/| |   / /\ \   |  ___/
 | |____  | | \ \  | |____   ____) |    | |             | | | |  | |  / ____ \  | |
  \_____| |_|  \_\ |______| |_____/     |_|             |_| |_|  |_| /_/    \_\ |_|

=======================================================================================
</pre>

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

**Run the model**

Modify main.py file and run

```python
python main.py
```

Or use jupyter notebook/lab to load the interface

```python
%load main.py
```

Or to profile the program, visualize with SnakeViz
```python
python -m cProfile -o profile_res.txt main.py
snakeviz profile_res.txt
```

## TODO
- [ ] Add calibration and sensitivity schemes
- [x] Add control file and cmd support
- [ ] Add user examples

## Contact

Allen Li (li1995@ou.edu)

# Membership Inference and Differential Privacy
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-membership-inference-and-differential-privacy)](https://api.reuse.software/info/github.com/SAP-samples/security-research-membership-inference-and-differential-privacy)

## Description
SAP Security Research sample code to reproduce the research done in our paper "Comparing local and central differential privacy using membership inference attacks"[1]. The MIAttack Framework follows the architecture suggested by Shokri et al.[2].

## Requirements
- [Python](https://www.python.org/) 3.6
- [Jupyter](https://jupyter.org/)
- [Tensorflow](https://github.com/tensorflow) 1.14
- [Tensorflow Privacy](https://github.com/tensorflow/privacy)
- [h5py](https://www.h5py.org/) 2.10.*
- [numpy](https://numpy.org/) 1.18.*
- [scikit-learn](https://scikit-learn.org/) 0.22.*
- [scipy](https://scipy.org/) 1.4.*

## Download and Installation
### MIAttack Framework

Implementation of membership inference attacks on artificial neural networks.

### Install

Running `make install` in the MIAttack folder should be enough for most usecases.
It creates the basic project directory structure and installs mia as well as other requirements.
You can use pip as your package manager and install the `mia` package via `python -m pip install -e ./`
For other package managers you need to install mia using `setup.py`.

### Directory Structure

After having run `make install`, the following directory structure should be created in your local 
file system. Note: Everything that must not be tracked by git is already in `.gitignore`.

```
MIAttack/
     |-- Makefile
     |-- setup.py
     |-- requirements.txt
     |-- data/                # data files
     |-- experiments/         # experiment results
     |-- logs/		          # log files
     |-- models/              # relevant pre-trained models
     |-- notebooks/           # evaluation notebooks
     |-- mia/			     # source root
          |--core/		     # the framework
          |--projects/	     # project implementations using mia

```

For every mia-project, e.g., dataset, a subdirectory should be created in `./MIAttack/mia/projects`. 
We recommend to create a file `<project>_data.py` for data preparation and loading, `<project>_model.py` to define your model and `<project>_run_attack.py` to perform the attack for each project. 
The results of a test run are supposed to be stored `./experiments/`. 

## ## Authors / Contributors
 - Daniel Bernau (corresponding author)
 - Jonas Robl
 - Philip-William Grassal
 - Steffen Schneider

 ## Citations
If you use this code in your research, please cite:

```
@article{DGR+19,
  author    = {Daniel Bernau and
               Philip{-}William Grassal and
               Jonas Robl and
               Florian Kerschbaum},
  title     = {Assessing differentially private deep learning with Membership Inference},
  journal   = {CoRR},
  volume    = {abs/1912.11328},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.11328},
  archivePrefix = {arXiv},
  eprint    = {1912.11328},
}
```

## References
[1] Daniel Bernau, Philip-William Grassal, Jonas Robl, Florian Kerschbaum:
Assessing differentially private deep learning with Membership Inference.
arXiv:1912.11328
https://arxiv.org/abs/1912.11328

[2] Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov:
Membership Inference Attacks against Machine Learning Models
In Proceedings of the IEEE Symposium on Security and Privacy (2017)
https://arxiv.org/abs/1610.05820

## License
Copyright (c) 2020 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.

Description
=========
This python package `classified_regression` contains the EIC2 framework described in the paper submission "Robot Learning with Crash Constraints". The user can run a 1D example where the algorithm finds the optimum on a constrained minimization problem with a single constraint. The objective f to be minimized is modeled with a standard GP. The constraint g is modeled with GPCR, i.e., the novel GP model proposed in this paper. Such model handles a hybrid set of observations: discrete labels (failure/success) and continuous values (obtained only upon success) and also estimates the constraint thresold from data.


Requirements
============

The algorithm runs in Python >= 3.7, and is developed under [BoTorch](https://botorch.org/).

> If your python installation does not meet the minimum requirement, we recommend creating a virtual environment with the required python version. For example, [Anaconda](https://www.anaconda.com/distribution/) allows this, and does not interfere with your system-wide Python installation underneath. 

> NOTE: We recommend opening this README.md file in an online Markdown editor, e.g., [StackEdit](https://stackedit.io/app#), for better readability.

[BoTorch](https://botorch.org/) is a flexible framework for developing new Bayesian optimization algorithms. It builts on [Pytorch](https://pytorch.org/) and uses [scipy Python optimizers](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html) for GP model fitting and acquisition function optimization. 


Installation 
============

1. Make sure your python version meets the required one. For this, open a terminal and type
```bash
python --version
```
2. Install the following dependencies
```bash
pip install numpy
pip install botorch
pip install matplotlib
pip install pyyaml
pip install hydra-core==0.11.3
pip install nlopt==2.6.2
```
3. Extract the contents of `classified_regression.zip`, provided in the supplementary material, to your desired path <path/to/classified_regression>
4. Navigate to the package folder and install it
```bash
cd <path/to/classified_regression>
pip install -e .
```

Running a 1D example
====================

```bash
cd <path/to/classified_regression>/classireg/experiments/numerical_benchmarks
python run_experiments.py
```

The algorithm is initialized with two points, randomly sampled within the domain.

Several verbose messages should be shown in the terminal, as the algorithm progresses. In addition, a plot similar to Fig. 2 in the paper should pop up, depicting the objective f, the constraint, the probability of constraint satifsfaction (omitted in the paper due to space constraints, but computable via eq. (9) in the paper), and the acquisition function (expected improvement with constraints).

General comments
================

 * All the hyperparameters are tunable and can be found in `<path/to/classified_regression>/classireg/experiments/numerical_benchmarks/config/simple1D.yaml`, and modified.
 * The first time any of the above algorithms are run, they can take a few seconds to start.

Known issues for macOS users
============================
 * If any of the aforementioned plots do not automatically pop up, try uncommenting line 3 in the file `<path/to/classified_regression>/classireg/utils/plotting_collection.py`
```python
matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
```

 * Support for using GPU is at the moment not fully implemented.
 
 * If you encounter problems while installing PyTorch, check [here](https://pytorch.org/get-started/locally/).



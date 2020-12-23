# con_bopt

The code contains an implementation of a constrained Bayesian optimization algorithm. The algorithm uses a regression GP for a return function and a classification GP for a binary success constraint. The implemented acquisition function combines the probability of improvement with a boundary uncertainty criteria.

The code uses the GPML toolbox, which is available at http://gaussianprocess.org/gpml/code.

A detailed description of the method can be found in:
> P. Englert, M. Toussaint:
> [Learning Manipulation Skills from a Single Demonstration](http://peter-englert.net/papers/2018_Englert_IJRR.pdf)
> International Journal of Robotics Research 37(1):137-154, 2018

## problems
This folder contains the definition of various constrained optimization problem for testing.


https://github.com/etpr/con_bopt
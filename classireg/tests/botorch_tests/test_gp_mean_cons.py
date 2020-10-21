import torch
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
# from botorch.test_functions.hartmann6 import neg_hartmann6
from botorch.test_functions import Hartmann
neg_hartmann6 = Hartmann(negate=True)
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective
from botorch.acquisition import MCAcquisitionObjective
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.samplers import MCSampler
from botorch.models.model import Model
# from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform

# from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from xsearch.models.gp_mean_cons import GPmeanConstrained
from botorch.models import FixedNoiseGP, ModelListGP
# from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional, Tuple
from torch import Tensor
# from typing import Optional
neg_hartmann6 = Hartmann(negate=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NOISE_SE = 0.5
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

BATCH_SIZE = 3
bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device=device, dtype=dtype)


import time
import warnings
import pdb

# Random seed for numpy and torch:
# np.random.seed(rep_nr)
torch.manual_seed(3)

def obj_callable(Z):
  return Z[..., 0]

def constraint_callable(Z):
  return 0.0 + Z[..., 1] 	# Z[...,1] represents g(x), with g(x) <= 0 meaning constraint satisfaction.
  												# If we need g(x) >= a, we must return a - Z[..., 1]

# define a feasibility-weighted objective for optimization
constrained_obj = ConstrainedMCObjective(
    objective=obj_callable,
    constraints=[constraint_callable],
    infeasible_cost=0.0,
    eta=1e-3,
)

def outcome_constraint(X):
  """L1 constraint; feasible if less than or equal to zero."""
  return X.sum(dim=-1) - 3

def weighted_obj(X):
  """Feasibility weighted objective; zero if not feasible."""
  return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)

def generate_initial_data(n=10):
  # generate training data
  train_x = torch.rand(10, 6, device=device, dtype=dtype)
  exact_obj = neg_hartmann6(train_x).unsqueeze(-1)  # add output dimension
  exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
  train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
  train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
  best_observed_value = weighted_obj(train_x).max().item()
  return train_x, train_obj, train_con, best_observed_value

def run():

	train_x, train_obj, train_con, best_observed_value_nei = generate_initial_data(n=10)

	# define models for objective and constraint
	model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
	model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
	# combine into a multi-output GP model
	model = ModelListGP(model_obj, model_con)
	mll = SumMarginalLogLikelihood(model.likelihood, model)

	fit_gpytorch_model(mll)

	acqui_gpmean_cons = GPmeanConstrained(model=model,objective=constrained_obj)

	# Forward:
	# X = torch.rand(size=(1,6))
	# acqui_gpmean_cons.forward(X)

	method_opti = "SLSQP" # constraints
	# method_opti = "COBYLA" # constraints
	# method_opti = "L-BFGS-B"

	# Below, num_restarts must be equal to q, otherwise, it fails...
	options = {"batch_limit": 1,"maxiter": 200,"ftol":1e-6,"method":method_opti}
	x_eta_c, eta_c = optimize_acqf(acq_function=acqui_gpmean_cons,bounds=bounds,q=1,num_restarts=1,
																raw_samples=500,return_best_only=True,options=options)

	


	pdb.set_trace()

if __name__ == "__main__":

	run()
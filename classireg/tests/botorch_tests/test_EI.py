import math
import torch
from torch import Tensor
from typing import Optional

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition import ExpectedImprovement
# from botorch.acquisition.monte_carlo import MCAcquisitionFunction, IdentityMCObjective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective, IdentityMCObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from botorch.optim import optimize_acqf
from botorch.gen import gen_candidates_scipy, gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def test(dim=1):

	assert dim == 1

	X0 = torch.Tensor([[0.93452506],
								 [0.18872502],
								 [0.89790337],
								 [0.95841797],
								 [0.82335255],
								 [0.45000000],
								 [0.50000000]])
	Y0 = torch.Tensor([-0.4532849,-0.66614552,-0.92803395,0.08880341,-0.27683621,1.000000,1.500000])
	# Y0 = Y0[:,None]
	Neval = Y0.shape[0]

	train_x = X0
	train_y = Y0[:,None]
	Nrestarts = 10

	model = SingleTaskGP(train_x, train_y)
	EI = ExpectedImprovement(model, best_f=0.2)

	print("[get_next_point()] Computing next candidate by maximizing the acquisition function ...")
	options={"batch_limit": 50,"maxiter": 200,"ftol":1e-9,"method":"L-BFGS-B","iprint":2,"maxls":30,"disp":True}
	x_next,alpha_next = optimize_acqf(acq_function=EI,bounds=torch.Tensor([[0.0]*dim,[1.0]*dim],device=device),q=1,num_restarts=Nrestarts,
																		raw_samples=500,return_best_only=True,options=options)

	print("x_next:",x_next)
	print("alpha_next:",alpha_next)


if __name__ == "__main__":

	test()






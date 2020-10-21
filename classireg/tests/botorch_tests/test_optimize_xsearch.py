import math
import torch
from torch import Tensor
from typing import Optional

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition import AcquisitionFunction
# from botorch.acquisition.monte_carlo import MCAcquisitionFunction, IdentityMCObjective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective, IdentityMCObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from xsearch.acquisitions.xsearch import qXsearch
from xsearch.models.gpmodel import GPmodel

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
	train_y = Y0

	gp = GPmodel(train_X=train_x, train_Y=train_y, noise_std=0.01)
	mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
	fit_gpytorch_model(mll)

	gp.set_hyperparameters(lengthscale=0.05*torch.ones(dim),outputscale=1.0,noise=0.01**2)
	gp.update_hyperparameters_of_model_grad()

	hdl_fig_obj,hdl_axes_obj_splots = plt.subplots(2,1,sharex=False,sharey=False,figsize=(10, 7))
	axes_GPobj = hdl_axes_obj_splots[0]
	axes_acqui = hdl_axes_obj_splots[1]

	Ndiv = 201
	gp.plot(title="",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3)

	print("Initializing qXsearch() ...")
	qXs = qXsearch(model=gp,u_vec=torch.Tensor([0.0]))

	# Optimize using BOtorch:
	# q refers to evaluating the acquisition function jointly on q points, like done in GLASSES.
	print("Get next point for both values of u_vec:")
	# bounds = torch.tensor([[0.0]*dim, [1.0]*dim], device=device, dtype=dtype)
	bounds = torch.tensor([[0.0]*dim, [1.0]*dim], device=device)
	BATCH_SIZE = 1
	u_vec = torch.Tensor([-1.2,-2.3])
	test_x_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
	test_x_vec = test_x_vec.unsqueeze(1) # Make this [Ntest x q x dim] = [n_batches x n_design_points x dim], with q=1 -> Double-check in the documentation!
	
	# # Test calling qXsearch sequentially VS jointly:
	# t_start = time.time()
	# alphaX_x_vec = qXs.forward(X=test_x_vec).detach().cpu().numpy() # This is ~35 times faster than doing it sequentially....
	# print("Elapsed all togetehr: ",time.time()-t_start)
	# t_start = time.time()
	# alphaX_x_vec_single = torch.zeros(test_x_vec.shape[0])
	# for k in range(test_x_vec.shape[0]):
	# 	alphaX_x_vec_single[k] = qXs.forward(X=test_x_vec[k,:,:])
	# print("Elapsed sequential: ",time.time()-t_start)
	# pdb.set_trace()


	Nfcmin = len(u_vec)
	xnext = torch.zeros(2)
	alpha_next = torch.zeros(2)
	alpha_next_ours = torch.zeros(2)
	alphaX_vec_all = np.zeros((Ndiv,len(u_vec)))
	for k in range(len(u_vec)):

		# print("Optimizing acqf using adam...")
		# Xinit = gen_batch_initial_conditions(qXs, bounds, q=BATCH_SIZE, num_restarts=5, raw_samples=500)
		# batch_candidates, batch_acq_values = gen_candidates_torch(initial_conditions=Xinit,
		# 																													acquisition_function=qXs,
		# 																													lower_bounds=bounds[0],
		# 																													upper_bounds=bounds[1],
		# 																													options={"maxiter":200, "lr":0.1})

		# candidate_location_adam = get_best_candidates(batch_candidates, batch_acq_values)
		# pdb.set_trace()

		print("Optimizing acqf...")
		qXs.update_u_vec(u_vec=torch.Tensor([u_vec[k].item()]))
		candidates_locations,\
		candidates_values = optimize_acqf(acq_function=qXs,bounds=bounds,q=BATCH_SIZE,num_restarts=5,
																			raw_samples=500,return_best_only=True,options={"batch_limit": 1,"maxiter": 200,"ftol":1e-6,"method":"SLSQP"})
		
		print("Evaluating candidate locations...")
		xnext[k] = candidates_locations.squeeze().item()
		alpha_next[k] = candidates_values.squeeze().item()
		# pdb.set_trace()
		# qXs.debug = True
		alpha_next_ours[k] = qXs.forward(X=candidates_locations.unsqueeze(1)).squeeze().item()

		print("Computing acquisition function in a grid:")
		alphaX_vec_all[:,k] = qXs.forward(X=test_x_vec).detach().cpu().numpy()


	# Consider using Adam! That's the only reason we carry the gradients along, right? and doesn't need to pass memory from torch to numpy...

	# Normalize:
	normalize = True
	if normalize:
		alpha_max = np.amax(alphaX_vec_all,axis=0)
		alphaX_vec_all_nor = alphaX_vec_all / alpha_max
		alpha_next_nor = alpha_next / alpha_max
	else:
		alphaX_vec_all_nor = alphaX_vec_all
		alpha_next_nor = alpha_next

	Nquant = len(u_vec)
	color = "mediumpurple"
	color_list = ["mediumpurple","darkgreen"]
	linestyle = "-"
	linewidth = 2.0
	for k in range(Nfcmin):
		axes_GPobj.plot(np.array([0.0,1.0]),u_vec[k]*np.ones(2),linestyle=linestyle,linewidth=1.5,color=color_list[k])
		axes_acqui.plot(test_x_vec.squeeze(),alphaX_vec_all_nor[:,k],linestyle="-",linewidth=linewidth,color=color_list[k])

		# Add next points, obtained from the optimizer:
		axes_acqui.plot(xnext[k],alpha_next_nor[k],marker="o",markersize=6,color=color_list[k],linestyle="None")

		# # Re-evaluate the cost function
		# axes_acqui.plot(xnext[k],alpha_next_ours[k],marker="v",markersize=6,color=color_list[k],linestyle="None")

	# Re-plot the evaluations:
	axes_GPobj.plot(X0.squeeze().detach().cpu().numpy(),
									Y0.squeeze().detach().cpu().numpy(),
									marker="o",markersize=6,color="maroon",linestyle="None")


	axes_GPobj.set_yticks([])
	axes_GPobj.set_xticks([])

	axes_acqui.set_yticks([])
	axes_acqui.set_xticks([])
	# axes_acqui.set_ylim([-0.1,1.1])
	axes_acqui.set_xlim([0.0,1.0])

	plt.show(block=True)

if __name__ == "__main__":

	test()
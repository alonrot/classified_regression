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

from acquisitions.xsearch import qXsearch
from models.gpmodel import GPmodel

import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot1D_toy_example(dim=1):

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

	Ndiv = 151
	gp.plot(title="",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3)

	print("Initializing qXsearch() ...")
	qXs = qXsearch(gp)
	u_vec = torch.Tensor([-1.,-2.])
	test_x_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
	test_x_vec = test_x_vec.unsqueeze(1) # Make this [Ntest x q x dim] = [n_batches x n_design_points x dim], with q=1 -> Double-check in the documentation!

	# Computing acquisition function in a grid:
	print("# Computing acquisition function in a grid:")
	alphaX_vec_all = np.zeros((Ndiv,len(u_vec)))
	for k in range(len(u_vec)):
		qXs.update_u_vec(u_vec=torch.Tensor([u_vec[k].item()]))
		alphaX_vec_all[:,k] = qXs.forward(X=test_x_vec).detach().cpu().numpy()

	# Maximum:
	alphaX_vec_all_nor = alphaX_vec_all / np.amax(alphaX_vec_all,axis=0)
	Nfcmin = u_vec.shape[0]


	Nquant = len(u_vec)
	color = "mediumpurple"
	color_list = ["mediumpurple","darkgreen"]
	linestyle = "-"
	linewidth = 2.0
	for k in range(Nfcmin):
		axes_GPobj.plot(np.array([0.0,1.0]),u_vec[k]*np.ones(2),linestyle=linestyle,linewidth=1.5,color=color_list[k])
		axes_acqui.plot(test_x_vec.squeeze(),alphaX_vec_all_nor[:,k],linestyle="-",linewidth=linewidth,color=color_list[k])

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

	plot1D_toy_example()
 
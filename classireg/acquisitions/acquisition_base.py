import torch
from botorch.models.model import Model
from abc import ABC, abstractmethod
import numpy as np
from classireg.models.gp_mean import GPmean
from botorch.optim import optimize_acqf
from botorch.gen import gen_candidates_scipy, gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from classireg.utils.plotting_collection import PlotProbability
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(linewidth=10000)
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
dtype = torch.float32


class AcquisitionBaseTools(ABC):

	def __init__(self, model: Model, Nrestarts_eta: int) -> None:
		"""
		"""

		logger.info("Starting AcquisitionBaseTools ...")

		# Define GP posterior mean:
		self.model = model
		self.gp_mean = GPmean(self.model)

		# Some options:
		self.Nrestarts_eta = Nrestarts_eta

		self.x_eta = None
		self.eta = None
		self.bounds = torch.tensor([[0.0]*self.model.dim, [1.0]*self.model.dim],device=device)
		# bounds = torch.tensor([[0.0]*dim, [1.0]*dim], device=device)

		# Optimization method:
		self.method_opti = "L-BFGS-B"
		# self.method_opti = "SLSQP" # constraints
		# self.method_opti = "COBYLA" # constraints


	@abstractmethod
	def get_next_point(self):
		pass

	def find_eta(self):
		"""
		Find the minimum of the posterior mean, i.e., min_x mu(x|D), where D is the data set D={Y,X}, and mu(x|D)
		is the posterior mean of the GP queried at location x. For this, we do local optimization with random
		restarts.
		"""

		logger.info("Finding min_x mu(x|D)...")

		options = {"batch_limit": 50, "maxiter": 200, "ftol": 1e-6, "method": self.method_opti}
		x_eta, _ = optimize_acqf(acq_function=self.gp_mean,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_eta,
																	raw_samples=500,return_best_only=True,options=options)

		logger.info("Done!")
		# Revaluate the mean (because the one returned might be affected by the constant that is added to ensure non-negativity)
		eta = self.model(x_eta).mean.view(1)

		return x_eta, eta

	def get_simple_regret(self, fmin_true):

		Y_obj = self.model.train_targets
		if Y_obj is None:
			return torch.tensor([+float("Inf")],device=device,dtype=dtype) # The regret cannot be computed
		else:
			f_simple = torch.min(Y_obj).view(1)

		regret_simple = f_simple - fmin_true

		return regret_simple

	def optimize_acqui_use_restarts_individually(self,options):

		# Get initial random restart points:
		logger.info("[get_next_point()] Generating random restarts ...")
		initial_conditions = gen_batch_initial_conditions(acq_function=self,bounds=self.bounds,q=1,
																num_restarts=self.Nrestarts,raw_samples=500, options=options)

		logger.info("[get_next_point()] Optimizing acquisition function with {0:d} restarts ...".format(self.Nrestarts))
		x_next_many = torch.zeros(size=(self.Nrestarts,1,self.dim))
		alpha_next_many = torch.zeros(size=(self.Nrestarts,))
		for k in range(self.Nrestarts):

			if (k+1) % 5 == 0:
				logger.info("[get_next_point()] restart {0:d} / {1:d}".format(k+1,self.Nrestarts))

			x_next_many[k,:], alpha_next_many[k] = gen_candidates_scipy(initial_conditions=initial_conditions[k,:].view((1,1,self.dim)),
																							acquisition_function=self, lower_bounds=0.0, upper_bounds=1.0, options=options)

		# Get the best:
		logger.info("[get_next_point()] Getting best candidates ...")
		x_next = get_best_candidates(x_next_many, alpha_next_many).detach()
		alpha_next = self.forward(x_next).detach()
		
		return x_next, alpha_next

	def optimize_acqui_use_restarts_as_batch(self,options):

		logger.info("[get_next_point()] Optimizing acquisition function ...")
		x_next, alpha_next = optimize_acqf(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts,
																			raw_samples=500,return_best_only=True,options=options)

		return x_next, alpha_next

	def plot(self,axes=None,block=False,title=None,plotting=False,Ndiv=41,showtickslabels=True,
					showticks=True,xlabel=None,ylabel=None,clear_axes=True,legend=False,labelsize=None,normalize=False,
					colorbar=False,color=None,label=None,local_axes=None,x_next=None,alpha_next=None):

		if plotting == False:
			return None

		if self.model.dim > 1:
			return None

		if local_axes is None and axes is None:
			self.fig,(local_axes) = plt.subplots(1,1,sharex=True,figsize=(10, 7))
		elif local_axes is None:
			local_axes = axes
		elif axes is None:
			pass # If the internal axes already have some value, and no new axes passed, do nothing
		elif local_axes is not None and axes is not None:
			local_axes = axes

		local_pp = PlotProbability()

		if x_next is not None and alpha_next is not None:
			x_next_local = x_next
			alpha_next_local = alpha_next
		else:
			x_next_local = None
			alpha_next_local = 1.0

		test_x_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
		test_x_vec = test_x_vec.unsqueeze(1) # Make this [Ntest x q x dim] = [n_batches x n_design_points x dim], with q=1 -> Double-check in the documentation!
		var_vec = self.forward(X=test_x_vec).detach().cpu().numpy()

		if self.model.dim == 1:
			local_axes = local_pp.plot_acquisition_function(var_vec=var_vec,xpred_vec=test_x_vec.squeeze(1),x_next=x_next_local,acqui_next=alpha_next_local,
																			xlabel=xlabel,ylabel=ylabel,title=title,legend=legend,axes=local_axes,clear_axes=clear_axes,
																			xlim=np.array([0.,1.]),block=block,labelsize=labelsize,showtickslabels=showtickslabels,showticks=showticks,
																			what2plot=None,color=color,ylim=None)
			plt.pause(0.25)

		elif self.model.dim == 2:
			if self.x_next is not None:
				Xs = np.atleast_2d(self.x_next)
			else:
				Xs = self.x_next
			local_axes = local_pp.plot_GP_2D_single(var_vec=var_vec,Ndiv_dim=Ndiv*np.ones(self.model.dim,dtype=np.int64),Xs=Xs,Ys=self.alpha_next,
													x_label=xlabel,y_label=ylabel,title=title,axes=local_axes,clear_axes=clear_axes,legend=legend,block=block,
													colorbar=colorbar,color_Xs="gold")
			plt.pause(0.25)

		return local_axes



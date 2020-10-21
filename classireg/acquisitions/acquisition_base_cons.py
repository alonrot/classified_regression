import torch
from botorch.models.model import Model
from abc import ABC, abstractmethod
import numpy as np
from classireg.models.gp_mean import GPmean
from botorch.optim import optimize_acqf
from botorch.gen import gen_candidates_scipy, gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from classireg.utils.plotting_collection import PlotProbability
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from typing import List
import pdb
from botorch.models import FixedNoiseGP, ModelListGP
from classireg.models.gp_mean_cons import GPmeanConstrained
dist_standnormal = Normal(loc=0.0,scale=1.0)
np.set_printoptions(linewidth=10000)
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
idxm = dict(obj=0,cons=1)

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

class AcquisitionBaseToolsConstrained(ABC):

	def __init__(self, model_list: List[Model], Nrestarts_eta_c: int) -> None:
		"""
		"""

		logger.info("Starting AcquisitionBaseTools ...")
		self.model_list = model_list

		# # Define GP posterior mean:
		# self.gp_mean_obj = GPmean(self.model_list[idxm['obj']])

		# define models for objective and constraint
		# model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
		# model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
		# combine into a multi-output GP model
		# mll = SumMarginalLogLikelihood(model.likelihood, model)
		# fit_gpytorch_model(mll)

		# pdb.set_trace()

		self.gp_mean_obj_cons = GPmeanConstrained(model=model_list, objective=constrained_obj)

		# Some options:
		self.Nrestarts_eta_c = Nrestarts_eta_c

		self.dim = self.model_list.models[idxm['cons']].dim
		self.x_eta_c = None
		self.eta_c = None
		self.bounds = torch.tensor([[0.0]*self.dim, [1.0]*self.dim],device=device)

		# Optimization method: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
		self.method_opti = "L-BFGS-B"
		# self.method_opti = "SLSQP" # constraints
		# self.method_opti = "COBYLA" # constraints

		logger.info("Starting AcquisitionBaseTools ... II")


	@abstractmethod
	def get_next_point(self):
		pass

	def find_eta_c(self,rho_t):
		"""
		Find the minimum of the posterior mean, i.e., min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t, where D is the data set D={Y,X}, mu(x|D)
		is the posterior mean of the GP queried at location x, and rho_t depends on the current budget of failures.
		"""

		logger.info("Finding min_x mu(x|D) s.t. Pr(g(x) <= 0) > {0:2.2f}".format(rho_t))
		self.gp_mean_obj_cons.rho_t = rho_t

		options = {"batch_limit": 1, "maxiter": 200, "ftol": 1e-6, "method": self.method_opti}
		x_eta_c, _ = optimize_acqf(acq_function=self.gp_mean_obj_cons,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_eta_c,
																	raw_samples=500,return_best_only=True,options=options)

		logger.info("Done!")
		
		# Revaluate the mean (because the one returned might be affected by the constant that is added to ensure non-negativity)
		eta_c = self.model_list.models[idxm['obj']](x_eta_c).mean.view(1)

		return x_eta_c, eta_c

	def get_simple_regret_cons(self, fmin_true):

		Ycons_safe = self.model_list.models[idxm['cons']].train_ys
		Yobj_safe = self.model_list.models[idxm['obj']].train_targets # Since we don't include the non-stable evaluations in GPCR, the safe evaluations are the evaluations themselves
		if len(Ycons_safe) == 0 and Yobj_safe is None: # No safe points, but obj has no evaluations at all either
			return torch.tensor([+float("Inf")],device=device,dtype=dtype) # The regret cannot be computed
		elif len(Ycons_safe) == 0: # No safe points, but obj has some evaluations already
			raise NotImplementedError("We assume that the objective only acquires evaluations if they are safe.")
			# f_simple = torch.max(Yobj_safe) # We take the worst observation here. Otherwise, the regret can become non-monotonic
		else:
			f_simple = torch.min(Yobj_safe).view(1)

		regret_simple = f_simple - fmin_true

		return regret_simple

	def plot(self,axes=None,block=False,title=None,plotting=False,Ndiv=41,showtickslabels=True,
					showticks=True,xlabel=None,ylabel=None,clear_axes=True,legend=False,labelsize=None,normalize=False,
					colorbar=False,color=None,label=None,local_axes=None,x_next=None,alpha_next=None,
					linewidth=2.0):

		if plotting == False:
			return None

		if self.dim > 1:
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

		post_batch = False
		if post_batch:
			var_vec = self.forward(X=test_x_vec).detach().cpu().numpy()
		else:
			var_vec = torch.zeros((Ndiv))
			for k in range(Ndiv):
				var_vec[k] = self.forward(X=test_x_vec[k,:].view(-1,self.dim))
			var_vec = var_vec.detach().cpu().numpy()

		if self.dim == 1:
			local_axes = local_pp.plot_acquisition_function(var_vec=var_vec,xpred_vec=test_x_vec.squeeze(1),x_next=x_next_local,acqui_next=alpha_next_local,
																			xlabel=xlabel,ylabel=ylabel,title=title,legend=legend,axes=local_axes,clear_axes=clear_axes,
																			xlim=np.array([0.,1.]),block=block,labelsize=labelsize,showtickslabels=showtickslabels,showticks=showticks,
																			what2plot=None,color=color,ylim=None,linewidth=linewidth)
			plt.pause(0.25)

		elif self.dim == 2:
			if self.x_next is not None:
				Xs = np.atleast_2d(self.x_next)
			else:
				Xs = self.x_next
			local_axes = local_pp.plot_GP_2D_single(var_vec=var_vec,Ndiv_dim=Ndiv*np.ones(self.dim,dtype=np.int64),Xs=Xs,Ys=self.alpha_next,
													x_label=xlabel,y_label=ylabel,title=title,axes=local_axes,clear_axes=clear_axes,legend=legend,block=block,
													colorbar=colorbar,color_Xs="gold")
			plt.pause(0.25)

		return local_axes



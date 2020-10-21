import torch
from torch import Tensor
from typing import Optional, List
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
from classireg.utils.plotting_collection import PlotProbability
import matplotlib.pyplot as plt
import numpy as np # These two libraries need to dissapear, as allt he code should be in torch
from scipy.special import erf
from botorch.optim import optimize_acqf
from .acquisition_base_cons import AcquisitionBaseToolsConstrained
import pdb
from torch.distributions.normal import Normal
from botorch.models import ModelListGP
from classireg.models import GPmodel, GPClassifier
from scipy.stats import norm
dist_standnormal = Normal(loc=0.0,scale=1.0)
from classireg.utils.optimize import ConstrainedOptimizationNonLinearConstraints, OptimizationNonLinear
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.gen import gen_candidates_scipy
from botorch.gen import get_best_candidates
np.set_printoptions(linewidth=10000)
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
idxm = dict(obj=0,cons=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class ExpectedImprovementWithConstraintsClassi():
	"""
	This class expects a GP classifier as constraint
	
	"""
	def __init__(self, dim: int, model_list: list, options: dict) -> None:
		
		logger.info("Starting EIC ...")

		self.model_list = model_list

		self.dim = dim
		self.Nrestarts = options.optimization.Nrestarts
		self.algo_name = options.optimization.algo_name
		self.constrained_opt = OptimizationNonLinear(	dim=self.dim,
																									fun_obj=self.forward,
																									algo_str=self.algo_name,
																									bounds=[ [0.0]*self.dim, [1.0]*self.dim ],
																									minimize=False,
																									what2optimize_str="EIC acquisition")

		# This is needed to 
		self.model_list[idxm['cons']](torch.randn(size=(1,1,self.dim)))

		# self.use_nlopt = False
		self.disp_info_scipy_opti = options.optimization.disp_info_scipy_opti

		# self._rho_conserv = options.prob_satisfaction
		self.x_next, self.alpha_next = None, None
		self.only_prob = False

		self.x_eta_c = None
		self.eta_c = None
		self.bounds = torch.tensor([[0.0]*self.dim, [1.0]*self.dim],device=device)

		self.maximize = False # If tru, we assume we that we want to maximize the objective. Herein, we consider it as cost, hence, we minimize it

	def get_simple_regret_cons(self, fmin_true):

		Ycons = self.model_list[idxm['cons']].train_targets

		N_Ycons_safe = torch.sum(Ycons == +1)

		Yobj_safe = self.model_list[idxm['obj']].train_targets # Since we don't include the non-stable evaluations in the objective GP, the safe evaluations are the evaluations themselves
		if N_Ycons_safe == 0 and Yobj_safe is None: # No safe points, but obj has no evaluations at all either
			return torch.tensor([+float("Inf")],device=device,dtype=dtype) # The regret cannot be computed
		elif N_Ycons_safe == 0: # No safe points, but obj has some evaluations already
			raise NotImplementedError("We assume that the objective only acquires evaluations if they are safe.")
			# f_simple = torch.max(Yobj_safe) # We take the worst observation here. Otherwise, the regret can become non-monotonic
		else:
			if Yobj_safe is None:
				pdb.set_trace()
			f_simple = torch.min(Yobj_safe).view(1)

		regret_simple = f_simple - fmin_true

		return regret_simple

	def __call__(self, X: Tensor) -> Tensor:
		return self.forward(X)

	# @t_batch_mode_transform(expected_q=1)
	def forward(self, X: Tensor) -> Tensor:
		r"""Evaluate Constrained Expected Improvement on the candidate set X.

		Args:
			X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
				points each.

		Returns:
			A `(b)`-dim Tensor of Expected Improvement values at the given
				design points `X`.
		"""
		# import pdb; pdb.set_trace()

		if X.dim() == 1:
			X = X.view(1,self.dim)

		# means, sigmas = self._get_posterior_reimplemented(X)

		# Get posterior of objective:
		mvn_obj = self.model_list[idxm['obj']](X)
		mean_obj = mvn_obj.mean
		sigma_obj = mvn_obj.variance.sqrt()

		# # (b) x 1
		# mean_obj = means[..., [self.objective_index]]
		# sigma_obj = sigmas[..., [self.objective_index]]

		# print("mean_obj.shape:",mean_obj.shape)
		# print("sigma_obj.shape:",sigma_obj.shape)
		# print("means.shape:",means.shape)
		# print("sigmas.shape:",sigmas.shape)
		
		# Probability of feasibility:
		prob_feas = self._compute_prob_feas(X=X)

		# print("prob_feas.shape:",prob_feas.shape)
		# pdb.set_trace()

		if self.only_prob:
			ei_times_prob = prob_feas # Use only the probability of feasibility
		else:
			u = (mean_obj - self.best_f.expand_as(mean_obj)) / sigma_obj
			if not self.maximize:
				u = -u
			normal = Normal(torch.zeros(1, device=u.device, dtype=u.dtype),torch.ones(1, device=u.device, dtype=u.dtype),)
			ei_pdf = torch.exp(normal.log_prob(u))  # (b) x 1
			ei_cdf = normal.cdf(u)
			ei = sigma_obj * (ei_pdf + u * ei_cdf)
			ei_times_prob = ei.mul(prob_feas)

		# print("ei_times_prob.shape:",ei_times_prob.shape)

		# pdb.set_trace()
		val = ei_times_prob.squeeze(dim=-1)
		if val.dim() == 1 and len(val) == 1 or val.dim() == 0:
			val = val.item()
		# else:
		# 	pdb.set_trace()

		# if isinstance(val,float):
		# 	pdb.set_trace()

		# if val.dim() == 1:
		# 	# if not val.shape[0] == 1:
		# 	if val.shape[0] != X.shape[0]:
		# 		pdb.set_trace()

		# print("X.shape:",X.shape)
		# print("val:",val)
		# pdb.set_trace()

		return val

	def get_best_constrained_evaluation(self):

		# pdb.set_trace()
		Ycons = self.model_list[idxm['cons']].train_targets
		Ycons_safe = Ycons[Ycons == +1]
		Yobj_safe = self.model_list[idxm['obj']].train_targets # Since we don't include the non-stable evaluations in GPCR, the safe evaluations are the evaluations themselves
		if len(Ycons_safe) > 0 and Yobj_safe is None:
			raise ValueError("This case should not happen (!) We assume that objective evaluations are only collected when the contraint is satisfied...")
		elif len(Ycons_safe) > 0:
			return torch.min(Yobj_safe).view(1)
		else: # No safe points yet
			return ValueError("This exception (no safe data collected yet) case is assumed to be handled in upper levels, so we should never enter here ...")

	def get_next_point(self) -> (Tensor, Tensor):

		# pdb.set_trace()
		if self.model_list[idxm["obj"]].train_targets is None: # No safe evaluations case
			self.eta_c = torch.zeros(1,device=device,dtype=dtype)
			self.x_eta_c = torch.zeros((1,self.dim),device=device,dtype=dtype)
			self.best_f = None
			self.only_prob = True
		else:

			self.eta_c = torch.zeros(1,device=device,dtype=dtype)
			self.x_eta_c = torch.zeros((1,self.dim),device=device,dtype=dtype)

			
			# # The following functions need to be called in the given order:
			# try:
			# 	self.update_eta_c(rho_t=self.rho_conserv) # Update min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t
			# except Exception as inst:
			# 	logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			# 	logger.info("Not optimizing eta_c ...")

			# self.best_f = self.eta_c
			self.best_f = self.get_best_constrained_evaluation() - self.model_list[idxm["obj"]].likelihood.noise.sqrt()[0].view(1)
			self.only_prob = False

		self.x_next, self.alpha_next = self.get_acqui_fun_maximizer()

		if self.x_next is not None and  self.alpha_next is not None:
			logger.info("xnext: " + str(self.x_next.view((1,self.dim)).detach().cpu().numpy()))
			logger.info("alpha_next: {0:2.2f}".format(self.alpha_next.item()))
		else:
			logger.info("xnext: None")
			logger.info("alpha_next: None")

		logger.info("self.x_eta_c: "+str(self.x_eta_c))
		logger.info("self.eta_c: "+str(self.eta_c))
		logger.info("self.best_f: "+str(self.best_f))
	
		return self.x_next,self.alpha_next

	def get_acqui_fun_maximizer(self):

		logger.info("Computing next candidate by maximizing the acquisition function ...")
		options = {"batch_limit": 50,"maxiter": 300,"ftol":1e-6,"method":"L-BFGS-B","iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}

		# Get initial random restart points:
		logger.info("Generating random restarts ...")
		initial_conditions = gen_batch_initial_conditions(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts,raw_samples=500, options=options)
		# logger.info("initial_conditions:" + str(initial_conditions))

		logger.info("Using nlopt ...")
		x_next, alpha_next = self.constrained_opt.run_optimization(initial_conditions.view((self.Nrestarts,self.dim)))

		# # TODO: Is this really needed?
		# prob_val = self.get_probability_of_safe_evaluation(x_next.unsqueeze(1))
		# if prob_val < self.rho_conserv:
		# 	logger.info("(Is this really needed????) x_next violates the probabilistic constraint...")
		# 	pdb.set_trace()
		
		logger.info("Done!")

		return x_next, alpha_next

	def _compute_prob_feas(self, X):

		# pdb.set_trace()

		# if "BernoulliLikelihood" in repr(self.model_list.models[idxm['cons']].likelihood): # GPClassi
		mvn_cons = self.model_list[idxm['cons']](X)
		prob_feas = self.model_list[idxm['cons']].likelihood(mvn_cons).mean
		# 	# print("prob_feas:",prob_feas)
		# else: # GPCR
		# 	prob_feas = super()._compute_prob_feas(X=X, means=means, sigmas=sigmas)

		return prob_feas


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
		var_vec = self.forward(X=test_x_vec).detach().cpu().numpy()

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



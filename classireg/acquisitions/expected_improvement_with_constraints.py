import torch
import math
from torch import Tensor
from typing import Optional, List
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
import numpy as np # These two libraries need to dissapear, as allt he code should be in torch
from scipy.special import erf
from botorch.optim import optimize_acqf
from .acquisition_base_cons import AcquisitionBaseToolsConstrained
import pdb
from torch.distributions.normal import Normal
from botorch.models import ModelListGP
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

class ExpectedImprovementWithConstraints(AcquisitionBaseToolsConstrained,ConstrainedExpectedImprovement):
	def __init__(self, model_list: List[Model], constraints, options: dict) -> None:
		
		# best_f = torch.min(model_list.models[0].train_targets)

		# Initialize parent classes inthe following order:
		ConstrainedExpectedImprovement.__init__(self, model=model_list, best_f=0.0, objective_index=0, constraints=constraints, maximize=False)

		AcquisitionBaseToolsConstrained.__init__(self, model_list=model_list, Nrestarts_eta_c=options.optimization.Nrestarts)

		logger.info("Starting EIC ...")

		self.dim = model_list.models[idxm['cons']].dim
		self.Nrestarts = options.optimization.Nrestarts
		self.algo_name = options.optimization.algo_name
		self.constrained_opt = OptimizationNonLinear(	dim=self.dim,
																									fun_obj=self.forward,
																									algo_str=self.algo_name,
																									bounds=[ [0.0]*self.dim, [1.0]*self.dim ],
																									minimize=False,
																									what2optimize_str="EIC acquisition")
		# self.use_nlopt = False
		self.disp_info_scipy_opti = options.optimization.disp_info_scipy_opti

		self._rho_conserv = options.prob_satisfaction
		self.x_next, self.alpha_next = None, None
		self.only_prob = False

		# pdb.set_trace()

	@property
	def rho_conserv(self):
		return self._rho_conserv	

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

		means, sigmas = self._get_posterior_reimplemented(X)

		# (b) x 1
		mean_obj = means[..., [self.objective_index]]
		sigma_obj = sigmas[..., [self.objective_index]]

		# print("mean_obj.shape:",mean_obj.shape)
		# print("sigma_obj.shape:",sigma_obj.shape)
		# print("means.shape:",means.shape)
		# print("sigmas.shape:",sigmas.shape)
		
		# Probability of feasibility:
		prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)

		# print("prob_feas.shape:",prob_feas.shape)

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

		val = ei_times_prob.squeeze(dim=-1)
		if val.dim() == 1 and len(val) == 1 or val.dim() == 0:
			val = val.item()
		# else:
		# 	pdb.set_trace()

		# print("X.shape:",X.shape)
		# print("val:",val)

		return val

	def _get_posterior_reimplemented(self, X: Tensor) -> Tensor:

		# Objective is assumed to be in the index 0. Constraints in the rest
		# Objective is assumed to be a classireg.models.GPmodel object
		# Constraints are assumed to be a classireg.models.GPCRmodel object
		
		means = torch.zeros([X.shape[0],self.model_list.num_outputs])
		sigmas = torch.zeros([X.shape[0],self.model_list.num_outputs])
		# pdb.set_trace()
		for k in range(self.model_list.num_outputs):
			means[...,k] 	= self.model_list.models[k].posterior(X).mean.squeeze()
			sigmas[...,k] = self.model_list.models[k].posterior(X).variance.squeeze().sqrt().clamp_min(1e-9)  # (b) x m
			# means[...,k] 	= self.model_list.models[k].posterior(X.view(1,self.dim)).mean.squeeze()
			# sigmas[...,k] = self.model_list.models[k].posterior(X.view(1,self.dim)).variance.squeeze().sqrt().clamp_min(1e-9)  # (b) x m

		return means, sigmas		

	def get_best_constrained_evaluation(self):

		Ycons_safe = self.model.models[idxm['cons']].train_ys
		Yobj_safe = self.model.models[idxm['obj']].train_targets # Since we don't include the non-stable evaluations in GPCR, the safe evaluations are the evaluations themselves
		if len(Ycons_safe) > 0:
			return torch.min(Yobj_safe).view(1)
		else:
			return None

	def get_next_point(self) -> (Tensor, Tensor):

		if self.model.models[idxm["obj"]].train_targets is None: # No safe evaluations case
			self.eta_c = torch.zeros(1,device=device,dtype=dtype)
			self.x_eta_c = torch.zeros((1,self.dim),device=device,dtype=dtype)
			self.only_prob = True
		else:
			
			# The following functions need to be called in the given order:
			try:
				self.update_eta_c(rho_t=self.rho_conserv) # Update min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t
			except Exception as inst:
				logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
				logger.info("Not optimizing eta_c ...")

			# self.best_f = self.eta_c
			self.best_f = self.get_best_constrained_evaluation() - self.model.models[idxm["obj"]].likelihood.noise.sqrt()[0].view(1)
			self.only_prob = False

		self.x_next, self.alpha_next = self.get_acqui_fun_maximizer()

		# Prevent from getting stuck into global minima:
		close_points,_ = self.model_list.models[idxm["cons"]]._identify_stable_close_to_unstable(	X_sta=self.x_next.cpu().numpy(),
																									X_uns=self.model_list.models[idxm["cons"]].train_x_sorted.cpu().numpy(),
																									top_dist=math.sqrt(self.dim)*0.005,
																									verbosity=False)
		if len(close_points) > 0:
			logger.info("Changed the evaluation to random as it was very close to an existing evaluation, within math.sqrt(self.dim)*0.005 = {0:f}".format(math.sqrt(self.dim)*0.005))
			self.x_next = draw_sobol_samples(bounds=torch.Tensor([[0.0]*self.dim,[1.0]*self.dim]),n=1,q=1).view(-1,self.dim)
			

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
		batch_limit = 2
		# batch_limit = 50 # This is a super bad idea for GPCR.
		options = {"batch_limit": batch_limit,"maxiter": 300,"ftol":1e-6,"method":"L-BFGS-B","iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}

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

	def get_probability_of_safe_evaluation(self, X: Tensor) -> Tensor:
		"""

		Code borrowed from botorch.acquisition.analytic.ConstrainedExpectedImprovement.forward()
		"""
		
		# posterior = super()._get_posterior(X=X)
		# means = posterior.mean.squeeze(dim=-2)  # (b) x m
		# sigmas = posterior.variance.squeeze(dim=-2).sqrt().clamp_min(1e-9)  # (b) x m


		means, sigmas = self._get_posterior_reimplemented(X)
		prob_feas = self._compute_prob_feas(X=X, means=means, sigmas=sigmas)
	
		return prob_feas

	def update_eta_c(self, rho_t):
		"""
		Search the constrained minimum of the posterior mean, i.e.,
		min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t
		If no safe area has been found yet, return the best obserbation of f(x) collected so far.
		
		NOTE: Normally, rho_t should be set to the conservative (safe) value, e.g., rho_t = 0.99
		"""

		if self._does_exist_at_least_one_safe_area(rho_t):
			self.x_eta_c, self.eta_c = self.find_eta_c(rho_t)
		elif self.model.models[1].train_xs.shape[0] > 0: # If there exists a safe evaluation but not a safe area:
			self.x_eta_c, self.eta_c = self.find_eta_c(0.0)
		else:
			self.x_eta_c, self.eta_c = None, None
		# else:
		# 	ind_min = torch.argmin(self.model_list.models[idxm['obj']].train_targets)
		# 	self.x_eta_c = self.model_list.models[idxm['obj']].train_inputs[0][ind_min,:].view((1,self.dim))
		# 	self.eta_c = self.model_list.models[idxm['obj']].train_targets[ind_min].view(1)

	def _does_exist_at_least_one_safe_area(self, rho_t):
		"""
		Check if at least one of the collected evaluations of the constraint is such that the probabilistic constraint is satisfied.
		If not, we can be sure the constraint is violated everywhere, and it won't make sense to run self.find_eta_c(rho_t)
		
		NOTE: Normally, rho_t should be set to the conservative (safe) value, e.g., rho_t = 0.99
		"""

		train_inputs = self.model_list.models[idxm['cons']].train_inputs[0]
		prob_feas = self.get_probability_of_safe_evaluation(train_inputs)
		exist_safe_areas = torch.any(prob_feas > rho_t)
		return exist_safe_areas


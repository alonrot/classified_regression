from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, MultitaskGaussianLikelihood
# from gpytorch.constraints import GreaterThan
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, UnwhitenedVariationalStrategy
# from gpytorch.models import ApproximateGP
# from botorch.fit import fit_gpytorch_model
# import random
import numpy as np
import torch
from torch import Tensor # alias from FloatTensor
import pdb
# from gpytorch.priors import GammaPrior
# import matplotlib.pyplot as plt
# from typing import Optional, Tuple
np.set_printoptions(linewidth=10000)
from classireg.utils.plotting_collection import PlotProbability
from classireg.varinf.expectation_propagation import ExpectationPropagation
from classireg.varinf.marginal_moments_EP_unbounded_hyperrectangle import marginal_moments_EP_unbounded_hyperrectangle
from classireg.utils.gaussian_tools import GaussianTools
from classireg.utils.parsing import get_logger
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from botorch.models.gpytorch import GPyTorchModel, BatchedMultiOutputGPyTorchModel
from botorch.utils import t_batch_mode_transform
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
import matplotlib.pyplot as plt
from gpytorch.models.gp import GP
from botorch.utils.sampling import draw_sobol_samples
# from torch.distributions.beta import Beta
# from torch.distributions.gamma import Gamma
from scipy.stats import gamma, beta
from classireg.models.mll_gpcr import MLLGPCR
from classireg.utils.optimize import OptimizationNonLinear
import scipy.linalg as la
import math

# class GPCRmodel(GPyTorchModel,GP):
class GPCRmodel(BatchedMultiOutputGPyTorchModel,GP):

	def __init__(self, dim: int, train_x: Tensor, train_yl: Tensor, options):
		"""
			train_X: A `batch_shape x n x d` tensor of training features.
			train_Y: A `batch_shape x n x m` tensor of training observations.
			train_Yvar: A `batch_shape x n x m` tensor of observed measurement noise.
		"""

		# Initialize parent class:
		super().__init__() # This is needed because torch.nn.Module, which is parent of GPyTorchModel, needs it

		print("\n")
		logger.info("### Initializing GPCR model for constraint g(x) ###")

		self.discard_too_close_points = options.discard_too_close_points

		self.dim = dim
		assert self.dim == train_x.shape[1], "The input dimension must agree with train_x"
		self.train_x = torch.tensor([],device=device, dtype=dtype, requires_grad=False)
		self.train_yl = torch.tensor([],device=device, dtype=dtype, requires_grad=False)
		self.update_XY(train_x,train_yl)

		# One output
		# ==========
		# pdb.set_trace()
		self._validate_tensor_args(X=self.train_xs, Y=self.train_ys.view(-1,1))
		# validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
		self._set_dimensions(train_X=self.train_xs, train_Y=self.train_ys.view(-1,1))
		# self.train_xs,_,_ = self._transform_tensor_args(X=self.train_xs, Y=self.train_ys)

		# # Two outputs
		# # ===========		
		# # pdb.set_trace()
		# self._validate_tensor_args(X=self.train_xs, Y=self.train_yl)
		# # validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
		# self._set_dimensions(train_X=self.train_xs, train_Y=self.train_yl)
		# # self.train_xs,_,_ = self._transform_tensor_args(X=self.train_xs, Y=self.train_ys)

		# Initialize hyperpriors using scipy because gpytorch's gamma and beta distributions do not have the inverse CDF
		hyperpriors = dict(	lengthscales = eval(options.hyperpars.lenthscales.prior),
												outputscale = eval(options.hyperpars.outputscale.prior),
												threshold = eval(options.hyperpars.threshold.prior))

		# Index hyperparameters:
		self.idx_hyperpars = dict(lengthscales=list(range(0,self.dim)) , outputscale=[self.dim] , threshold=[self.dim+1])
		self.dim_hyperpars = sum( [ len(val) for val in self.idx_hyperpars.values() ] )

		# Get bounds:
		self.hyperpars_bounds = self._get_hyperparameters_bounds(hyperpriors)
		logger.info("hyperpars_bounds:" + str(self.hyperpars_bounds))

		# Define meand and covariance modules with dummy hyperparameters
		self.mean_module = ZeroMean()
		self.covar_module = ScaleKernel(base_kernel=MaternKernel(nu=2.5,ard_num_dims=self.dim,lengthscale=0.1*torch.ones(self.dim)),outputscale=10.0)

		# # If non-zero mean, constant mean is assumed:
		# if "constant" in dir(self.mean_module):
		# 	self.__threshold = self.mean_module.constant
		# else:
		# 	self.__threshold = 0.0

		# If non-zero mean, constant mean is assumed:
		if "constant" in dir(self.mean_module):
			self.__threshold = self.mean_module.constant
			self.thres_init = self.mean_module.constant
		else:
			self.__threshold = options.hyperpars.threshold.init
			self.thres_init = options.hyperpars.threshold.init
		
		# Get a hyperparameter sample within bounds (not the same as sampling from the corresponding priors):
		hyperpars_sample = self._sample_hyperparameters_within_bounds(Nsamples=1).squeeze(0)
		self.covar_module.outputscale = hyperpars_sample[self.idx_hyperpars["outputscale"]]
		print("self.covar_module.outputscale:",str(self.covar_module.outputscale))
		self.covar_module.base_kernel.lengthscale = hyperpars_sample[self.idx_hyperpars["lengthscales"]]
		self.threshold = hyperpars_sample[self.idx_hyperpars["threshold"]]
		self.noise_std = options.hyperpars.noise_std.value # The evaluation noise is fixed, and given by the user
		
		self.gauss_tools = GaussianTools()

		# Initialize EP
		self.ep = ExpectationPropagation(	prior_mean=self.mean_module(train_x).cpu().detach().numpy(),
																			prior_cov=self.covar_module(train_x).cpu().detach().numpy(),
																			Maxiter=options.ep.maxiter,required_precission=options.ep.prec,verbosity=options.ep.verbo)

		# Initialize marginal log likelihood for the GPCR model.
		# mll_objective is callable
		# MLLGPCR can internally modify the model hyperparameters, and will do so throughout the optimization routine
		self.mll_objective = MLLGPCR(model_gpcr=self, hyperpriors=hyperpriors)

		# Define nlopt optimizer:
		self.opti = OptimizationNonLinear(dim=self.dim_hyperpars,
																			fun_obj=self.mll_objective,
																			algo_str=options.hyperpars.optimization.algo_name,
																			tol_x=1e-3,
																			Neval_max_local_optis=options.hyperpars.optimization.Nmax_evals,
																			bounds=self.hyperpars_bounds,
																			what2optimize_str="GPCR hyperparameters")

		# Extra parameters:
		self.top_dist_ambiguous_points = 0.5*torch.min(self.covar_module.base_kernel.lengthscale).item()
		self.factor_heteroscedastic_noise = 10**4

		# Update hyperparameters:
		self.Nrestarts_hyperpars = options.hyperpars.optimization.Nrestarts
		self._update_hyperparameters(Nrestarts=self.Nrestarts_hyperpars)

		# self.likelihood = FixedNoiseGaussianLikelihood(noise=torch.eye())
		self.likelihood = None

		# nearest_points: [array([0, 0]), array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4]), array([5, 5]), array([6, 6]), array([7, 7]), array([8, 8]), array([ 9,  9, 12, 14]), array([10, 10]), array([11, 11]), array([12,  9, 12, 14, 15, 16]), array([13, 13]), array([14,  9, 12, 14, 15, 16]), array([15, 12, 14, 15, 16]), array([16, 12, 14, 15, 16])]
		# [classireg.acquisitions.acquisition_base_cons] Starting AcquisitionBaseTools ...

		# Keep for compatibility with BOtorch acquisition functions:
		# pdb.set_trace()
		# _replace(v=node.v)
		# self.num_outputs = 1

		# self.eval()

	@property
	def threshold(self):
		return self.__threshold

	@threshold.setter
	def threshold(self,value):
		"""

		TODO/NOTE: This function adds the desired threshold value to another pre-existing value.
		This makes sense when self.threshold is set as the result of hyperparameter optimization,
		since therein the quantity learned is an increment over torch.max(self.train_ys).
		However, in general, setting self.threshold manually is a BAD idea. This
		should be changed.
		"""
		if len(self.train_ys) > 0: # If there exist safe evaluations
			self.__threshold = torch.max(self.train_ys) + value
		else:
			# self.__threshold = value
			self.__threshold = self.thres_init + value
	
	@threshold.getter
	def threshold(self):
		return self.__threshold
	
	def forward(self,x_in: Tensor) -> MultivariateNormal:
		"""

		This method is not strictly needed, because we won't train this model
		as a NN, nor use autograd for it. However, it's left here for compatibility
		and also used in a few places.
		"""

		return MultivariateNormal(mean=self.mean_module(x_in),covariance_matrix=self.covar_module(x_in))

	# @t_batch_mode_transform(expected_q=1)
	def predictive(self,x_in):

		# A `num_restarts x q x d` tensor of initial conditions.

		# print("")
		# print("x_in.shape: "+str(x_in.shape))
		# mean_shape = x_in.shape[:-1]
		x_in = self._error_checking_x_in(x_in)
		# # print("x_in.shape: "+str(/x_in.shape))
		# print("x_in:",x_in)

		if self.train_x_sorted.shape[0] == 0: # No data case
			return self.forward(x_in)
		else:
			with torch.no_grad():
				k_Xxp = self.covar_module(self.train_x_sorted,x_in).evaluate()
				k_xpxp = self.covar_module(x_in).evaluate()
				# K_XX_inv_k_Xxp = torch.solve(input=k_Xxp,A=self.Kprior_cov.evaluate() + 1e-6*torch.eye(self.train_x_sorted.shape[0]))[0]
				K_XX_inv_k_Xxp = torch.solve(input=k_Xxp,A=self.Kprior_cov.evaluate())[0]

				# mean_pred = K_XX_inv_k_Xxp.T.dot(self.expectation_posterior)
				# mean_pred = torch.matmul(K_XX_inv_k_Xxp.t(),self.expectation_posterior[:,None])
				mean_pred = torch.matmul(K_XX_inv_k_Xxp.t(),self.expectation_posterior)
				# cov_pred 	= k_xpxp - k_Xxp.T.dot(K_XX_inv_k_Xxp) + K_XX_inv_k_Xxp.T.dot(self.covariance_posterior).dot(K_XX_inv_k_Xxp)
				# cov_pred 	= k_xpxp - torch.matmul(k_Xxp.t(),K_XX_inv_k_Xxp) + torch.chain_matmul(K_XX_inv_k_Xxp.t(),self.covariance_posterior,K_XX_inv_k_Xxp)
				cov_pred 	= k_xpxp - torch.matmul(k_Xxp.t(),K_XX_inv_k_Xxp) + torch.matmul(K_XX_inv_k_Xxp.t(),torch.matmul(self.covariance_posterior,K_XX_inv_k_Xxp))
				# cov_pred 	= k_xpxp - torch.matmul(k_Xxp.t(),K_XX_inv_k_Xxp) + torch.matmul(K_XX_inv_k_Xxp.t(),torch.matmul(self.covariance_posterior+1e-5*torch.eye(self.train_x_sorted.shape[0]),K_XX_inv_k_Xxp))
				
				cov_pred_numpy = cov_pred.cpu().numpy()
				# cov_pred_numpy = self.gauss_tools.fix_singular_matrix(cov_pred_numpy,verbosity=False,what2fix="Fixing prior cov...") # DBG: TEMPORARY TRIAL; NOT ADDING NOISE
				cov_pred = torch.from_numpy(cov_pred_numpy).to(device=device,dtype=dtype)
				# pdb.set_trace()
				# cov_pred += 1e-4*torch.eye(cov_pred.shape[0])

		"""
		Re-shape mean
		
		TODO: This might not be needed anymore, since we're using _get_posterior_reimplemented in EIC
		"""
		if x_in.dim() == 3:
			batch_shape = torch.Size([1])
		elif x_in.dim() == 2:
			batch_shape = torch.Size([])
		else:
			raise ValueError("No way")
		test_shape = torch.Size([x_in.shape[0]])
		mean_pred = mean_pred.view(*batch_shape, *test_shape).contiguous()

		try:
			mvn = MultivariateNormal(mean=mean_pred,covariance_matrix=cov_pred)
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			# pdb.set_trace() # DBG: TEMPORARY TRIAL; NOT ADDING NOISE
			mvn = MultivariateNormal(mean=mean_pred,covariance_matrix=cov_pred + 1e-6*torch.eye(x_in.shape[0]))

		# print("mean_pred.shape:"+str(mean_pred.shape))
		# print("cov_pred.shape:"+str(cov_pred.shape))
		# print("mvn.batch_shape: "+str(mvn.batch_shape))
		# print("mvn.event_shape: "+str(mvn.event_shape))
		# print("mvn:",mvn)

		return mvn

	def update_XY(self,x_eval,yl_eval):
		'''
		x_eval [1 x dim]: A single points
		yl_eval [2,]: evaluation and label, i.e., 
			y_eval = yl_eval[0]
			l_eval = yl_eval[1]
		'''

		# Append datapoint:
		self.train_x = torch.cat([self.train_x, x_eval],dim=0)
		self.train_yl = torch.cat([self.train_yl, yl_eval],dim=0)

		# Update internal variables:
		logger.info("Updating after adding new data point...")
		self._update_subsets()

		# # Update hyperparameters
		# # Internally, this also updates the GPCR approximate posterior, so
		# # we do not need to call self._update_approximate_posterior() again
		# if learn_hyperparameters:
		# 	self._update_hyperparameters(Nrestarts=self.Nrestarts_hyperpars)

	def _update_subsets(self):

		self.train_xu = self.train_x[self.train_yl[:,1] == -1,:] # Training set with unsafe points
		self.train_xs = self.train_x[self.train_yl[:,1] == +1,:] # Training set with safe points
		self.train_ys = self.train_yl[self.train_yl[:,1] == +1,0]

		if self.discard_too_close_points:

			# Eliminate points that are *extremely* close to each other, to avoid numerical unstability
			ind_stay_in_stable = self.discard_points_that_are_too_close_to_avoid_numerical_unstability("stable")
			ind_stay_in_unstable = self.discard_points_that_are_too_close_to_avoid_numerical_unstability("unstable")
			# if torch.any(~ind_stay_in_stable) or torch.any(~ind_stay_in_unstable):
			# 	pdb.set_trace()

			self.train_xs = self.train_xs[ind_stay_in_stable,:]
			self.train_xu = self.train_xu[ind_stay_in_unstable,:]
			self.train_ys = self.train_ys[ind_stay_in_stable]

		# Concatenate both inputs:
		self.train_x_sorted = torch.cat([self.train_xs,self.train_xu],dim=0) # Sorted training set

		# For compatibility, although not needed:
		self.train_inputs = [self.train_x]
		self.train_targets = self.train_yl

	def discard_points_that_are_too_close_to_avoid_numerical_unstability(self,do_it_with="stable",debug=False):
		"""
		"""

		if do_it_with == "stable":
			close_points,_ = self._identify_stable_close_to_unstable(X_sta=self.train_xs.cpu().numpy(),X_uns=self.train_xs.cpu().numpy(),
																																																	top_dist=math.sqrt(self.dim)*0.02,
																																																	verbosity=False)
			Nels = self.train_xs.shape[0]
			train_x_new = self.train_xs.clone()
		else:
			close_points,_ = self._identify_stable_close_to_unstable(X_sta=self.train_xu.cpu().numpy(),X_uns=self.train_xu.cpu().numpy(),
																																																	top_dist=math.sqrt(self.dim)*0.02,
																																																	verbosity=False)
			Nels = self.train_xu.shape[0]
			train_x_new = self.train_xu.clone()
		

		ind_sel = torch.ones(Nels,dtype=bool,device=device)
		for k in range(len(close_points)):

			# A point will always be close to itself, so we skip this case:
			if len(close_points[k]) == 2:
				continue

			# If the current k is among the already discarded points, we skip this case:
			if torch.any(k == torch.tensor(range(Nels))[~ind_sel]):
				continue

			close_points_to_k = close_points[k][2::]
			# if self.train_yl[:,1][]

			ind_sel[close_points_to_k] = False # Starting at 2 assumes the points are sorted

		# train_x_new = train_x_new[ind_sel,:]
		# logger.info("\n")
		# logger.info(do_it_with)
		# logger.info("close_points: {0:s}".format(str(close_points)))
		# logger.info("ind_sel: {0:s}".format(str(ind_sel)))
		# if do_it_with == "stable":
		# 	logger.info("self.train_xs: {0:s}".format(str(self.train_xs)))
		# else:
		# 	logger.info("self.train_xu: {0:s}".format(str(self.train_xu)))
		if self.dim <= 2:
			logger.info("{0:s} points discarded ({1:d}): {2:s}".format(do_it_with,sum(~ind_sel),str(train_x_new[~ind_sel,:])))
		else:
			logger.info("{0:s} points discarded: {1:d}".format(do_it_with,sum(~ind_sel)))

		return ind_sel

	def display_hyperparameters(self):
		logger.info("    Evaluation noise (stddev) (fixed): | {0:2.4f}".format(self.noise_std))
		logger.info("  Re-optimized hyperparameters")
		logger.info("  ----------------------------")
		logger.info("    Outputscale (stddev) | {0:2.4f}".format(self.covar_module.outputscale.item()))
		logger.info("    Lengthscale(s)       | {0:s}".format(str(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten())))
		logger.info("    Optimal threshold    | {0:2.4f}".format(self.threshold.item()))

	def logging(self):
		log_out = dict()
		log_out["lengthscale"] = self.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
		log_out["outputscale"] = self.covar_module.outputscale.item()
		log_out["threshold"] = self.threshold.item()
		log_out["train_inputs"] = None if self.train_inputs is None else self.train_inputs[0].detach().cpu().numpy()
		log_out["train_targets"] = None if self.train_targets is None else self.train_targets.detach().cpu().numpy()
		log_out["train_xs"] = self.train_xs.detach().cpu().numpy()
		log_out["train_xu"] = self.train_xu.detach().cpu().numpy()
		log_out["train_ys"] = self.train_ys.detach().cpu().numpy()
		log_out["train_x_sorted"] = self.train_x_sorted.detach().cpu().numpy()

		return log_out

	def _update_prior(self):
		'''
		Recompute prior covariance matrix with the sorted inputs
		'''
		if self.train_x_sorted.shape[0] > 0:
			Kprior_cov = self.covar_module(self.train_x_sorted)
			self.Kprior_cov = Kprior_cov # DBG: TEMPORARY TRIAL; NOT ADDING NOISE
			# self.Kprior_cov = Kprior_cov + 1e-6*torch.eye(self.train_x_sorted.shape[0]) # DBG: TEMPORARY TRIAL; NOT ADDING NOISE
			# self.Kprior_cov = self.gauss_tools.fix_singular_matrix(Kprior_cov,verbosity=False,what2fix="Fixing prior cov...")
			# self.Kprior_cov = self.covar_module(self.train_x_sorted)
		else:
			self.Kprior_cov = None

	def _update_EP_object(self):
		'''
		This function assumes that self._update_prior() has been updated
		'''

		if len(self.train_ys) > 0:
			
			Sigma1_diag = self.noise_std**2 * np.ones(self.train_ys.shape[0])

			# Modify noise matrix if needed:
			if self.top_dist_ambiguous_points > 0.0:
				nearest_points_to_X_sta_i,nearest_points_to_X_uns_i = self._identify_stable_close_to_unstable(X_sta=self.train_xs.cpu().numpy(),
																																																			X_uns=self.train_xu.cpu().numpy(),
																																																			top_dist=self.top_dist_ambiguous_points,
																																																			verbosity=False)
				if len(nearest_points_to_X_sta_i) > 0: 
					str_banner = "<<<< Will modify the noise matrix >>>>"
					logger.info("="*len(str_banner))
					logger.info(str_banner)
					logger.info("="*len(str_banner))
					Sigma1_diag = self._modify_noise_matrix(nearest_points_to_X_sta_i,Sigma1_diag,factor=self.factor_heteroscedastic_noise,verbosity=False)

			mu1 = self.train_ys.cpu().numpy()
			Sigma1 = np.diag(Sigma1_diag)

		else:
			Sigma1 = mu1 = None

		# Product of Gaussians:
		D,m = self.gauss_tools.product_gaussian_densities_different_dimensionality(	mu1=mu1,
																																								Sigma1=Sigma1,
																																								mu12=np.zeros(self.train_x_sorted.shape[0]),
																																								Sigma12=self.Kprior_cov.cpu().numpy())

		# D = self.gauss_tools.fix_singular_matrix(D,verbosity=False,what2fix="Fixing D before updating the EP object")

		self.ep.restart(prior_mean=m,prior_cov=D)

	def _update_approximate_posterior(self):

		# if self.ep is None:
		# 	self.covariance_posterior = None
		# 	self.expectation_posterior = None

		# Nu = self.data['Nu']
		# Ns = self.data['Ns']
		# Xs = self.data['Xs']
		# Xu = self.data['Xu']
		# Y = self.data['Y']

		# # Check data existance:
		# if Nu == 0 and Ns == 0:
		# 	return None,None

		# Use c_opti to create the integration limits:
		lim_lower, lim_upper = self._create_integration_limits(self.train_yl, self.threshold)

		# # Modify integration limits if necessary:
		# if self.top_dist_ambiguous_points > 0.0:			
		# 	nearest_points_to_X_sta_i,nearest_points_to_X_uns_i = self._identify_stable_close_to_unstable(Xs,Xu,top_dist=self.top_dist_ambiguous_points,verbosity=False)
		# 	if len(nearest_points_to_X_uns_i) > 0: 
		# 		logger.info("\n==============================================\n <<<< Will modify the integration limits >>>>\n==============================================")
		# 	lim_lower = self._modify_integration_limits_for_ambiguous_points(nearest_points_to_X_uns_i,lim_lower,Ns,Nu,Ysta=Y,verbosity=False)

		try:
			self.covariance_posterior, self.expectation_posterior, self.logZ = self.ep.run_EP(marginal_moments_EP_unbounded_hyperrectangle,lim_lower.cpu().numpy(),lim_upper.cpu().numpy())

			# self.covariance_posterior += 1e-5*np.eye(self.covariance_posterior.shape[0]) # DBG: TEMPORARY TRIAL; NOT ADDING NOISE


			self.covariance_posterior = torch.from_numpy(self.covariance_posterior).to(device=device,dtype=dtype)
			self.expectation_posterior = torch.from_numpy(self.expectation_posterior).to(device=device,dtype=dtype)
		except Exception as inst:
			print(type(inst),inst.args)
			raise ValueError("EP failed when computing the posterior moments...")


	def _update_hyperparameters(self, Nrestarts=5):

		logger.info("Fitting GPCR model g(x) ...")
		logger.info("---------------------------")

		# Get random restarts:
		x_restarts = self._sample_hyperparameters_within_bounds(Nsamples=Nrestarts)
		# logger.info("x_restarts:" + str(x_restarts))
	
		# Store current hyperparameters, just in case the optimization below fails:
		outputscale = self.covar_module.outputscale.detach()
		lengthscale = self.covar_module.base_kernel.lengthscale.detach().flatten()
		threshold = self.threshold.detach()

		try:
			new_hyperpars, _ = self.opti.run_optimization(x_restarts=x_restarts)

			loss_new_hyperpars = self.mll_objective(new_hyperpars.flatten())
			logger.info("  Loss (new hyperparameters): {0:f}".format(loss_new_hyperpars.item()))

			self.display_hyperparameters()
		except Exception as inst:
			logger.info("  Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("  Hyperparameter optimization failed (!!) Keeping the old ones ...")
			try:
				loss_old_hyperpars = self.mll_objective.log_marginal(lengthscale, outputscale, threshold)
				logger.info("  Loss (old hyperparameters): {0:f}".format(loss_old_hyperpars.item()))
			except Exception as inst:
				logger.info("    Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
				logger.info("    Old hyperparameters do not work either. Setting some random ones ...")
				self.mll_objective(x_restarts[0,:].flatten())
			self.display_hyperparameters()

	def _get_hyperparameters_bounds(self,hyperpriors):

		# Compute the domain for hyperparameter search by truncating the support of the corresponding hyperprior at the .75 quantile
		# The lower bound is necessary for numerical stability, i.e., when computing logpdf() in classireg.models.mll_gpcr.log_marginal()
		# All values of the dictionary are defined as double lists
		hyperpriors_support = dict(	lengthscales=[[0.05]*self.dim,[hyperpriors["lengthscales"].ppf(.75)]*self.dim],
																outputscale=[[0.05],[hyperpriors["outputscale"].ppf(.75)]],
																threshold=[[0.05],[hyperpriors["threshold"].ppf(.75)]])

		# Automatically get the bounds from the dictionary:
		hyperpars_lb = []
		hyperpars_ub = []
		for hyperpar in hyperpriors_support.values():
			hyperpars_lb += hyperpar[0]
			hyperpars_ub += hyperpar[1]
		hyperpars_bounds = [hyperpars_lb, hyperpars_ub]

		return hyperpars_bounds

	def _sample_hyperparameters_within_bounds(self, Nsamples):

		# Get a sample from the prior for initialization:
		new_seed = torch.randint(low=0,high=100000,size=(1,)).item() # Top-level seeds have an impact on this one herein; contrary to the case new_seed = None
		hyperpars_restarts = draw_sobol_samples(bounds=torch.tensor(self.hyperpars_bounds), n=Nsamples, q=1, seed=new_seed)
		hyperpars_restarts = hyperpars_restarts.squeeze(1) # Remove batch dimension [n q dim] -> [n dim]

		return hyperpars_restarts

	def _create_integration_limits(self,train_yl,c):

		Ns = torch.sum(train_yl[:,1] == +1)
		Nu = torch.sum(train_yl[:,1] == -1)
		Neval = Ns + Nu

		# Limits of integration:
		lim_lower = torch.zeros(Neval)
		lim_upper = torch.zeros(Neval)
		for i in range(Ns):
			lim_lower[i] = -float("Inf")
			lim_upper[i] = c

		for i in range(Nu):
			lim_lower[Ns+i] = c
			lim_upper[Ns+i] = +float("Inf")

		return lim_lower, lim_upper

	def _modify_noise_matrix(self,nearest_points_to_X_sta_i,noise_diag,factor,verbosity=False):
		'''
		Modify the diagonal of the noise matrix
		noise_diag: It's a vector!
		'''

		# Error checking:
		Nsta_affected_points = len(nearest_points_to_X_sta_i)
		if Nsta_affected_points == 0:
			return noise_diag
		elif noise_diag is None:
			raise ValueError("noise_diag is None, but Nsta_affected_points != 0. Shouldn't noise_diag have a value?")
		elif noise_diag.ndim != 1:
			raise ValueError("noise_diag must be a vector")
		elif noise_diag.shape[0] == 0:
			raise ValueError("noise_diag must be a non-empty vector")
		else:
			noise_diag_out = noise_diag.copy()

		if factor < 1.0:
			raise ValueError("The factor, in principle, is meant to increase the noise in ambiguous regions")

		# Modify the diagonal:
		for k in range(Nsta_affected_points):
			ind_i = nearest_points_to_X_sta_i[k][0]
			noise_diag_out[ind_i] *= factor

		if verbosity == True:
			logger.info("noise_diag_out")
			logger.info("===============")
			logger.info(str(noise_diag_out))

		return noise_diag_out

	def _modify_integration_limits_for_ambiguous_points(self,nearest_points_to_X_uns_i,lim_lower_i,Nsta,Nuns,Ysta,verbosity=False):
		'''
		Modify the c threshold for those pair of points
		that are very close to each other
		'''

		# Error checking:
		Nuns_affected_points = len(nearest_points_to_X_uns_i)
		if Nuns_affected_points == 0:
			return lim_lower_i
		else:
			lim_lower_i_out = lim_lower_i.copy()

		# Modify points:
		for k in range(Nuns_affected_points):

			# Take index of the affected unstable point X_uns[ind_i,:] :
			ind_i = nearest_points_to_X_uns_i[k][0]

			# Take the corresponding indices from X_sta that are affecting X_uns[ind_i,:] 
			indices_j = nearest_points_to_X_uns_i[k][1::]

			# Modify the lower integration limits of f_uns: we assign the minimum
			# observed stable value among all the stable points that are affecting X_uns[ind,:]
			c_opti = lim_lower_i_out[Nsta+ind_i]
			pow_ = 2.0
			# pow_ = 1./2
			alpha = (1./(1.+len(indices_j)))**(pow_)
			lim_lower_i_out[Nsta+ind_i] = alpha*c_opti + (1.-alpha)*np.amin(Ysta[indices_j])

			# lim_lower_i_out[Nsta+ind_i] = np.amin(Ysta[indices_j])

		if verbosity == True:
			logger.info("\nlim_lower_i_out")
			logger.info("===============")
			logger.info(lim_lower_i_out)

		return lim_lower_i_out

	def _identify_stable_close_to_unstable(self,X_sta,X_uns,top_dist,verbosity=False):
		'''
		
		Inputs
		======
		X_sta: [Ns,D], where D is the input dimensionality, and Ns is the number of stable points
		X_uns: [Nu,D], where D is the input dimensionality, and Ns is the number of unstable points
		
		Outputs
		=======
		nearest_points_to_X_sta_i: list

		Explanation
		===========
		For all the stable points, is there any unstable point that is close enough?
		This method returns a list of arrays.
		For each point X_sta[i,:], we check how close are each one of the points X_uns[j,:]
		If there is at least one X_uns[j,:] that is close enough, we add a new array
		to the list. The first element of the array is the corresponding index i.
		The subsequent elements are the j indices such that norm(X_sta[i,:]-X_uns[j,:]) < top_dist
		If no point from X_uns is close enough to each element i of X_sta, the list will be empty.
		NOTE: we also do the same from the point of view of X_uns, and return it


		TODO: Consider reurning only nearest_points_to_X_uns_i, and 
		the first element of the array in each position i on the list, i.e. nearest_points_to_X_sta_i[i][0]
		The reason is that modify_integration_limits_for_ambiguous_points() uses needs only nearest_points_to_X_uns_i
		and _modify_noise_matrix() needs only nearest_points_to_X_sta_i[i][0]
		'''

		# If there's no stable or unstable values yet, we return an empty list:
		# if len(X_sta) == 0 or len(X_uns) == 0 or top_dist == 0.0:
		if X_sta is None or X_uns is None or top_dist == 0.0:
			return [], []
		elif top_dist < 0.0:
			raise NotImplementedError

		Ns 	= X_sta.shape[0]
		Nu 	= X_uns.shape[0]

		# Nearest points to X_sta:
		nearest_points_to_X_sta_i = []
		for i in range(Ns):
			norms_X_sta_i = la.norm(X_sta[i,:]-X_uns,ord=2,axis=1)
			ind_j, = np.where(norms_X_sta_i < top_dist)
			if len(ind_j) > 0:
				aux = np.insert(ind_j,0,i)
				nearest_points_to_X_sta_i.append(aux)

		# Nearest points to X_uns:
		nearest_points_to_X_uns_i = []
		for i in range(Nu):
			norms_X_uns_i = la.norm(X_uns[i,:]-X_sta,ord=2,axis=1)
			ind_j, = np.where(norms_X_uns_i < top_dist)
			if len(ind_j) > 0:
				aux = np.insert(ind_j,0,i)
				nearest_points_to_X_uns_i.append(aux)

		if verbosity == True:
			logger.info("")
			logger.info("nearest_points_to_X_sta_i")
			logger.info("=========================")
			logger.info(str(nearest_points_to_X_sta_i))
			logger.info("nearest_points_to_X_uns_i")
			logger.info("=========================")
			logger.info(str(nearest_points_to_X_uns_i))
			logger.info("X_sta")
			logger.info("=========================")
			logger.info(str(X_sta))
			logger.info("X_uns")
			logger.info("=========================")
			logger.info(str(X_uns))

		return nearest_points_to_X_sta_i, nearest_points_to_X_uns_i


	def _error_checking_x_in(self,x_in: Tensor) -> None:
		
		assert not torch.any(torch.isnan(x_in)), "x_in cannot contain NaNs"
		if x_in.dim() == 1:
			x_in = x_in[None,:]
		assert x_in.shape[-1] == self.dim, "x_in must be N x self.dim, where N is the number of points and self.dim is the dimensionality"

		if x_in.dim() >= 3:
			return x_in.view(-1,self.dim)
		else:
			return x_in


	def __call__(self,x_in: Tensor):
		return self.predictive(x_in)

	def plot(self,axes=None,block=False,Ndiv=100,legend=True,title="GPgrad",plotting=True,plotCDF=False,clear_axes=False,Nsamples=None,ylabel=None,ylim=None,
							pause=None,showtickslabels_x=True,xlabel=None,labelsize=None,showtickslabels=None,showticks=None,linewidth=None,color=None,prob=False):
		'''
		This function hardcodes the plotting limits between zero and one for now
		'''
		if plotting == False or self.dim > 1:
			return

		pp = PlotProbability()
		xpred_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
		# xpred_vec = xpred_vec.unsqueeze(0) # Ndiv batches of [q=1 x self.dim] dimensions each

		# Compute one by one:
		logger.info("Computing posterior while plotting ... (!!)")
		post_batch = False
		if post_batch:

			# Predict:
			posterior = self.posterior(X=xpred_vec,observation_noise=False) # observation_noise MUST be always false; this class is not prepared otherwise
									# Internally, self.posterior(xpred_vec) calls self(xpred_vec), which calls self.predictive(xpred_vec)

			# pdb.set_trace()

			# Get upper and lower confidence bounds (2 standard deviations from the mean):
			lower_ci, upper_ci = posterior.mvn.confidence_region()

			# Posterior mean:
			mean_vec = posterior.mean
			std_vec = posterior.variance.sqrt()

		else:

			lower_ci = torch.zeros((Ndiv))
			upper_ci = torch.zeros((Ndiv))
			mean_vec = torch.zeros((Ndiv))
			std_vec = torch.zeros((Ndiv))
			for k in range(Ndiv):
				mvn = self.predictive(xpred_vec[k,:].view(-1,self.dim))
				lower_ci[k], upper_ci[k] = mvn.confidence_region()
				mean_vec[k] = mvn.mean
				std_vec[k] = mvn.variance.sqrt()


		if self.dim == 1:
			if prob == False:
				axes = pp.plot_GP_1D(	xpred_vec=xpred_vec.squeeze().cpu().numpy(),
														fpred_mode_vec=mean_vec.squeeze().detach().cpu().numpy(),
														fpred_quan_minus=lower_ci.squeeze().detach().cpu().numpy(),
														fpred_quan_plus=upper_ci.squeeze().detach().cpu().numpy(),
														X_uns=self.train_xu.detach().cpu().numpy(),
														X_sta=self.train_xs.detach().cpu().numpy(),
														Y_sta=self.train_ys.detach().cpu().numpy(),
														title=title,axes=axes,block=block,
														legend=legend,clear_axes=True,xlabel=None,ylabel=ylabel,xlim=np.array([0.,1.]),ylim=ylim,
														labelsize="x-large",legend_loc="upper left",colormap="paper",
														showtickslabels_x=showtickslabels_x)
			else:
				normal = Normal(loc=mean_vec.squeeze(),
												# scale=posterior.variance.sqrt().squeeze())
												scale=std_vec.squeeze())
				ei_cdf = normal.cdf(self.threshold)		
				# pdb.set_trace()
				axes = pp.plot_acquisition_function(var_vec=ei_cdf,xpred_vec=xpred_vec.cpu().numpy(),
																			xlabel=xlabel,ylabel=ylabel,title=title,legend=legend,axes=axes,clear_axes=True,
																			xlim=np.array([0.,1.]),block=block,labelsize=labelsize,showtickslabels=showtickslabels,showticks=showticks,
																			what2plot="",color=color,ylim=np.array([0.,1.1]),linewidth=linewidth)


			if Nsamples is not None:
				f_sample = posterior.sample(sample_shape=torch.Size([Nsamples]))
				for k in range(Nsamples):
					axes.plot(xpred_vec.squeeze().detach().cpu().numpy(),
											f_sample[k,:,0],linestyle="--",linewidth=1.0,color="sienna")
		
		elif self.dim == 2:
			pass

		plt.show(block=block)
		if pause is not None:
			plt.pause(pause)

		return axes




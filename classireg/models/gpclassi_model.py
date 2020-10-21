from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, ConstantMeanGrad, ZeroMean
# from gpytorch.models import ExactGP
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel, RBFKernelGrad, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, FixedNoiseGaussianLikelihood, BernoulliLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import VariationalELBO
from classireg.models.mll_gp import MLLGP
from botorch.fit import fit_gpytorch_model
from torch.optim import Adam
import random
import numpy as np
import torch
from torch import Tensor
import pdb
from classireg.utils.plotting_collection import PlotProbability
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions
# from classireg.utils.parsing import extract_prior
from botorch.gen import gen_candidates_scipy,get_best_candidates
from gpytorch.priors import GammaPrior
from scipy.stats import gamma, beta
import matplotlib.pyplot as plt
from classireg.utils.optimize import OptimizationNonLinear
from typing import Optional, Tuple
from botorch.optim import optimize_acqf
np.set_printoptions(linewidth=10000)
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class GPClassifier(ApproximateGP):

	_num_outputs = 1  # to inform GPyTorchModel API

	def __init__(self, dim: int, train_X: Tensor, train_Y: Tensor, options: dict, which_type: Optional[str] = "obj") -> None:

		variational_distribution = CholeskyVariationalDistribution(train_X.size(0))
		variational_strategy = UnwhitenedVariationalStrategy(self, train_X, variational_distribution, learn_inducing_locations=False)
		super(GPClassifier, self).__init__(variational_strategy)

		self.dim = dim

		# pdb.set_trace()
		if len(train_X) == 0: # No data case
			train_X = None
			train_Y = None
			self.train_inputs = None
			self.train_targets = None
			self.train_x = None
			self.train_yl = None
		else:
			# Error checking:
			assert train_Y.dim() == 1, "train_Y is required to be 1D"
			assert train_X.shape[-1] == self.dim, "Input dimensions do not agree ... (!)"
			self.train_inputs = [train_X.clone()]
			self.train_targets = train_Y.clone()
			self.train_x = train_X.clone()
			self.train_yl = torch.cat([torch.zeros((len(train_Y)),1) , train_Y.view(-1,1) ],dim=1)


		print("\n")
		logger.info("### Initializing GP classifier for constraint g(x) ###")

		# Likelihood:
		noise_std = options.hyperpars.noise_std.value
		self.likelihood = BernoulliLikelihood()

		# For compatibility:
		self.threshold = torch.tensor([float("Inf")])

		# Initialize hyperpriors using scipy because gpytorch's gamma and beta distributions do not have the inverse CDF
		hyperpriors = dict(	lengthscales = eval(options.hyperpars.lenthscales.prior),
												outputscale = eval(options.hyperpars.outputscale.prior))

		# Index hyperparameters:
		self.idx_hyperpars = dict(lengthscales=list(range(0,self.dim)) , outputscale=[self.dim] )
		self.dim_hyperpars = sum( [ len(val) for val in self.idx_hyperpars.values() ] )

		# Get bounds:
		self.hyperpars_bounds = self._get_hyperparameters_bounds(hyperpriors)
		logger.info("hyperpars_bounds:" + str(self.hyperpars_bounds))

		# Initialize prior mean:
		# self.mean_module = ConstantMean()
		self.mean_module = ZeroMean()

		# Initialize covariance function:
		base_kernel = MaternKernel(nu=2.5,ard_num_dims=self.dim,lengthscale=0.1*torch.ones(self.dim))
		self.covar_module = ScaleKernel(base_kernel=base_kernel)

		self.disp_info_scipy_opti = True

		# Get a hyperparameter sample within bounds (not the same as sampling from the corresponding priors):
		hyperpars_sample = self._sample_hyperparameters_within_bounds(Nsamples=1).squeeze(0)
		self.covar_module.outputscale = hyperpars_sample[self.idx_hyperpars["outputscale"]]
		self.covar_module.base_kernel.lengthscale = hyperpars_sample[self.idx_hyperpars["lengthscales"]]
		self.noise_std = options.hyperpars.noise_std.value # The evaluation noise is fixed, and given by the user

		self.Nrestarts = options.hyperpars.optimization.Nrestarts

		self._update_hyperparameters()

		self.eval()
		self.likelihood.eval()

		# pdb.set_trace()

	def set_hyperparameters(self,lengthscale,outputscale,noise):
		self.covar_module.base_kernel.lengthscale = lengthscale
		self.covar_module.outputscale = outputscale
		# self.likelihood.noise[:] = noise
		# self.mean_module.constant[:] = 0.0 # Assume zero mean

	def display_hyperparameters(self):
		logger.info("  Re-optimized hyperparameters")
		logger.info("  ----------------------------")
		logger.info("    Outputscale (stddev) | {0:2.4f}".format(self.covar_module.outputscale.item()))
		logger.info("    Lengthscale(s)       | " + str(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()))

	def logging(self):
		log_out = dict()
		log_out["lengthscale"] = self.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
		log_out["outputscale"] = self.covar_module.outputscale.item()
		# log_out["noise"] = self.likelihood.noise.detach().cpu().numpy()
		log_out["train_inputs"] = None if self.train_inputs is None else self.train_inputs[0].detach().cpu().numpy()
		log_out["train_targets"] = None if self.train_targets is None else self.train_targets.detach().cpu().numpy()

		return log_out

	def _update_hyperparameters(self):

		# Find optimal model hyperparameters
		self.train()
		self.likelihood.train()

		# Use the adam optimizer
		optimizer = Adam(self.parameters(), lr=0.1)

		# "Loss" for GPs - the marginal log likelihood
		# num_data refers to the number of training datapoints
		mll = VariationalELBO(self.likelihood, self, self.train_targets.numel())

		training_iterations = 50
		for i in range(training_iterations):
			# Zero backpropped gradients from previous iteration
			optimizer.zero_grad()
			# Get predictive output
			output = self(self.train_inputs[0])
			# Calc loss and backprop gradients
			loss = -mll(output, self.train_targets)
			loss.backward()
			# print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
			optimizer.step()

	def _optimize_acqui_use_restarts_individually(self):

		# Get initial random restart points:
		logger.info("  Generating random restarts ...")
		options={"maxiter": 200,"ftol":1e-9,"method":"L-BFGS-B","iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}
		bounds = torch.tensor(self.hyperpars_bounds,device=device,dtype=dtype)
		initial_conditions = gen_batch_initial_conditions(acq_function=self.mll_objective,bounds=bounds,q=1,
																num_restarts=self.Nrestarts,raw_samples=500, options=options)

		logger.info("  Optimizing loss function with {0:d} restarts ...".format(self.Nrestarts))
		new_hyperpars_many = torch.zeros(size=(self.Nrestarts,1,self.dim_hyperpars))
		new_hyperpars_loss_many = torch.zeros(size=(self.Nrestarts,))

		new_hyperpars, _ = self.opti_hyperpars.run_optimization(x_restarts=initial_conditions.view(self.Nrestarts,self.dim_hyperpars))
		
		logger.info("  Done!")
		
		return new_hyperpars

	def _get_hyperparameters_bounds(self,hyperpriors):

		# Compute the domain for hyperparameter search by truncating the support of the corresponding hyperprior at the .75 quantile
		# The lower bound is necessary for numerical stability, i.e., when computing logpdf() in classireg.models.mll_gpcr.log_marginal()
		# All values of the dictionary are defined as double lists
		hyperpriors_support = dict(	lengthscales=[[0.001]*self.dim,[hyperpriors["lengthscales"].ppf(.75)]*self.dim],
																outputscale=[[0.001],[hyperpriors["outputscale"].ppf(.75)]])

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

	def forward(self, x):

		# A `num_restarts x q x d` tensor of initial conditions.

		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		mvn = MultivariateNormal(mean_x, covar_x)
		return mvn

	def plot(self,axes=None,block=False,Ndiv=100,legend=True,title="GPgrad",plotting=True,plotCDF=False,clear_axes=False,Nsamples=None,ylabel=None,ylim=None,
							pause=None,showtickslabels_x=True,xlabel=None,labelsize=None,showtickslabels=None,showticks=None,linewidth=None,color=None,prob=False):

		'''
		This function hardcodes the plotting limits between zero and one for now
		'''
		if plotting == False or self.dim > 1:
			return

		pp = PlotProbability()
		xpred_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
		xpred_vec = xpred_vec.unsqueeze(0) # Ndiv batches of [q=1 x self.dim] dimensions each

		mvn_cons = self(xpred_vec)
		pred_lik = self.likelihood(mvn_cons)
		mean_vec = pred_lik.mean
		
		# Get upper and lower confidence bounds (2 standard deviations from the mean):
		var_vec = pred_lik.variance
		std_vec = var_vec.sqrt()
		lower_ci, upper_ci = mean_vec - std_vec, mean_vec + std_vec

		if self.dim == 1:
			axes = pp.plot_GP_1D(	xpred_vec=xpred_vec.squeeze().cpu().numpy(),
														fpred_mode_vec=mean_vec.squeeze().detach().cpu().numpy(),
														fpred_quan_minus=lower_ci.squeeze().detach().cpu().numpy(),
														fpred_quan_plus=upper_ci.squeeze().detach().cpu().numpy(),
														X_sta=None if self.train_inputs is None else self.train_inputs[0].detach().cpu().numpy(),
														Y_sta=None if self.train_targets is None else self.train_targets.detach().cpu().numpy(),
														title=title,axes=axes,block=block,
														legend=legend,clear_axes=True,xlabel=xlabel,ylabel=ylabel,xlim=np.array([0.,1.]),ylim=ylim,
														labelsize="x-large",legend_loc="best",colormap="paper",
														showtickslabels_x=showtickslabels_x)

			if Nsamples is not None:
				f_sample = posterior.sample(sample_shape=torch.Size([Nsamples]))
				for k in range(Nsamples):
					axes.plot(xpred_vec.squeeze().detach().cpu().numpy(),
											f_sample[k,0,:,0],linestyle="--",linewidth=1.0,color="sienna")
		
		elif self.dim == 2:
			pass

		plt.show(block=block)
		if pause is not None:
			plt.pause(pause)

		return axes




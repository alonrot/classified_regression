import torch
from torch import Tensor
from typing import Optional, List
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
import numpy as np # These two libraries need to dissapear, as allt he code should be in torch
from scipy.special import erf
from botorch.optim import optimize_acqf
from .acquisition_base import AcquisitionBaseTools
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

class ExpectedImprovementVanilla(AcquisitionBaseTools,ExpectedImprovement):
	def __init__(self, model: Model, options: dict) -> None:
		
		# best_f = torch.min(model_list.models[0].train_targets)

		# Initialize parent classes inthe following order:
		ExpectedImprovement.__init__(self, model=model, best_f=0.0, maximize=False)

		AcquisitionBaseTools.__init__(self, model=model, Nrestarts_eta=options.optimization.Nrestarts)

		logger.info("Starting EI ...")

		self.dim = model.dim
		self.Nrestarts = options.optimization.Nrestarts
		self.algo_name = options.optimization.algo_name
		self.constrained_opt = OptimizationNonLinear(	dim=self.dim,
																									fun_obj=self.forward,
																									algo_str=self.algo_name,
																									bounds=[ [0.0]*self.dim, [1.0]*self.dim ],
																									minimize=False,
																									what2optimize_str="EI acquisition")
		# self.use_nlopt = False
		self.disp_info_scipy_opti = options.optimization.disp_info_scipy_opti
		self.method = "L-BFGS-B"

		self.x_next, self.alpha_next = None, None

	def get_next_point(self):

		# Find and store the minimum of the posterior mean, i.e., min_x mu(x|D), where D is the data set D={Y,X}, and mu(x|D)
		# is the posterior mean of the GP queried at location x
		if self.model.train_targets is None: # No safe evaluations case
			self.eta = torch.zeros(1,device=device,dtype=dtype)
			self.x_eta = torch.zeros((1,self.dim),device=device,dtype=dtype)
		else:
			# The following functions need to be called in the given order:
			self.x_eta, self.eta = self.find_eta() # Update min_x mu(x|D)
			self.best_f = self.eta

		# self.best_f = self.eta
		self.best_f = torch.tensor([torch.min(self.model.train_targets).item() - self.model.likelihood.noise.sqrt()[0].item()],device=device,dtype=dtype)

		logger.info("[get_next_point()] Computing next candidate by maximizing the acquisition function ...")
		options={"batch_limit": 50,"maxiter": 200,"ftol":1e-9,"method":self.method,"iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}

		if self.dim > 2:
			self.x_next, self.alpha_next = self.optimize_acqui_use_restarts_as_batch(options)
		else:
			self.x_next, self.alpha_next = self.optimize_acqui_use_restarts_individually(options)
		
		logger.info("Done!")
		if self.x_next is not None and  self.alpha_next is not None:
			logger.info("xnext: " + str(self.x_next.view((1,self.dim)).detach().cpu().numpy()))
			logger.info("alpha_next: {0:2.2f}".format(self.alpha_next.item()))
		else:
			logger.info("xnext: None")
			logger.info("alpha_next: None")

		logger.info("self.x_eta: "+str(self.x_eta))
		logger.info("self.eta: "+str(self.eta))
		logger.info("self.best_f: "+str(self.best_f))

		return self.x_next,self.alpha_next





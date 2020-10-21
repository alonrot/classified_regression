import torch
from botorch.test_functions.synthetic import Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
from classireg.models.gpcr_model import GPCRmodel
from classireg.utils.parsing import get_logger
from classireg.utils.plotting_collection import plotting_tool_uncons
import logging
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
import pdb
INF = -float("inf")
from botorch.fit import fit_gpytorch_model
from botorch.utils.sampling import manual_seed
from botorch.acquisition import ExpectedImprovement, ConstrainedExpectedImprovement
from classireg.acquisitions.expected_improvement_with_constraints import ExpectedImprovementWithConstraints
import hydra
from omegaconf import DictConfig
from botorch.models import ModelListGP

def test_optimize_pytorch(dim):

	train_x, train_yl = get_initial_evaluations(dim=dim)

	Neval = train_x.shape[0]
	dim = train_x.shape[1]

	gpcr = GPCRmodel(train_x=train_x, train_yl=train_yl, noise_std=0.01)

	mll = MLLGPCR_pytorch(likelihood=None,model=gpcr)

	mll(None,None)

	fit_gpytorch_model(mll,max_retries=10) # See https://botorch.org/api/_modules/botorch/optim/utils.html#sample_all_priors
													# max_retries: https://botorch.org/api/_modules/botorch/fit.html#fit_gpytorch_model


	# Can't use fit_gpytorch_model because it internally attempts to compute the gradient of the loss w.r.t the hyperparameters, which
	# is not possible in our case. It uses scipy's minimize() and calls it by setting jac=True and it computes the gradients w.r.t the hyperpars
	# by using the objective scipy_objective=_scipy_objective_and_grad()
	# _scipy_objective_and_grad() is called from botorch.optim.fit.fit_gpytorch_scipy()
	# The objective _scipy_objective_and_grad() is defined in botorch.optim.utils


def get_initial_evaluations(dim,eval_type=1):

	if dim == 1:
		if eval_type == 1:
			train_x = torch.tensor([[0.1],[0.3],[0.65],[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
			train_y = torch.tensor([-0.5,2.0,1.0,INF,INF],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
			train_l = torch.tensor([+1, +1, +1, -1, -1],device=device, requires_grad=False, dtype=torch.float32)
		elif eval_type == 2:
			train_x = torch.tensor([[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
			train_y = torch.tensor([INF,INF],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
			train_l = torch.tensor([-1, -1],device=device, requires_grad=False, dtype=torch.float32)
		else:
			train_x = torch.tensor([[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
			train_y = torch.tensor([-0.5,2.0],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
			train_l = torch.tensor([+1, +1],device=device, requires_grad=False, dtype=torch.float32)

		# Put them together:
		train_yl = torch.cat([train_y[:,None], train_l[:,None]],dim=1)
	elif dim == 2:
		branin_fun = Branin(noise_std=None, negate=False)
		train_x = draw_sobol_samples(bounds=torch.tensor(([0.0]*dim,[1.0]*dim)),n=4,q=1).squeeze(1)
		train_y = branin_fun(train_x)
	elif dim == 6:
		hartman_fun = Hartmann()
		train_x = draw_sobol_samples(bounds=torch.tensor(([0.0]*dim,[1.0]*dim)),n=4,q=1).squeeze(1)
		train_y = hartman_fun(train_x)

	return train_x, train_yl

def test_simple(dim):

	train_x, train_yl = get_initial_evaluations(dim=dim)

	Neval = train_x.shape[0]
	dim = train_x.shape[1]

	gpcr = GPCRmodel(train_x=train_x, train_yl=train_yl, noise_std=0.01)

	# Plotting:
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr,None,axes_GPobj=None,axes_acqui=None,axes_fmin=None)
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr,None,axes_GPobj,axes_acqui,axes_fmin,xnext=None,alpha_next=None,block=True)

@hydra.main(config_path="test_classireg_model_cfg.yaml")
def test_optimize(cfg: DictConfig):

	# torch.manual_seed(2)

	train_x, train_yl = get_initial_evaluations(dim=1)

	Neval = train_x.shape[0]
	dim = train_x.shape[1]

	gpcr = GPCRmodel(train_x=train_x, train_yl=train_yl, options=cfg.gpcr_model)

	# Plotting:
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr,None,axes_GPobj=None,axes_acqui=None,axes_fmin=None)
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr,None,axes_GPobj,axes_acqui,axes_fmin,xnext=None,alpha_next=None,block=False)

	logger.info("Pausing...")
	input()

	# Add one more evaluation:
	# train_x_new = torch.tensor([[0.25]])
	train_x_new = torch.tensor([[0.85]])
	train_yl_new = torch.tensor([[1.0, +1]])
	gpcr.update_XY(train_x_new,train_yl_new)

	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr,None,axes_GPobj,axes_acqui,axes_fmin,xnext=None,alpha_next=None,block=True)


@hydra.main(config_path="config/test_classireg_model.yaml")
def test_EIC(cfg: DictConfig):

	train_x, train_yl = get_initial_evaluations(dim=1)

	Neval = train_x.shape[0]
	dim = train_x.shape[1]

	gpcr1 = GPCRmodel(train_x=train_x, train_yl=train_yl, options=cfg.gpcr_model)
	gpcr2 = GPCRmodel(train_x=train_x.clone(), train_yl=train_yl.clone(), options=cfg.gpcr_model)

	model_list = ModelListGP(gpcr1,gpcr2)

	constraints = {1: (None, gpcr2.threshold )}
	# EIC = ConstrainedExpectedImprovement(model=model_list, best_f=0.2, objective_index=0, constraints=constraints)
	eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=constraints, options=cfg.acquisition_function)
	eic_val = eic(torch.tensor([[0.5]]))

	x_next, alpha_next = eic.get_next_point()

	# Plotting:
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr1,eic,axes_GPobj=None,axes_acqui=None,axes_fmin=None)
	axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gpcr1,eic,axes_GPobj,axes_acqui,axes_fmin,xnext=x_next,alpha_next=alpha_next,block=True)

if __name__ == "__main__":

	# test_simple(dim=1)

	# test_optimize()

	test_EIC()

	# test_optimize_pytorch(dim=1)




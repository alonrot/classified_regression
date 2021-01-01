import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from botorch.test_functions.synthetic import Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
from classireg.models.gpcr_model import GPCRmodel
from classireg.models.gpmodel import GPmodel
from classireg.utils.parsing import get_logger
from classireg.utils.plotting_collection import plotting_tool_uncons, plotting_tool_cons
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
import matplotlib.pyplot as plt
from matplotlib import rc



def get_init_evals_obj(eval_type=1):

	# train_x = torch.tensor([[0.1],[0.3],[0.5],[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
	# train_y = torch.tensor([0.5,-0.6,1.0,-1.0,-1.2],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
	train_x = torch.tensor([[0.1],[0.3],[0.5]],device=device, dtype=torch.float32, requires_grad=False)
	train_y = torch.tensor([0.5,-0.6,1.0],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement

	return train_x, train_y

def get_init_evals_cons(eval_type=1):

	train_x = torch.tensor([[0.1],[0.3],[0.5],[0.7],[0.9]],device=device, dtype=torch.float32, requires_grad=False)
	train_y = torch.tensor([-0.75,+1.0,-0.2,INF,INF],device=device, dtype=torch.float32, requires_grad=False) # We place inf to emphasize the absence of measurement
	train_l = torch.tensor([+1, +1, +1, -1, -1],device=device, requires_grad=False, dtype=torch.float32)

	# Put them together:
	train_yl = torch.cat([train_y[:,None], train_l[:,None]],dim=1)

	return train_x, train_yl

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig):

	fontsize_labels = 35
	rc('font', family='serif')
	rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size': fontsize_labels})
	rc('text', usetex=True)
	rc('legend',fontsize=fontsize_labels)


	dim = 1
	train_x_obj, train_y_obj = get_init_evals_obj(eval_type=1)
	train_x_cons, train_yl_cons = get_init_evals_cons(eval_type=1)

	gp_obj = GPmodel(dim=dim, train_X=train_x_obj, train_Y=train_y_obj.view(-1), options=cfg.gpmodel)
	gp_cons = GPCRmodel(dim=dim, train_x=train_x_cons.clone(), train_yl=train_yl_cons.clone(), options=cfg.gpcr_model)
	gp_cons.covar_module.base_kernel.lengthscale = 0.15
	constraints = {1: (None, gp_cons.threshold )}
	model_list = ModelListGP(gp_obj,gp_cons)
	eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=constraints, options=cfg.acquisition_function)

	# Get next point:
	x_next, alpha_next = eic.get_next_point()


	hdl_fig = plt.figure(figsize=(20, 10))
	# hdl_fig.suptitle("Bayesian optimization with unknown constraint and threshold")
	grid_size = (3,1)
	axes_GPobj  = plt.subplot2grid(grid_size, (0,0), rowspan=1,fig=hdl_fig)
	axes_GPcons = plt.subplot2grid(grid_size, (1,0), rowspan=1,fig=hdl_fig)
	# axes_GPcons_prob = plt.subplot2grid(grid_size, (2,0), rowspan=1,fig=hdl_fig)
	axes_acqui  = plt.subplot2grid(grid_size, (2,0), rowspan=1,fig=hdl_fig)


	# Plotting:
	axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj=axes_GPobj,axes_GPcons=axes_GPcons,
																				axes_GPcons_prob=None,axes_acqui=axes_acqui,cfg_plot=cfg.plot,
																				xnext=x_next,alpha_next=alpha_next,plot_eta_c=False)

	
	axes_GPobj.set_xticklabels([])
	axes_GPobj.set_yticks([],[])
	axes_GPobj.set_yticklabels([],[])
	axes_GPobj.set_yticks([0])
	axes_GPobj.set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	
	axes_GPcons.set_yticks([],[])
	axes_GPcons.set_xticklabels([],[])
	axes_GPcons.set_yticks([0])
	axes_GPcons.set_ylabel(r"$g(x)$",fontsize=fontsize_labels)
	
	axes_acqui.set_yticks([],[])
	axes_acqui.set_xticks([0.0,0.5,1.0])
	axes_acqui.set_ylabel(r"$\alpha(x)$",fontsize=fontsize_labels)
	axes_acqui.set_xlabel(r"$x$",fontsize=fontsize_labels)
	plt.pause(0.5)

	save_plot = False
	# save_plot = True
	if save_plot == True:
		logger.info("Saving plot to {0:s} ...".format(cfg.plot.path))
		hdl_fig.tight_layout()
		plt.savefig(fname=cfg.plot.path,dpi=300,facecolor='w', edgecolor='w')
	else:
		plt.show(block=True)


	# pdb.set_trace()

	# # General plotting settings:
	# fontsize = 25
	# fontsize_labels = fontsize + 3
	# from matplotlib import rc
	# import matplotlib.pyplot as plt
	# from matplotlib.ticker import FormatStrFormatter
	# rc('font', family='serif')
	# rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size': fontsize})
	# rc('text', usetex=True)
	# rc('legend',fontsize=fontsize_labels)
	# ylim = [-8,+8]

	# hdl_fig, axes_GPcons = plt.subplots(1,1,figsize=(6, 6))
	# gp_cons.plot(title="",block=False,axes=axes_GPcons,plotting=True,legend=False,Ndiv=100,Nsamples=None,ylim=ylim,showtickslabels_x=False,ylabel=r"$g(x)$")

	# if "threshold" in dir(gp_cons):
	# 	 axes_GPcons.plot([0,1],[gp_cons.threshold.item()]*2,linestyle="--",color="mediumpurple",linewidth=2.0,label="threshold")

	# axes_GPcons.set_xticks([])
	# axes_GPcons.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj,axes_GPcons,
	# 																												axes_GPcons_prob,axes_acqui,cfg.plot,
	# 																												xnext=x_next,alpha_next=alpha_next,Ndiv=100)

	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj=None,axes_GPcons=None,axes_GPcons_prob=None,axes_acqui=None,cfg_plot=cfg.plot,Ndiv=201)
	# axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj,axes_GPcons,axes_GPcons_prob,axes_acqui,cfg.plot,xnext=x_next,alpha_next=alpha_next)

	# Ndiv = 100
	# xpred = torch.linspace(0,1,Ndiv)[:,None]
	# prob_vec = eic.get_probability_of_safe_evaluation(xpred)
	# axes_acqui.plot(xpred.cpu().detach().numpy(),prob_vec.cpu().detach().numpy())
	# import matplotlib.pyplot as plt
	# plt.show(block=True)

if __name__ == "__main__":

	main()



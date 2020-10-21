import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from xsearch.acquisitions.xsearch import qXsearch
from xsearch.models.gpmodel import GPmodel
from botorch.test_functions.synthetic import Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def initialize_logging_variables():
    logvars = dict( mean_bg_list=[],
                    x_bg_list=[],
                    x_next_list=[],
                    alpha_next_list=[],
                    Xevals=[],
                    Yevals=[])
    return logvars

def append_logging_variables(logvars,eta,x_eta,x_next,alpha_next):
    logvars["mean_bg_list"].append(eta.view(1).detach().cpu().numpy())
    logvars["x_bg_list"].append(x_eta.detach().cpu().numpy())
    logvars["x_next_list"].append(x_next.detach().cpu().numpy())
    logvars["alpha_next_list"].append(alpha_next.view(1).detach().cpu().numpy())
    return logvars

def get_initial_evaluations(dim):

    if dim == 1:
        # Initial points used for figure 1 in the paper.
        train_x = torch.Tensor([[0.93452506],
                                 [0.18872502],
                                 [0.89790337],
                                 [0.95841797],
                                 [0.82335255],
                                 [0.45000000],
                                 [0.50000000]])
        train_y = torch.Tensor([-0.4532849,-0.66614552,-0.92803395,0.08880341,-0.27683621,1.000000,1.500000])
    elif dim == 2:
        branin_fun = Branin(noise_std=None, negate=False)
        train_x = draw_sobol_samples(bounds=torch.Tensor(([0.0]*dim,[1.0]*dim)),n=4,q=1).squeeze(1)
        train_y = branin_fun(train_x)
    elif dim == 6:
        hartman_fun = Hartmann()
        train_x = draw_sobol_samples(bounds=torch.Tensor(([0.0]*dim,[1.0]*dim)),n=4,q=1).squeeze(1)
        train_y = hartman_fun(train_x)

    return train_x, train_y


def function_evaluate(xnext,model=None):

    dim = xnext.shape[1]

    if dim == 1:
        predictive = model(xnext)
        y_new = predictive.sample(torch.Size([1]))
    elif dim == 2:
        branin_fun = Branin(noise_std=None, negate=False)
        y_new = branin_fun(xnext)
    elif dim == 6:
        hartman_fun = Hartmann()
        y_new = hartman_fun(xnext)

    return y_new

def plotting_tool(gp,qXs,axes_GPobj,axes_acqui,Ndiv=201,xnext=None,alpha_next=None):

    if gp.dim > 1:
        return None, None

    if axes_GPobj is None and axes_acqui is None:
        hdl_fig_obj, hdl_axes_obj_splots = plt.subplots(2,1,sharex=False,sharey=False,figsize=(10, 7))
        hdl_fig_obj.suptitle("Excursion Search (XS)")
        axes_GPobj = hdl_axes_obj_splots[0]
        axes_acqui = hdl_axes_obj_splots[1]
        gp.plot(title="",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3)
    else:
        qXs.plot(axes=axes_acqui,plotting=True,Ndiv=Ndiv,x_next=xnext,alpha_next=alpha_next,
                    color="darkgreen",ylabel="alpha(x)")
        gp.plot(title="",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3)
        for k in range(len(qXs.u_vec)):
            if k == 0:
                axes_GPobj.plot([0,1],torch.ones(2)*qXs.u_vec[k],linestyle="-",color="mediumpurple",linewidth=0.5,label="fmin samples")
            else:
                axes_GPobj.plot([0,1],torch.ones(2)*qXs.u_vec[k],linestyle="-",color="mediumpurple",linewidth=0.5)
        axes_GPobj.plot(qXs.x_eta.squeeze(),qXs.eta.squeeze(),marker="v",markersize=6,color="darkgreen",linestyle="None",label="min_x mu(x|D)")
        axes_GPobj.legend()
        plt.pause(0.5)

    return axes_GPobj, axes_acqui


def test(dim):

    train_x, train_y = get_initial_evaluations(dim=dim)

    Neval = train_y.shape[0]
    dim = train_x.shape[1]

    gp = GPmodel(train_X=train_x, train_Y=train_y, noise_std=0.01)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    if dim == 1:
        # These initial hyperparameters reproduce Figure 1 in the paper:
        gp.set_hyperparameters(lengthscale=0.05*torch.ones(dim),outputscale=1.0,noise=0.01**2)
    gp.update_hyperparameters_of_model_grad()

    # Initialize acquisition function:
    logger.info("Initializing qXsearch() ...")
    options = dict(Nsamples_fmin=10, Nrestarts_eta=5, Nrestarts=10)
    qXs = qXsearch(model=gp, options=options)

    # Plotting:
    axes_GPobj,axes_acqui = plotting_tool(gp,qXs,axes_GPobj=None,axes_acqui=None)
    
    logvars = initialize_logging_variables()

    # average over multiple trials
    N_TRIALS = 10
    for trial in range(N_TRIALS):
        
        logger.info("Iteration {0:d} / {1:d}".format(trial+1,N_TRIALS))
        
        # Get next point:
        xnext, alpha_next = qXs.get_next_point()

        # Plot only in 1D:
        axes_GPobj,axes_acqui = plotting_tool(gp,qXs,axes_GPobj,axes_acqui,xnext=xnext,alpha_next=alpha_next)
        append_logging_variables(logvars,qXs.eta,qXs.x_eta,qXs.x_next,qXs.alpha_next)
        
        # Collect evaluation at xnext:
        y_new = function_evaluate(xnext,gp)
        logvars["Xevals"].append(xnext.detach().cpu().numpy())
        logvars["Yevals"].append(y_new.view(1))

        # Update GP model:
        train_inputs_new = torch.cat([gp.train_inputs[0], xnext])
        train_targets_new = torch.cat([gp.train_targets, y_new.view(1)])
        del(gp); gp = GPmodel(train_X=train_inputs_new, train_Y=train_targets_new, noise_std=0.01)
        
        logger.info("Fitting model...")
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        # gp.set_hyperparameters(lengthscale=0.05*torch.ones(dim),outputscale=1.0,noise=0.01**2)
        gp.update_hyperparameters_of_model_grad()

        # Update the model in other classes:
        del(qXs); qXs = qXsearch(model=gp, options=options)


if __name__ == "__main__":

    assert len(sys.argv) == 2, " <dim>"
    dim = int(sys.argv[1])
    assert dim in [1,2,6]
    test(dim)






